import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from torch.cuda.amp import GradScaler
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation
from torch.cuda.amp import autocast as autocast


class Processor():
    def __init__(self, arg, rank, world_size):
        self.arg = arg
        self.rank = rank
        self.world_size = world_size

        # Only rank 0 handles file system operations to prevent race conditions
        if self.rank == 0:
            if not os.path.exists(self.arg.work_dir):
                os.makedirs(self.arg.work_dir)
            shutil.copy2(__file__, self.arg.work_dir)
            shutil.copy2('./configs/baseline.yaml', self.arg.work_dir)
            shutil.copy2('./modules/tconv.py', self.arg.work_dir)
            shutil.copy2('./modules/resnet.py', self.arg.work_dir)
            if not os.path.exists(self.arg.work_dir + '/openai/'):
                shutil.copytree('./modules/openai/', self.arg.work_dir + '/openai/')

        # Ensure all processes sync before continuing
        dist.barrier()

        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)

        # Correctly set device for DDP
        self.device = torch.device(f'cuda:{self.rank}')
        torch.cuda.set_device(self.device)

        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1

        self.model, self.optimizer = self.loading()
        self.scaler = GradScaler()
        # self.scaler = None

    def start(self):
        if self.arg.phase == 'train':
            best_dev = 100.0
            best_epoch = 0
            total_time = 0
            epoch_time = 0
            if self.rank == 0:
                self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args['num_epoch']):

                save_model = False
                eval_model = epoch % self.arg.eval_interval == 0
                epoch_time = time.time()

                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder, self.rank, self.scaler)

                if eval_model:  # Only evaluate on rank 0
                    dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder, self.arg.evaluate_tool, self.rank)
                    if self.rank == 0:
                        self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                        if dev_wer < best_dev:
                            best_dev = dev_wer
                            best_epoch = epoch
                            model_path = "{}_best_model.pt".format(self.arg.work_dir)
                            self.save_model(epoch, model_path)
                            self.recoder.print_log('Save best model')
                        self.recoder.print_log('Best_dev: {:05.2f}, Epoch : {}'.format(best_dev, best_epoch))
                    if save_model:
                        model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer, epoch)
                        seq_model_list.append(model_path)
                        print("seq_model_list", seq_model_list)
                        self.save_model(epoch, model_path)

                dist.barrier()  # Sync all processes after each epoch

                epoch_time = time.time() - epoch_time
                total_time += epoch_time
                if self.rank == 0:
                    self.recoder.print_log(
                        'Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time) // 60, int(epoch_time) % 60))
            if self.rank == 0:
                self.recoder.print_log('Training costs {} hours {} mins {} seconds'.format(int(total_time) // 60 // 60,
                                                                                           int(total_time) // 60 % 60,
                                                                                           int(total_time) % 60))
        elif self.arg.phase == 'test':
            if self.rank == 0:
                if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                    print('Please appoint --weights.')
                self.recoder.print_log('Model:    {}.'.format(self.arg.model))
                self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))

            # Use rank 0 for evaluation to avoid redundant computation
            dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device, "dev", 6667,
                               self.arg.work_dir, self.recoder, self.arg.evaluate_tool, self.rank)
            test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device, "test", 6667,
                                self.arg.work_dir, self.recoder, self.arg.evaluate_tool, self.rank)
            if self.rank == 0:
                self.recoder.print_log('Evaluation Done.\n')
                self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                self.recoder.print_log("Dev WER: {:05.2f}%".format(test_wer))
            dist.barrier()  # Sync all processes after evaluation
        elif self.arg.phase == "features":
            if self.rank == 0:
                for mode in ["train", "dev", "test"]:
                    seq_feature_generation(
                        self.data_loader[mode + "_eval" if mode == "train" else mode],
                        self.model, self.device, mode, self.arg.work_dir, self.recoder
                    )
            dist.barrier()

    def save_arg(self):
        if self.rank == 0:
            arg_dict = vars(self.arg)
            if not os.path.exists(self.arg.work_dir):
                os.makedirs(self.arg.work_dir)
            with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
                yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        if self.rank == 0:
            # save only the state dict from the underlying model
            state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
                'rng_state': self.rng.save_rng_state(),
            }, save_path)

    def adjust_lr(self, model):
        # ... (unchanged)
        normal_weight = []
        normal_bias = []
        no_grad_weight = []
        no_grad_bias = []
        for name, m in model.named_modules():
            if 'resblocks_no_grad' in name:
                if len(list(m.parameters())) > 0:
                    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d) or isinstance(m,
                                                                                                      torch.nn.BatchNorm2d) or isinstance(
                            m, torch.nn.Conv1d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m,
                                                                                                     torch.nn.BatchNorm3d) or isinstance(
                            m, torch.nn.Linear) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.MultiheadAttention):
                        if len(list(m.parameters())) > 0:
                            ps = list(m.parameters())
                            no_grad_weight.append(ps[0])
                            if len(ps) == 2:
                                no_grad_bias.append(ps[1])
                    elif len(list(m.parameters())) > 0 and len(m._modules) == 0:
                        raise ValueError(
                            "New atomic module type: {}. Need to give it a learning policy".format(type(m)))
            elif len(list(m.parameters())) > 0 and len(m._modules) == 0:
                ps = list(m.parameters())
                if len(ps) == 1:
                    normal_weight.append(ps[0])
                elif len(ps) == 2:
                    normal_weight.append(ps[0])
                    normal_bias.append(ps[1])
                else:
                    print(f'assign lr for unusual network components : {type(m)}')
                    normal_weight.extend(ps)
            elif len(list(m.parameters())) > 0:
                pass
        return [{'params': no_grad_weight, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "no_grad_weight"},
                {'params': no_grad_bias, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "no_grad_bias"},
                {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "normal_weight"},
                {'params': normal_bias, 'lr_mult': 1, 'decay_mult': 1,
                 'name': "normal_bias"},
                ]

    def freeze_parameter(self, model):
        for name, param in model.conv2d.named_parameters():
            if 'ln_post' not in name and 'ada_weight' not in name and\
                    'Adapter' not in name and 'prefix_embedding' not in name and\
                    'aggblocks' not in name and name.split('.')[-1]!='proj' and\
                    name.split('.')[-1]!='proj_cls' and name.split('.')[-1]!='query':

            # if 'Adapter' not in name and 'prefix_embedding' not in name and 'taggblocks' not in name and\
            #         'lora' not in name and 'visual_projection' not in name and name.split('.')[-1]!='proj':
                param.requires_grad = False
        if self.rank == 0:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"Name: {name}, Shape: {param.shape}")

    def loading(self):
        if self.rank == 0:
            print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        if self.rank == 0:
            shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)

        # Freeze parameters before moving to device and wrapping with DDP
        self.freeze_parameter(model)
        model = convert_model(model)

        # Move model to the correct device
        model = model.to(self.device)

        # Wrap model with DDP
        model = DDP(model, device_ids=[self.rank])

        # Convert model after DDP wrap if needed
        # model = convert_model(model)

        # Use model.module.parameters() for optimizer
        optimizer = utils.Optimizer(model.module.parameters(), self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)

        self.kernel_sizes = model.module.conv1d.kernel_size  # Access through .module
        if self.rank == 0:
            print("Loading model finished.")

        self.load_data()
        return model, optimizer

    def load_model_weights(self, model, weight_path):
        # Load weights on a single device, then broadcast
        map_location = {'cuda:0': f'cuda:{self.rank}'}
        state_dict = torch.load(weight_path, map_location=map_location)

        if self.rank == 0:
            if len(self.arg.ignore_weights):
                for w in self.arg.ignore_weights:
                    if state_dict.pop(w, None) is not None:
                        print('Successfully Remove Weights: {}.'.format(w))
                    else:
                        print('Can Not Remove Weights: {}.'.format(w))

        # Access the underlying model with .module
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        model.module.load_state_dict(weights, strict=True)
        dist.barrier()  # Sync all processes after loading weights

    @staticmethod
    def modified_weights(state_dict, modified=False):
        # ... (unchanged)
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        # ... (unchanged)
        # Load weights on a single device, then broadcast
        map_location = {'cuda:0': f'cuda:{self.rank}'}
        state_dict = torch.load(self.arg.load_checkpoints, map_location=map_location)

        if self.rank == 0:
            if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
                print("Loading random seeds...")
            if "optimizer_state_dict" in state_dict.keys():
                print("Loading optimizer parameters...")
            if "scheduler_state_dict" in state_dict.keys():
                print("Loading scheduler parameters...")

        # Access the underlying model with .module
        model.module.load_state_dict(state_dict['model_state_dict'], strict=True)
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        # optimizer.to(self.device)  # No need for this line with DDP

        # Make sure to set rng state for all ranks
        self.rng.set_rng_state(state_dict['rng_state'])

        if "scheduler_state_dict" in state_dict.keys():
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        if self.rank == 0:
            self.recoder.print_log(f"Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")
        dist.barrier()

    def load_data(self):
        if self.rank == 0:
            print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        if self.rank == 0:
            shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)

        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "dev", "test"], [True, False, False])
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "dev", "test"], [True, False, False])

        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size=self.kernel_sizes,
                                             dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], train_flag)

            if self.rank == 0:
                print("Loading data finished.")
                # 确保所有进程都同步，因为rank 0加载了数据而其他进程没有
            dist.barrier()

    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed) + worker_id)

    def build_dataloader(self, dataset, train_flag):
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=train_flag)
        return torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=self.arg.batch_size,
            num_workers=self.arg.num_worker,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.feeder.collate_fn,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


def main_worker(rank, world_size, args):
    setup_distributed(rank, world_size)
    processor = Processor(args, rank, world_size)

    # Pack code only on the main process
    if rank == 0:
        utils.pack_code("./", args.work_dir)
    dist.barrier()  # Sync

    processor.start()
    dist.destroy_process_group()


def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()

    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)

    args = sparser.parse_args()

    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training.")

    # Use torch.multiprocessing.spawn to launch processes
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args), join=True)