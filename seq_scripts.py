import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import torch.distributed as dist


def seq_train(loader, model, optimizer, device, epoch_idx, recoder, rank, scaler=None):
    """
    Args:
        loader: DataLoader for training.
        model: DDP-wrapped model.
        optimizer: The optimizer.
        device: The device for this process (e.g., torch.device('cuda:0')).
        epoch_idx: Current epoch number.
        recoder: Logger.
        rank: The rank of the current process.
        world_size: The total number of processes.
        scaler: GradScaler for mixed precision training.
    """
    model.train()
    loss_value = []
    # clr = [group['lr'] for group in optimizer.optimizer.param_groups]

    # Use DistributedSampler's epoch to ensure data shuffling is different for each epoch
    loader.sampler.set_epoch(epoch_idx)

    # Wrap tqdm with condition to show progress bar only on rank 0
    data_iterator = tqdm(loader) if rank == 0 else loader

    for batch_idx, data in enumerate(data_iterator):
        vid = data[0].to(device)
        vid_lgt = data[1].to(device)
        label = data[2].to(device)
        label_lgt = data[3].to(device)

        optimizer.zero_grad()

        # In DDP, the autocast() context should be handled inside the model's forward pass.
        # So we can remove the 'with autocast():' block here.
        ret_dict = model(vid)
        loss = model.module.criterion_calculation(ret_dict, label,
                                                  label_lgt)  # Use .module to access the original method

        if np.isinf(loss.item()) or np.isnan(loss.item()):
            if rank == 0:
                print('loss is nan')
                print(str(data[1]) + ' frames')
                print(str(data[3]) + ' glosses')
            # For DDP, all processes need to call backward()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer.optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.optimizer.step()

        loss_value.append(loss.item())

        if rank == 0 and batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item()))

        del ret_dict
        del loss

    # Wait for all processes to finish before moving to the next epoch
    dist.barrier()

    # Log mean loss on rank 0
    if rank == 0:
        recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))

    optimizer.scheduler.step()
    return loss_value


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder, evaluate_tool="python", rank=0):
    """
    Args:
        rank: The rank of the current process. Evaluation will only run on rank 0.
    """
    python_eval = True if evaluate_tool == "python" else False
    model.eval()
    local_sent = []
    local_info = []
    local_conv = []

    data_iterator = tqdm(loader) if rank == 0 else loader
    for batch_idx, data in enumerate(data_iterator):
        recoder.record_timer("device")
        vid = data[0].to(device)
        vid_lgt = data[1].to(device)
        label = data[2].to(device)
        label_lgt = data[3].to(device)

        with torch.no_grad():
            ret_dict = model(vid)

        local_info += [file_name['fileid'] for file_name in data[-1]]
        local_sent += ret_dict['recognized_sents']
        local_conv += ret_dict['conv_sents']

        # ---------- gather results ----------
    world_size = dist.get_world_size()
    gathered_info = [None for _ in range(world_size)]
    gathered_sent = [None for _ in range(world_size)]
    gathered_conv = [None for _ in range(world_size)]

    dist.all_gather_object(gathered_info, local_info)
    dist.all_gather_object(gathered_sent, local_sent)
    dist.all_gather_object(gathered_conv, local_conv)

    ret = None
    if rank == 0:
        # flatten
        total_info = sum(gathered_info, [])
        total_sent = sum(gathered_sent, [])
        total_conv = sum(gathered_conv, [])

        write2file(f"{work_dir}output-hypothesis-{mode}.ctm", total_info, total_sent)
        write2file(f"{work_dir}output-hypothesis-{mode}-conv.ctm", total_info, total_conv)

        conv_ret = evaluate(
            prefix=work_dir, mode=mode,
            output_file=f"output-hypothesis-{mode}-conv.ctm",
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            python_evaluate=python_eval,
        )

        lstm_ret = evaluate(
            prefix=work_dir, mode=mode,
            output_file=f"output-hypothesis-{mode}.ctm",
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            python_evaluate=python_eval,
            triplet=True,
        )

        # try:
        #     python_eval = True if evaluate_tool == "python" else False
        #
        #     write2file(f"{work_dir}output-hypothesis-{mode}.ctm", total_info, total_sent)
        #     write2file(f"{work_dir}output-hypothesis-{mode}-conv.ctm", total_info, total_conv)
        #
        #     conv_ret = evaluate(
        #         prefix=work_dir, mode=mode,
        #         output_file=f"output-hypothesis-{mode}-conv.ctm",
        #         evaluate_dir=cfg.dataset_info['evaluation_dir'],
        #         evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
        #         python_evaluate=python_eval,
        #     )
        #
        #     lstm_ret = evaluate(
        #         prefix=work_dir, mode=mode,
        #         output_file=f"output-hypothesis-{mode}.ctm",
        #         evaluate_dir=cfg.dataset_info['evaluation_dir'],
        #         evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
        #         python_evaluate=python_eval,
        #         triplet=True,
        #     )
        # except Exception as e:
        #     print("Unexpected error during evaluation:", e)
        #     lstm_ret, conv_ret = 100.0, 100.0

        ret = min(conv_ret, lstm_ret)
        recoder.print_log(f"Epoch {epoch}, {mode} {ret:2.2f}%", f"{work_dir}/{mode}.txt")

    return ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder, rank=0):
    """
    Args:
        rank: The rank of the current process. Generation will only run on rank 0.
    """
    if rank != 0:
        return

    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = data[0].to(device)
        vid_lgt = data[1].to(device)
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)

        if not os.path.exists(src_path):
            os.makedirs(src_path)

        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])

    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))