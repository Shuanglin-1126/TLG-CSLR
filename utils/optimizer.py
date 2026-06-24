import pdb
import torch
import numpy as np
import math
import torch.optim as optim


class Optimizer(object):
    def __init__(self, params, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.optim_dict['base_lr'],
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            alpha = self.optim_dict['learning_ratio']
            self.optimizer = optim.Adam(
                # [
                #     {'params': model.conv2d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.conv1d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.rnn.parameters()},
                #     {'params': model.classifier.parameters()},
                # ],
                # model.conv1d.fc.parameters(),
                params,#params,
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'AdamW':
            self.optimizer = optim.AdamW(
                params,
                lr=self.optim_dict['base_lr'],
                weight_decay=self.optim_dict['weight_decay'],
                eps=self.optim_dict['eps'],
                betas=(self.optim_dict['beta1'], self.optim_dict['beta2']))
        else:
            raise ValueError()
        # self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.cos_scheduler)

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
            return lr_scheduler
        else:
            raise ValueError()

    def cos_scheduler(self, current_epoch):
        warmup_epochs = self.optim_dict['warm_epoch']
        warmup_start_lr = self.optim_dict['warm_lr']
        base_lr = self.optim_dict['base_lr']
        max_epoch = self.optim_dict['num_epoch']
        if current_epoch < warmup_epochs:
            # 线性 warmup，从 warmup_start_lr 到 base_lr
            return (warmup_start_lr + (base_lr - warmup_start_lr) * current_epoch / warmup_epochs) / base_lr
        else:
            # cosine decay，从 base_lr 衰减到 0
            cosine_epoch = current_epoch - warmup_epochs
            cosine_total = max_epoch - warmup_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * cosine_epoch / cosine_total))
            return cosine_decay

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
