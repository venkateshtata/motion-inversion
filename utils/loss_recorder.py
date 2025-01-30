from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import os


class SingleLoss:
    def __init__(self, name: str, writer: SummaryWriter, base=0):
        self.name = name
        self.loss_step = []
        self.loss_epoch = []
        self.loss_epoch_tmp = []
        self.writer = writer
        if base:
            self.loss_epoch = [0] * base
            self.loss_step = [0] * base * 10

    def add_event(self, val, step=None, name='scalar'):
        if step is None: step = len(self.loss_step)
        if val is None:
            val = 0
        else:
            callee = getattr(self.writer, 'add_' + name)
            callee(self.name + '_step', val, step)
        self.loss_step.append(val)
        self.loss_epoch_tmp.append(val)

    def epoch(self, step=None):
        if step is None: step = len(self.loss_epoch)
        loss_avg = sum(self.loss_epoch_tmp) / len(self.loss_epoch_tmp)
        self.loss_epoch_tmp = []
        self.loss_epoch.append(loss_avg)
        self.writer.add_scalar('Train/epoch_' + self.name, loss_avg, step)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        loss_step = np.array(self.loss_step)
        loss_epoch = np.array(self.loss_epoch)
        np.save(path + self.name + '_step.npy', loss_step)
        np.save(path + self.name + '_epoch.npy', loss_epoch)

    def last_epoch(self):
        return self.loss_epoch[-1]


class LossRecorder:
    def __init__(self, writer: SummaryWriter, base=0):
        self.losses = {}
        self.writer = writer
        self.base = base

    def report_scalar(self, group_name, name, iteration=None, value=0.):
        name = group_name + '/' + name
        if isinstance(value, torch.Tensor): value = value.item()
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer, self.base)
        self.losses[name].add_event(value, iteration, 'scalar')

    def add_figure(self, name, val, step=None):
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer, self.base)
        self.losses[name].add_event(val, step, 'figure')

    def verbose(self):
        lst = {}
        for key in self.losses.keys():
            lst[key] = self.losses[key].loss_step[-1]
        lst = sorted(lst.items(), key=lambda x: x[0])
        return str(lst)

    def epoch(self, step=None):
        for loss in self.losses.values():
            loss.epoch(step)

    def save(self, path):
        for loss in self.losses.values():
            loss.save(path)

    def last_epoch(self):
        res = []
        for loss in self.losses.values():
            res.append(loss.last_epoch())
        return res
