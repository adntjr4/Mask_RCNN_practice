import os
from os import path

import torch
from torch import nn
from torch import optim
from torch.nn import DataParallel

from src.util.timer import Timer
from src.util.progress_msg import ProgressMsg
from src.util.logger import Logger
from src.util.schduler import WarmupLRScheduler


class Trainer:
    def __init__(self, model, data_loader, config, human_only=False):
        self.model = DataParallel(model).cuda()
        self.data_loader = data_loader
        self.config = config
        self.human_only = human_only

        self.epoch = self.start_epoch = 0
        self.max_epoch = self.config['train']['max_epoch']

        self.warmup_iter = 1000

        self.loss_weight = {}
        self.loss_weight['rpn_obj'] = float(self.config['train']['RPN_objectness_loss_weight'])
        self.loss_weight['rpn_reg'] = float(self.config['train']['RPN_regression_loss_weight'])
        self.loss_weight['box_obj'] = float(self.config['train']['box_objectness_loss_weight'])
        self.loss_weight['box_reg'] = float(self.config['train']['box_regression_loss_weight'])

        self.checkpoint_name = self.config.get_model_name()
        if self.human_only:
            self.checkpoint_name += '_human'
        self.checkpoint_name += '_checkpoint.pth'

        self.timer = Timer()

        #self.logger = Logger()
        self.progress_msg = ProgressMsg((self.max_epoch, len(self.data_loader)))

        self.log_out_iter = 10

        self.avg_loss = {'total':0., 'rpn_obj':0., 'rpn_reg':0., 'box_obj':0., 'box_reg':0. }
        
    def train(self):
        self.model.train()

        # load checkpoint to resume
        if self.config['resume']:
            self.load_checkpoint(path.join(self.config['train']['checkpoint_dir'], self.checkpoint_name))
            self.log_out('keep training from last checkpoint...')
        else:
            self._set_optimizer()

        # schduler
        #self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.max_epoch*0.75, gamma=0.1)

        self.progress_msg.start((self.epoch, 0))

        # warmup
        if self.config['warmup']:
            self.warmp_scheduler = WarmupLRScheduler(self.optimizer, warmup_iter=self.warmup_iter)
            self.warmup()

        # training
        for self.epoch in range(self.epoch, self.max_epoch):
            # before training 1 epoch

            # train 1 epoch
            self.train_1epoch()

            # after training  epoch
            self.save_checkpoint()

            # scheduler stop
            #self.scheduler.step()
        self.progress_msg.print_finish_msg()

        self.log_out('saving model...')
        self.save_checkpoint()

    def warmup(self):
        self.log_out('start warmup')
        current_warmup_iter = 0
        for idx, data in enumerate(self.data_loader):
            # to device
            cuda_data = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    cuda_data[k] = v.cuda()

            # get losses (return dict)
            losses = self.model(cuda_data, mode='train')
            losses = {k:losses[k].mean() for k in losses}

            # loss weight
            for loss_key in losses:
                losses[loss_key] *= self.loss_weight[loss_key]

            # backward
            self.optimizer.zero_grad()
            total_loss = sum(v for v in losses.values())
            total_loss.backward()
            self.optimizer.step()

            current_warmup_iter += 1
            self.warmp_scheduler.step()

            # print loss
            for loss_key in losses:
                self.avg_loss['total'] += losses[loss_key].item()
                self.avg_loss[loss_key] += losses[loss_key].item()

            if (idx+1) % self.log_out_iter == 0:
                loss_out_str = '[warmup] %04d/%04d - '%(current_warmup_iter, self.warmup_iter)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                loss_out_str += 'lr : %.6f | '%(current_lr)

                for loss_name in self.avg_loss:
                    loss_out_str += '%s : %.4f | '%(loss_name, self.avg_loss[loss_name]/self.log_out_iter)
                    self.avg_loss[loss_name] = 0.
                self.progress_msg.line_reset()
                self.log_out(loss_out_str)

            self.progress_msg.print_prog_msg((self.epoch, idx))

            if current_warmup_iter >= self.warmup_iter:
                self.log_out('end warmup')
                break

    def train_1epoch(self):
        for idx, data in enumerate(self.data_loader):
            # to device
            cuda_data = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    cuda_data[k] = v.cuda()

            # get losses (return dict)
            losses = self.model(cuda_data, mode='train')
            losses = {k:losses[k].mean() for k in losses}

            # loss weight
            for loss_key in losses:
                losses[loss_key] *= self.loss_weight[loss_key]

            # backward
            self.optimizer.zero_grad()
            total_loss = sum(v for v in losses.values())
            total_loss.backward()
            self.optimizer.step()

            # print loss
            for loss_key in losses:
                self.avg_loss['total'] += losses[loss_key].item()
                self.avg_loss[loss_key] += losses[loss_key].item()

            if (idx+1) % self.log_out_iter == 0:
                loss_out_str = '[epoch %02d/%02d] %05d/%05d - '%(self.epoch+1, self.max_epoch, idx+1, len(self.data_loader))
                
                current_lr = self.optimizer.param_groups[0]['lr']
                loss_out_str += 'lr : %.6f | '%(current_lr)

                for loss_name in self.avg_loss:
                    loss_out_str += '%s : %.4f | '%(loss_name, self.avg_loss[loss_name]/self.log_out_iter)
                    self.avg_loss[loss_name] = 0.
                self.progress_msg.line_reset()
                self.log_out(loss_out_str)

            self.progress_msg.print_prog_msg((self.epoch, idx))

    def save_checkpoint(self):
        torch.save({'epoch': self.epoch+1,
                    'model_weight': self.model.module.state_dict(),
                    'optimizer': self.optimizer},
                    path.join(self.config['train']['checkpoint_dir'], self.checkpoint_name))

    def load_checkpoint(self, file_name):
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        self.model.module.load_state_dict(saved_checkpoint['model_weight'])
        self.optimizer = saved_checkpoint['optimizer']

    def _set_optimizer(self):
        # trainable network parameter select
        param = self.model.module.get_parameters()
        
        # hyperparameters
        lr = self.config['train']['lr']
        
        # optimizer select
        optimizer = self.config['train']['optimizer']
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(param, lr, betas=(0.5, 0.999))
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(param, lr, momentum=0.9, weight_decay=1e-4)

    def log_out(self, message):
        print(message)
    
