import os
from os import path

import torch
from torch import nn
from torch import optim
from torch.nn import DataParallel

from src.util.timer import Timer
from src.util.progress_msg import ProgressMsg


class Trainer:
    def __init__(self, model, data_loader, config, human_only=False):
        self.model = DataParallel(model).cuda()
        self.data_loader = data_loader
        self.config = config
        self.human_only = human_only

        self.epoch = self.start_epoch = 0
        self.max_epoch = self.config['train']['max_epoch']

        self.loss_weight = {}
        self.loss_weight['rpn_cls_loss'] = float(self.config['train']['RPN_cls_loss_weight'])
        self.loss_weight['rpn_box_loss'] = float(self.config['train']['RPN_box_loss_weight'])

        self.checkpoint_name = self.config.get_model_name()
        if self.human_only:
            self.checkpoint_name += '_human'
        self.checkpoint_name += '_checkpoint.pth'

        self.timer = Timer()

        self.progress_msg = ProgressMsg((self.max_epoch, len(self.data_loader)))

        self.log_out_iter = 10
        
    def train(self):
        self.model.train()

        # load checkpoint to resume
        if self.config['resume']:
            self.load_checkpoint(path.join(self.config['train']['checkpoint_dir'], self.checkpoint_name))
            self.log_out('keep training from last checkpoint...')
        else:
            self._set_optimizer()

        

        # schduler
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.max_epoch*0.75, gamma=0.1)

        self.progress_msg.start((self.epoch, 0))

        # training
        for self.epoch in range(self.epoch, self.max_epoch):
            # before training 1 epoch

            # train 1 epoch
            self.train_1epoch()

            # after training  epoch
            self.save_checkpoint()

            # scheduler stop
            self.scheduler.step()
        self.progress_msg.print_finish_msg()

        self.log_out('saving model...')
        self.save_checkpoint()

    def warmup(self):
        raise NotImplementedError

    def train_1epoch(self):
        avg_loss = {'rpn_cls_loss':0., 'rpn_box_loss':0. }
        for idx, data in enumerate(self.data_loader):
            #self.timer.data_load_end()
            # to device
            cuda_data = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    cuda_data[k] = v.cuda()

            # get losses (return dict)
            #self.timer.model_start()
            losses = self.model(cuda_data, mode='train')
            losses = {k:losses[k].mean() for k in losses}
            #self.timer.model_end()

            # loss weight
            for loss_key in losses:
                losses[loss_key] *= self.loss_weight[loss_key]

            # backward
            self.optimizer.zero_grad()
            total_loss = sum(v for v in losses.values())
            total_loss.backward()
            self.optimizer.step()

            # print loss
            for loss_name in losses:
                avg_loss[loss_name] += losses[loss_name].item()


            if (idx+1) % self.log_out_iter == 0:
                loss_out_str = '[epoch %02d/%02d] %04d/%04d : '%(self.epoch+1, self.max_epoch, idx+1, len(self.data_loader)) 
                #loss_out_str += '(data:%.02fs, model:%.02fs), '%(self.timer.data_load_time, self.timer.model_time)
                for loss_name in avg_loss:
                    loss_out_str += '%s : %.4f / '%(loss_name, avg_loss[loss_name]/self.log_out_iter)
                    avg_loss[loss_name] = 0.
                self.progress_msg.line_reset()
                self.log_out(loss_out_str)

            self.progress_msg.print_prog_msg((self.epoch, idx))

            #self.timer.data_load_start()

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
    
