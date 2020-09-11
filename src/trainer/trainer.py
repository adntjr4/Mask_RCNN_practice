import os
from os import path

import torch
from torch import nn
from torch import optim
from torch.nn import DataParallel


class Trainer:
    def __init__(self, model, data_loader, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        self.model = DataParallel(model).to(self.device)
        self.data_loader = data_loader
        self.config = config

        self.epoch = self.start_epoch = 0
        self.max_epoch = self.config['train']['max_epoch']

        self.loss_weight = {}
        self.loss_weight['rpn_cls_loss'] = float(self.config['train']['RPN_cls_loss_weight'])
        self.loss_weight['rpn_box_loss'] = float(self.config['train']['RPN_box_loss_weight'])
        
    def train(self):
        self.model.train()

        # load checkpoint to resume
        if self.config['resume']:
            self.load_checkpoint(path.join(self.config['train']['checkpoint_dir'], '{}_checkpoint.pth'.format(self.config.get_model_name())))
            self.log_out('keep training from last checkpoint...')
        else:
            self._set_optimizer()

        # training
        for self.epoch in range(self.epoch, self.max_epoch):
            # before training 1 epoch

            # train 1 epoch
            self.train_1epoch()

            # after training  epoch
            #self.save_checkpoint()

        self.log_out('saving...')
        self.save_checkpoint()

    def train_1epoch(self):
        for idx, data in enumerate(self.data_loader):
            # to device
            cuda_data = {k: v.cuda() for k, v in data.items()}

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
            loss_out_str = '[epoch %03d] %04d/%04d : '%(self.epoch+1, idx+1, len(self.data_loader)) 
            for loss_name in losses:
                loss_out_str += '%s : %.4f / '%(loss_name, losses[loss_name])
            self.log_out(loss_out_str)

    def save_checkpoint(self):
        checkpoint_dir = self.config['train']['checkpoint_dir']
        checkpoint_name = self.config.get_model_name()
        torch.save({'epoch': self.epoch+1,
                    'model_weight': self.model.module.state_dict(),
                    'optimizer': self.optimizer},
                    '%s/%s_checkpoint.pth'%(checkpoint_dir, checkpoint_name))

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
        self.optimizer = optim.Adam(param, lr, betas=(0.5, 0.999))

    def log_out(self, message):
        print(message)
    
