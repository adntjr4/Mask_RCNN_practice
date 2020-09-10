import os
from os import path

import torch
from torch import nn
from torch import optim


class Trainer:
    def __init__(self, model, data_loader, config, loss_switch=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

        if loss_switch is None:
            self.loss_switch = {'rpn_cls':True, 'rpn_box':True}
        else:
            self.loss_switch = loss_switch

        self.model = model
        self.model.cuda()
        self.data_loader = data_loader
        self.config = config

        self.epoch = self.start_epoch = 0
        self.max_epoch = self.config['train']['max_epoch']

        self._set_criterion()
        
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
            #if self.epoch % 10 == 0:
            #    self.save_checkpoint()

        self.log_out('saving...')
        self.save_checkpoint()

    def train_1epoch(self):
        for idx, data in enumerate(self.data_loader):
            # to device
            cuda_data = {k: v.cuda() for k, v in data.items()}

            # forward
            model_out = self.model(cuda_data['img'])

            # get losses (return dict)
            losses = self.criterion(model_out, cuda_data)

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
                    'model_weight': self.model.state_dict(),
                    'optimizer': self.optimizer},
                    '%s/%s_checkpoint.pth'%(checkpoint_dir, checkpoint_name))

    def load_checkpoint(self, file_name):
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        self.model.load_state_dict(saved_checkpoint['model_weight'])
        self.optimizer = saved_checkpoint['optimizer']

    def criterion(self, model_out, gt):
        losses = dict()

        cls_score = model_out['rpn_cls_score'] # List([B, 1*k, H, W])
        bbox_pred = model_out['rpn_bbox_pred'] # List([B, 4*k, H, W])
        sample_number = self.config['train']['RPN_sample_number']

        reg_loss_weight = self.config['train']['RPN_reg_loss_weight']

        # [RPN] anchor labeling
        anchor_info = self.model.RPN.get_anchor_label(gt['bbox'], cls_score, bbox_pred)

        # [RPN] class loss
        if self.loss_switch['rpn_cls']:
            selected_cls_out, label = self.model.RPN.get_cls_output_target(anchor_info['cls_score'], anchor_info['anchor_label'], sample_number)
            losses['rpn_cls_loss'] = self.rpn_cls_criterion(selected_cls_out, label)

        # [RPN] bbox regression loss
        if self.loss_switch['rpn_box']:
            predicted_t, calculated_t = self.model.RPN.get_box_output_target(gt['bbox'], anchor_info['origin_anchors'], anchor_info['bbox_pred'], anchor_info['anchor_label'], anchor_info['closest_gt'])
            losses['rpn_box_loss'] = self.rpn_box_criterion(predicted_t, calculated_t) * reg_loss_weight

        return losses

    def _set_criterion(self):
        # RPN losses (cls_loss, reg_loss)
        self.rpn_cls_criterion = nn.BCELoss() # log loss
        self.rpn_box_criterion = nn.SmoothL1Loss() # robust loss

        # and something more...

    def _set_optimizer(self):
        # trainable network parameter select
        param = list(self.model.RPN.parameters())
        
        # hyperparameters
        lr = self.config['train']['lr']
        
        # optimizer select
        self.optimizer = optim.Adam(param, lr, betas=(0.5, 0.999))

    def log_out(self, message):
        print(message)
    
