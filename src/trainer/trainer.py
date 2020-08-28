import os

import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, data_loader, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

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
            self.load_checkpoint('%s/%s_checkpoint.pth'%(self.config['train']['checkpoint_dir'], self.config.get_model_name()))
            self.log_out('keep training from last checkpoint...')
        else:
            self._set_optimizer()
        
        # training
        for self.epoch in range(self.epoch, self.max_epoch):
            # before training 1 epoch

            # train 1 epoch
            losses = self.train_1epoch()

            # after training  epoch
            #if self.epoch % 10 == 0:
            #    self.save_checkpoint()
            
            self.log_out('[epoch %d]'%(self.epoch+1))
            for loss_name in losses:
                self.log_out('%s : %.4f'%(loss_name, losses[loss_name]))

        print("testing")
        self.save_checkpoint()
        # image out
        for data in self.data_loader:
            model_out = self.model(data['img'].cuda())
            RoI_bbox = self.model.RPN.get_proposed_RoI(model_out['cls_out'], model_out['reg_out'], self.config['evaluate']['RPN_cls_threshold'])
            from src.util.util import draw_boxes
            import cv2
            boxed_img = draw_boxes(data['img'][0], RoI_bbox)
            cv2.imwrite('data/tmp/RPN_test.jpg', boxed_img)

    def train_1epoch(self):
        running_losses = {'cls_loss':0.0, 'reg_loss':0.0}
        for data in self.data_loader:
            # to device
            cuda_data = dict()
            for data_name in data:
                cuda_data[data_name] = data[data_name].cuda()

            # forward
            model_out = self.model(cuda_data['img'])

            # get losses (return dict)
            losses = self.criterion(model_out, cuda_data)

            # backward
            self.optimizer.zero_grad()
            for loss_name in losses:
                losses[loss_name].backward(retain_graph=True)
                running_losses[loss_name] += losses[loss_name].item()
            self.optimizer.step()
        
        for loss_name in running_losses:
            running_losses[loss_name] /= self.data_loader.__len__()

        return running_losses

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

        cls_out = model_out['cls_out'] # [B, 2*k, H, W]
        reg_out = model_out['reg_out'] # [B, 4*k, H, W]
        sample_number = self.config['train']['RPN_sample_number']

        reg_loss_weight = self.config['train']['RPN_reg_loss_weight']

        # RPN
        anchor_label = self.model.RPN.get_anchor_label(gt, reg_out)

        # RPN class
        selected_cls_out, label = self.model.RPN.RPN_label_select(cls_out, anchor_label, sample_number)
        losses['cls_loss'] = self.cls_criterion(selected_cls_out, label)

        # RPN bbox regression
        if anchor_label['highest_gt'].size()[0] != 0:
            predicted_t, calculated_t = self.model.RPN.RPN_cal_t_regression(reg_out, gt, anchor_label)
            losses['reg_loss'] = self.reg_criterion(predicted_t, calculated_t) * reg_loss_weight

        return losses

    def _set_criterion(self):
        # RPN losses (cls_loss, reg_loss)
        self.cls_criterion = nn.BCELoss() # log loss
        self.reg_criterion = nn.SmoothL1Loss() # robust loss

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
    
