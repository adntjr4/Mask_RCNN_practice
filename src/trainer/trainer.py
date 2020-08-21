
import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, data_loader, config, special_mode=None):
        self.model = model
        self.data_loader = data_loader
        self.config = config

        self.epoch = self.start_epoch = 0
        self.max_epoch = self.config['train']['max_epoch']

        self._set_criterion()
        self._set_optimizer()

    def train(self):
        self.model.train()

        # load checkpoint to resume
        if self.config['resume']:
            self.load_model()
        
        # training
        for self.epoch in range(self.start_epoch, self.max_epoch):
            # before training 1 epoch

            # train 1 epoch
            losses = self.train_1epoch()

            # after training 1 epoch
            #self.save_model()
            
            self.log_out('[epoch %d]'%self.epoch+1)
            for loss_key in losses:
                self.log_out('%s : %.4f'%(loss_key, losses[loss_key]))

    def train_1epoch(self):
        for data_idx, data in enumerate(self.data_loader):
            # image, size, label, bbox
            image, size, label, bbox = data[0].to(self.device), data[1]

            # forward
            model_out = self.model(image)

            # get losses (Dict)
            losses = self.criterion(model_out, image, anno)

            # backward
            self.optimizer.zero_grad()
            for loss_key in losses:
                losses[loss_key].backward()
            self.optimizer.step()
        
        return losses

    def save_model(self):
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def criterion(self, model_out, image, anno):
        losses = dict()

        # RPN cls
        losses['cls_loss'] = self.cls_criterion()

        # RPN bbox reg
        losses['reg_loss'] = self.reg_criterion()

    def _set_device(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    def _set_criterion(self):
        # RPN losses (cls_loss, reg_loss)
        self.cls_criterion = nn.CrossEntropyLoss() # log loss
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
    
