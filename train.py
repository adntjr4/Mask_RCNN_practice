import argparse, os

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.util.config_parse import ConfigParser
from src.data_set.data_set import DataSet, batch_collate
from src.model.base import BaseModel
from src.trainer.trainer import Trainer


def main(config):
    # gpu number
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

    # data loader
    conf_dl= config['data_loader']
    batch_size, num_workers = conf_dl['batch_size'], conf_dl['num_workers']

    train_data_set = DataSet(conf_dl, mode='val')
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=batch_collate)

    # model
    conf_model = config['model']
    model = BaseModel(conf_model)

    for data in train_data_loader:
        from src.util.util import draw_boxes
        import cv2

        #print(data['bbox'].size())
        #data['bbox'] = data['bbox'][0][:6].unsqueeze(0)
        #print(data['bbox'].size())

        anchor_pos_label, anchor_neg_label, anchor_bbox = model.RPN.get_anchor_label(data, None)
        boxed_img = draw_boxes(data['img'][0], anchor_bbox[anchor_pos_label])

        t = anchor_pos_label
        for i in range(4):
            t = torch.sum(t, dim=0)
        print(t)

        #boxed_img = draw_boxes(data['img'][0], data['bbox'][0])
        cv2.imwrite('data/tmp/tt.jpg', boxed_img)
        exit()
    
    # trainer
    trainer = Trainer(model, train_data_loader, config)

    # train
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-d', '--device', default=None, type=str)
    args.add_argument('-r', '--resume', action='store_true')
    
    args = args.parse_args()

    args.config = 'conf/config.yaml'
    args.device = '2,3'

    assert args.config is not None, 'config file path is needed'

    config = ConfigParser(args)

    main(config)
