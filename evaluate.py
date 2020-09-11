import argparse, os

import cv2
import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import numpy as np

from src.util.util import load_model, img_process, draw_boxes
from src.util.config_parse import ConfigParser
from src.data_set.data_set import DataSet, batch_collate
from src.model.base import BaseModel
from src.trainer.trainer import Trainer


def main(config):
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

    # model
    conf_model = config['model']
    model = BaseModel(conf_model)
    model.load_state_dict(load_model(config['weight']))
    model.eval()
    model.cuda()

    # input image load
    img = cv2.imread(config['input'])
    input_size = tuple(config['data_loader']['input_size'])
    img_tensor, _ = img_process(img, input_size)
    img_tensor = img_tensor.cuda()
    img_data = {'img':img_tensor.unsqueeze(0)}

    # get RoI bbox
    model_out = model(img_data, 'eval')
    RoI_bbox = model.RPN.region_proposal_threshold(model_out['rpn_cls_score'], model_out['rpn_bbox_pred'], config['model']['RPN']['proposal_threshold'])

    boxed_img = draw_boxes(img_tensor.cpu(), RoI_bbox.cpu())
    cv2.imwrite('data/tmp/RPN_evaluation.jpg', boxed_img)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-w', '--weight', default=None)
    args.add_argument('-d', '--device', default=None, type=str)
    args.add_argument('-i', '--input',  default=None)
    
    args = args.parse_args()

    args.weight = 'data/saved/model/R50_human_epoch5.pth'
    args.input = 'data/coco/val2017/000000397133.jpg'

    assert args.config is not None, 'config file path is needed'
    assert args.weight is not None, 'model weight is needed for evaluation'

    config = ConfigParser(args)

    main(config)

'''
0: 000000397133.jpg
1: 000000037777.jpg
2: 000000252219.jpg
3: 000000087038.jpg
4: 000000174482.jpg
5: 000000403385.jpg
6: 000000006818.jpg
7: 000000480985.jpg
8: 000000458054.jpg
9: 000000331352.jpg
'''
