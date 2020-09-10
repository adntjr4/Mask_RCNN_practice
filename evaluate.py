import argparse, os

import cv2
import torch
from torch.utils.data import DataLoader
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

    # get RoI bbox
    model_out = model(img_tensor.unsqueeze(0))
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

    args.weight = 'data/saved/checkpoint/R50_checkpoint.pth'
    args.input = 'data/coco/val2017/000000397133.jpg'

    assert args.config is not None, 'config file path is needed'
    assert args.weight is not None, 'model weight is needed for evaluation'

    config = ConfigParser(args)

    main(config)
