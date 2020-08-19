import argparse

import torch
import numpy as np

from src.util.config_parse import ConfigParser
from src.data_loader import data_loader
from src.model.base import BaseModel


def main(config):
    # data loader
    train_data_set = data_loader.CocoDataSet(config, mode='val')

    # model
    model = BaseModel()

    # trainer
    

    # train


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-r', '--resume', action='store_true')
    args.add_argument('-d', '--device', default=None, type=str)
    args = args.parse_args()

    args.config = 'conf/config.yaml'

    assert args.config is not None, 'config file path is needed'

    config = ConfigParser(args)

    main(config)
