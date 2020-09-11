import argparse, os

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.util.config_parse import ConfigParser
from src.data_set.data_set import DataSet, batch_collate
from src.model.base import BaseModel
from src.trainer.trainer import Trainer
from src.trainer.trainer_human import TrainerHuman


def main(config):
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

    # data loader
    conf_dl= config['data_loader']
    batch_size, num_workers = conf_dl['batch_size'], conf_dl['num_workers']

    train_data_set = DataSet(conf_dl, mode='train', human_only=True)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=batch_collate)

    # model
    conf_model = config['model']
    model = BaseModel(conf_model)

    # trainer
    trainer = TrainerHuman(model, train_data_loader, config)

    # train
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-d', '--device', default=None, type=str)
    args.add_argument('-r', '--resume', action='store_true')
    
    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'

    config = ConfigParser(args)

    main(config)
