import argparse, os

from torch.utils.data import DataLoader

from src.util.util import load_model
from src.util.config_parse import ConfigParser
from src.data_set.data_set import DataSet, batch_collate
from src.model.base import BaseModel
from src.trainer.evaluator import Evaluator


def main(config):
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']

    # data loader
    conf_dl= config['data_loader']
    
    num_workers = conf_dl['num_workers']

    batch_size = conf_dl['batch_size'] if config['batch_size'] == -1 else config['batch_size']
    test_data_set = DataSet(conf_dl, mode='val', human_only=True)
    test_data_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=config['shuffle'], num_workers=num_workers, collate_fn=batch_collate)

    # model
    model = BaseModel(config['model'])
    model.load_state_dict(load_model(config['weight']))

    # evaluator
    evaluator = Evaluator(model, test_data_loader, test_data_set, config)

    # image out
    evaluator.image_prediction()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-w', '--weight', default=None)
    args.add_argument('-d', '--device', default='0', type=str)
    args.add_argument('-b', '--batch_size', default=-1, type=int)
    args.add_argument('--shuffle', action='store_true')
    args.add_argument('-n', '--img_num', default=1, type=int)
    args.add_argument('--img_name', default='test', type=str)
    
    args = args.parse_args()

    args.weight = 'data/saved/checkpoint/R50_human_checkpoint.pth'

    assert args.config is not None, 'config file path is needed'
    assert args.weight is not None, 'model weight is needed for evaluation'

    config = ConfigParser(args)

    main(config)
