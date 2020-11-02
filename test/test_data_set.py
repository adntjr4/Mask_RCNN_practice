import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse

from torch.utils.data import DataLoader

from src.data_set.data_set import DataSet, batch_collate
from src.util.config_parse import ConfigParser
from src.util.debugger import debug_draw_bbox_cv_img
from src.util.util import transform_xywh


def main(args):
    batch_num = 4
    times = 25

    config = ConfigParser(args)
    data_set = DataSet(config['data_loader'], mode='val', human_only=True)

    # single load
    single_loader = DataLoader(data_set, batch_size=1, collate_fn=batch_collate)
    idx = 0
    for data in single_loader:
        # gt
        img = data_set.get_original_img_from_id(data['img_id'][0].item())
        bboxes = transform_xywh(data['bbox'][0], data['inv_trans'][0])
        # img = data['img'][0].permute(1,2,0).numpy()[:,:,[2,1,0]]
        # bboxes = data['bbox'][0]
        debug_draw_bbox_cv_img(img, bboxes, f'single{idx}.jpg')
        idx += 1
        
        if idx >= batch_num*times:
            break

    # batch load
    # batch_loader = DataLoader(data_set, batch_size=batch_num, collate_fn=batch_collate)
    # idx = 0
    # for data in batch_loader:
    #     for img_idx, data in enumerate(zip(data['img_id'], data['bbox'], data['inv_trans'])):
    #         img_id, bbox, inv_trans = data
    #         # gt
    #         img = data_set.get_original_img_from_id(img_id.item())
    #         bboxes = transform_xywh(bbox, inv_trans)
    #         debug_draw_bbox_cv_img(img, bboxes, f'batch{idx*batch_num+img_idx}.jpg')
    #     idx += 1
    #     if idx >= times:
    #         break

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default=None, type=str)
    args.add_argument('-d', '--device', default=None, type=str)
    args.add_argument('-r', '--resume', action='store_true')

    args = args.parse_args()

    args.config = 'conf/resnet_cfg.yaml'

    main(args)