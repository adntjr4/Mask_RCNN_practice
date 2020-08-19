from pathlib import Path
import json

import torch
import torch.utils.data as data
import torchvision.transforms as tr
from PIL import Image, ImageDraw 

from src.util.util import draw_boxes


class CocoDataSet(data.Dataset):
    def __init__(self, config, mode='test'):

        assert mode in ['train', 'val', 'test']

        self.config = config['data_loader']
        self.mode = mode

        self.data_path = Path(self.config['data_dir']) / self.config['%s_data'%mode]
        if mode != 'test':
            self.anno_path = Path(self.config['data_dir']) / self.config['%s_instance'%mode]
            with open(self.anno_path, 'r') as json_f:
                self.anno_file = json.load(json_f)

    def _get_attr(self, index, key):
        return self.anno_file['images'][index][key]

    def __getitem__(self, index):
        img_path = self.data_path / self._get_attr(index, 'file_name')
        img = Image.open(img_path).convert('RGB')

        anno_list = []
        for anno in self.anno_file['annotations']:
            if anno['image_id'] == self._get_attr(0, 'id'):
                anno_list.append(anno)

        return tr.ToTensor(img), anno_list

    def __len__(self):
        return len(self.anno_file['images'])

    # for debugging
    def save_img(self, index, bbox=True):
        img_path = self.data_path / self._get_attr(index, 'file_name')
        img = Image.open(str(img_path)).convert('RGB')
        img_id = self._get_attr(0, 'id')

        boxes = []
        for anno in self.anno_file['annotations']:
            if anno['image_id'] == img_id:
                boxes.append(anno['bbox'])
        
        draw_boxes(img, boxes)

        img.save('data/tmp/tmp.jpg')

