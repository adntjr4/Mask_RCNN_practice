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

        self.config = config
        self.mode = mode

        self.data_path = Path(self.config['data_dir']) / self.config['%s_data'%mode]
        if mode != 'test':
            self.anno_path = Path(self.config['data_dir']) / self.config['%s_instance'%mode]
            with open(self.anno_path, 'r') as json_f:
                self.anno_file = json.load(json_f)

        resize = config['resize']
        self.img_tr = tr.Compose([ tr.Resize((resize, resize)),
                                   tr.ToTensor() 
                                   ])

    def _get_attr(self, index, key):
        return self.anno_file['images'][index][key]

    def __getitem__(self, index):
        '''
        N : number of object
        Returns:
            sample_dict {'image': Tensor[H, W], 'size':list[2], 'label': Tensor[N], 'bbox': Tensor[N,4]} + segmentation (Tensor[N, ?]) (NOT IMPLEMENTED)
                To see final return of batch, see more 'dectection_collate()'
        '''
        # open image
        img_path = self.data_path / self._get_attr(index, 'file_name')
        img = Image.open(img_path).convert('RGB')

        # parse image annotation
        label_list, bbox_list = [], []
        for anno in self.anno_file['annotations']:
            if anno['image_id'] == self._get_attr(index, 'id'):
                label_list.append(anno['category_id'])
                bbox_list.append(anno['bbox'])
        sample_dict = {'image': self.img_tr(img), 'size': list(img.size), 'label': torch.Tensor(label_list), 'bbox': torch.Tensor(bbox_list)}
        
        return sample_dict

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

def dectection_collate(samples):
    '''
    Returns:
        sample_dict {'image': Tensor[B, H, W] (float), 'size': Tensor[B, 2] (float), 'label': Tensor[B, N] (int), 'bbox': Tensor[B, N, 4] (float)}
    '''
    images = [sample['image'] for sample in samples]
    sizes =  [list(reversed(sample['size'])) for sample in samples]
    labels = [sample['label'] for sample in samples]
    bboxes = [sample['bbox'] for sample in samples]
    images_tensor = torch.stack(images)
    sizes_tensor = torch.Tensor(sizes)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    padded_bboxes = torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True)
    return {'image': images_tensor, 'size': sizes_tensor, 'label': padded_labels, 'bbox': padded_bboxes}
