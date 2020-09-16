from os import path

from pycocotools.coco import COCO
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as tr
import numpy as np


from src.util.util import img_process, transform_xywh


class DataSet(data.Dataset):
    def __init__(self, config, mode='test', human_only=False):

        assert mode in ['train', 'val', 'test']

        self.config = config
        self.mode = mode
        self.human_only = human_only

        self.data_dir = self.config['data_dir']
        self.data_type = '%s2017'%mode
        self.json_dir = self.data_dir + '/' + self.config['%s_instance'%mode]

        # init coco object
        self.coco = COCO(self.json_dir)
        self.img_id_list = self.coco.getImgIds()

        self.input_size = tuple(self.config['input_size'])

    def __getitem__(self, index):
        '''
        N : number of object
        Returns:
            sample_dict {'img': Tensor[C, H, W], 'img_size': Tensor[2] (H, W), 'label': Tensor[N] (int), 'bbox': Tensor[N, 4] (float)}
                To see final return batch form, see more 'dectection_collate()'
        '''
        # open image
        img_object = self.coco.loadImgs(self.img_id_list[index])[0]
        img = cv2.imread(path.join(self.data_dir, self.data_type, img_object['file_name']))
        
        img_tensor, trans, inv_trans = img_process(img, self.input_size)

        # parse image annotation
        img_size = torch.Tensor([img_object['height'], img_object['width']])
        if self.human_only:
            cat_ids = self.coco.getCatIds(catNms=['person'])
            ann_ids = self.coco.getAnnIds(imgIds=self.img_id_list[index], catIds=cat_ids)
        else:
            ann_ids = self.coco.getAnnIds(imgIds=self.img_id_list[index])
        anns = self.coco.loadAnns(ann_ids)

        label, bbox = list(), list()
        for ann in anns:
            label.append(ann['category_id'])
            bbox.append(ann['bbox'])
        bbox = transform_xywh(bbox, trans)

        item = {'img': img_tensor, 'img_id': self.img_id_list[index], 'img_size': img_size, 'label': torch.Tensor(label), 'bbox': torch.Tensor(bbox), 'inv_trans': inv_trans}

        return item

    def get_original_img_from_index(self, index):
        img_object = self.coco.loadImgs(self.img_id_list[index])[0]
        return cv2.imread(path.join(self.data_dir, self.data_type, img_object['file_name']))

    def get_original_img_from_id(self, img_id):
        img_object = self.coco.loadImgs(img_id)[0]
        return cv2.imread(path.join(self.data_dir, self.data_type, img_object['file_name']))

    def __len__(self):
        return len(self.img_id_list)

def batch_collate(samples):
    '''
    Returns:
        sample_dict {'image': Tensor[B, C, H, W] (float), 'img_id': List[B], 'img_size': Tensor[B, 2] (float), 'label': Tensor[B, N]) (int), 'bbox': Tensor[B, N, 4]) (float), 'inv_trans' : List()}
    '''
    images = [sample['img'] for sample in samples]
    img_id = [sample['img_id'] for sample in samples]
    img_sizes =  [sample['img_size'] for sample in samples]
    labels = [sample['label'] for sample in samples]
    bboxes = [sample['bbox'] for sample in samples]
    inv_trans = [sample['inv_trans'] for sample in samples]

    images_tensor = torch.stack(images)
    img_sizes_tensor = torch.stack(img_sizes)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    padded_bboxes = torch.nn.utils.rnn.pad_sequence(bboxes, batch_first=True)

    return {'img': images_tensor, 'img_id': img_id, 'img_size': img_sizes_tensor, 'label': padded_labels, 'bbox': padded_bboxes, 'inv_trans': inv_trans}
