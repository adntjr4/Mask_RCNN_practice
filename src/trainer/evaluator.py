import json

import torch
from torch.nn import DataParallel
from pycocotools.cocoeval import COCOeval
import cv2

from src.util.util import transform_xywh, draw_boxes


class Evaluator():
    def __init__(self, model, data_loader, data_set, config):
        self.model = DataParallel(model).cuda()
        self.data_loader = data_loader
        self.data_set = data_set
        self.config = config

        self.result_file_dir = './data/saved/result/result.json'

        self.reset_results()

    def eval(self):
        # detection prediction
        self.detection_prediction()

        # file out
        with open(self.result_file_dir, 'w') as json_file:
            json.dump(self.detection_results, json_file)

        # load result
        cocoDt = self.data_set.coco.loadRes(self.result_file_dir)

        # run evaluation
        cocoEval = COCOeval(self.data_set.coco, cocoDt, iouType='bbox')
        cocoEval.params.catIds = 1
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def detection_prediction(self):
        self.log_out('predicting...')

        self.model.train()

        for idx, data in enumerate(self.data_loader):
            # to device
            cuda_data = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    cuda_data[k] = v.cuda()

            img_id = data['img_id'][0]
            inv_trans = data['inv_trans'][0]

            # get prediction
            classes, bboxes, scores = self.model(cuda_data, mode='eval')
            classes, bboxes, scores = classes.detach().cpu(), bboxes.detach().cpu(), scores.detach().cpu()
            inverse_bboxes = transform_xywh(bboxes, inv_trans)
            self.save_one_image_detection(img_id, classes, inverse_bboxes, scores)

            if (idx+1) % 10 == 0:
                self.log_out('prediction [%04d/%04d]'%(idx+1, self.data_loader.__len__()))

        self.log_out('end prediction.')
    
    def reset_results(self):
        self.detection_results = []

    def save_one_image_detection(self, image_id, category, bboxes, scores):
        '''
        Args:
            image_id (int)
            category (Tensor)
            bboxes (Tensor)
            scores (Tensor)
        '''
        for cat, bbox, score in zip(category, bboxes, scores):
            result = {
                'image_id': image_id,
                'category_id': cat.item(),
                'bbox': bbox.tolist(),
                'score': score.item()
            }
            self.detection_results.append(result)

    def image_out(self, img, img_name, classes, bboxes):
        draw_img = draw_boxes(img, bboxes)
        cv2.imwrite('data/tmp/%s'%(img_name), draw_img)

    def log_out(self, message):
        print(message)
