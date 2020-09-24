import json

import torch
from torch.nn import DataParallel
from pycocotools.cocoeval import COCOeval
import cv2

from src.util.util import transform_xywh, draw_boxes


class Evaluator():
    def __init__(self, model, data_loader, data_set, config, human_only=False):
        self.model = DataParallel(model).cuda()
        self.data_loader = data_loader
        self.data_set = data_set
        self.config = config
        self.human_only = human_only

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
        cocoEval.params.catIds = self.data_set.coco.getCatIds(catNms=['person'])
        cocoEval.params.imgIds = self.data_set.img_id_list[:self.data_set.__len__()]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def detection_prediction(self):
        self.log_out('predicting...')

        self.model.eval()

        with torch.no_grad():
            for data_idx, data in enumerate(self.data_loader):
                # to device
                cuda_data = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        cuda_data[k] = v.cuda()

                # get prediction
                bboxes, scores, img_id_map = self.model(cuda_data, mode='eval')

                # record prediction
                self.save_one_image_detection(bboxes, scores, img_id_map)

                if (data_idx+1) % 10 == 0:
                    self.log_out('prediction [%04d/%04d]'%(data_idx+1, self.data_loader.__len__()))

            self.log_out('end prediction.')
    
    def reset_results(self):
        self.detection_results = []

    def image_prediction(self):
        self.log_out('predicting...')

        self.model.eval()

        with torch.no_grad():
            for data_idx, data in enumerate(self.data_loader):
                # to device
                cuda_data = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        cuda_data[k] = v.cuda()

                # get prediction
                bboxes, _, _ = self.model(cuda_data, mode='eval')

                # record prediction
                img = self.data_set.get_original_img_from_id(data['img_id'].item())
                self.image_out(img, 'test%d.jpg'%data_idx, bboxes)

            self.log_out('end prediction.')

    def save_one_image_detection(self, bboxes, scores, img_ids):
        '''
        Args:
            bboxes (Tensor) : [sum(N0, 4]
            scores (Tensor) : [sum(N)]
            img_ids (Tensor) : [sum(N)]
        '''
        for bbox, score, img_id in zip(bboxes, scores, img_ids):
            result = {
                'image_id': img_id.item(),
                'category_id': 1,
                'bbox': bbox.tolist(),
                'score': score.item()
            }
            #img = self.data_set.get_original_img_from_id(img_id.item())
            #self.image_out(img, 'aaaaaa.jpg', [bbox])
            self.detection_results.append(result)

    def image_out(self, img, img_name, bboxes):
        draw_img = draw_boxes(img, bboxes)
        cv2.imwrite('data/tmp/%s'%(img_name), draw_img)

    def log_out(self, message):
        print(message)
