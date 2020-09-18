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

        self.model.eval()

        with torch.no_grad():
            for data_idx, data in enumerate(self.data_loader):
                # to device
                cuda_data = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        cuda_data[k] = v.cuda()

                # get prediction
                anchor, cls_score, proposal_map = self.model(cuda_data, mode='eval')
                bboxes = [anchor[map_idx][one_map].detach().cpu() for map_idx, one_map in enumerate(proposal_map)]
                scores = [cls_score[map_idx][one_map].detach().cpu() for map_idx, one_map in enumerate(proposal_map)]
                
                for img_idx, img_id in enumerate(data['img_id']):
                    inverse_bboxes = transform_xywh(bboxes[img_idx], data['inv_trans'][img_idx])
                    self.save_one_image_detection(img_id, inverse_bboxes, scores[img_idx])

                if (data_idx+1) % 10 == 0:
                    self.log_out('prediction [%04d/%04d]'%(data_idx+1, self.data_loader.__len__()))

            self.log_out('end prediction.')
    
    def reset_results(self):
        self.detection_results = []

    def save_one_image_detection(self, image_id, bboxes, scores):
        '''
        Args:
            image_id (int)
            bboxes (Tensor)
            scores (Tensor)
        '''
        for bbox, score in zip(bboxes, scores):
            result = {
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox.tolist(),
                'score': score.item()
            }
            self.detection_results.append(result)

    def image_out(self, img, img_name, classes, bboxes):
        draw_img = draw_boxes(img, bboxes)
        cv2.imwrite('data/tmp/%s'%(img_name), draw_img)

    def log_out(self, message):
        print(message)
