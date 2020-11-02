import json

import torch
from torch.nn import DataParallel
from pycocotools.cocoeval import COCOeval
import cv2

from src.util.util import transform_xywh, draw_boxes_with_score


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

    @torch.no_grad()
    def detection_prediction(self):
        self.log_out('predicting...')

        self.model.eval()

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

    @torch.no_grad()
    def image_prediction(self):
        self.log_out('predicting...')

        self.model.eval()

        img_num = 0
        for data in self.data_loader:
            # to device
            cuda_data = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    cuda_data[k] = v.cuda()

            # get prediction
            bboxes, scores, img_id_map = self.model(cuda_data, mode='eval')

            # setting image names
            image_names = []
            batch_size, _ = data['img_id'].size()
            for img_idx in range(batch_size):
                if img_num < self.config['img_num']:
                    img_name = self.config['img_name']
                    image_names.append(f'{img_name}{img_num}.jpg')
                else:
                    image_names.append('')
                img_num += 1

            self.save_one_image_detection_as_image(data['img_id'], image_names, bboxes, scores, img_id_map)

            if img_num >= self.config['img_num']:
                break

        self.log_out('end prediction.')

    def save_one_image_detection_as_image(self, img_ids, image_names, bboxes, scores, img_id_map):
        '''
        save result as image with bbox
        Args:
            img_ids (Tensor) : [B]
            image_names (List)
            bboxes (Tensor) : [sum(N), 4]
            scores (Tensor) : [sum(N)]
            img_id_map (Tensor) : [sum(N)]
        '''
        batch_size, _ = img_ids.size()

        assert len(image_names) == batch_size

        for batch_idx in range(batch_size):
            # gather corresponding bboxes and scores
            bboxes_list, scores_list = [], []
            for bbox, score, img_id in zip(bboxes, scores, img_id_map):
                if img_ids[batch_idx].item() == img_id.item():
                    bboxes_list.append(bbox.cpu().tolist())
                    scores_list.append(score.cpu().tolist())

            # save image
            if image_names[batch_idx] != '':
                img = self.data_set.get_original_img_from_id(img_ids[batch_idx].item())
                self.image_out(img, image_names[batch_idx], bboxes_list, scores_list)
                self.log_out('%s >> saved'%(image_names[batch_idx]))

    def save_one_image_detection(self, bboxes, scores, img_id_map):
        '''
        save result in class list
        Args:
            bboxes (Tensor) : [sum(N), 4]
            scores (Tensor) : [sum(N)]
            img_id_map (Tensor) : [sum(N)]
        '''
        for bbox, score, img_id in zip(bboxes, scores, img_id_map):
            result = {
                'image_id': img_id.item(),
                'category_id': 1,
                'bbox': bbox.tolist(),
                'score': score.item()
            }
            #img = self.data_set.get_original_img_from_id(img_id.item())
            #self.image_out(img, 'aaaaaa.jpg', [bbox])
            self.detection_results.append(result)

    def image_out(self, img, img_name, bboxes, scores):
        draw_img = draw_boxes_with_score(img, bboxes, scores)
        cv2.imwrite('data/tmp/%s'%(img_name), draw_img)

    def log_out(self, message):
        print(message)
