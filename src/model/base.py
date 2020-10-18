
import torch
import torch.nn as nn
import numpy as np

from src.model.backbone.backbone import BackBone
from src.model.backbone.fpn import FPN
from src.model.rpn.rpn import RPN
from src.model.rpn.rpn_head import RPNHead
from src.util.util import transform_xywh


class BaseModel(nn.Module):
    def __init__(self, conf_model):
        super().__init__()

        self.conf_backbone = conf_model['backbone']
        self.conf_RPN = conf_model['RPN']
        self.RPN_sample_number = conf_model['RPN']['RPN_sample_number']

        self._build_backbone()
        self._build_RPN()

    def _build_backbone(self):
        if 'FPN' in self.conf_backbone:
            self.backbone = FPN(self.conf_backbone)
        else:
            self.backbone = BackBone(self.conf_backbone)

    def _build_RPN(self):
        input_size = self.conf_backbone['input_size']
        self.RPN = RPN( 'FPN' in self.conf_backbone,
                        self.backbone.get_feature_channel(), 
                        self.backbone.get_feature_size(input_size),
                        self.conf_RPN)

    def forward(self, data, mode):
        '''
        Entire model forward function.
        Args:
            data (dict) : input of model
            mode (str) : 'train' or 'eval'
        Returns (train):
            losses
        Returns (eval):
            bbox (Tensor) : [sum(N), 4]
            img_id : [sum(N)]
        '''
        feature_map = self.backbone(data['img'])

        # training mode (return losses)
        if mode == 'train':
            losses = dict()

            # RPN
            proposals, losses = self.RPN(feature_map, data, mode)
            losses.update(losses)

            # RoI head

            return losses

        # evaluation mode (return bboxes, scores ...)
        else:
            # RPN
            proposals = self.RPN(feature_map, data, mode)

            # RoI head
            # bboxes, scores, img_id_map = 여기에 RoI head 넣기

            # transform anchors into original image space 
            # (currently all transformation is done on cpu TODO: transforming with GPU Tensor)
            bboxes_list = []
            for idx in range(len(data['img_id'])):
                transfromed_xywh = transform_xywh(bboxes[idx].cpu(), np.array(data['inv_trans'][idx].cpu()))
                bboxes_list.append(scores.new(transfromed_xywh))
            bboxes = torch.cat(bboxes_list)

            return bboxes, scores, img_id_map

    def get_parameters(self):
        return  list(self.backbone.get_parameters()) + \
                list(self.RPN.parameters())

