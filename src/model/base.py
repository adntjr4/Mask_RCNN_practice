
import torch
import torch.nn as nn
import numpy as np

from src.model.backbone.backbone import BackBone
from src.model.backbone.fpn import FPN
from src.model.rpn.rpn import RPN
from src.model.rpn.rpn_head import RPNHead
from src.model.box_head.box_head import BoxHead
from src.util.util import transform_xywh_with_img_id


class BaseModel(nn.Module):
    def __init__(self, conf_model):
        super().__init__()

        self.conf_backbone = conf_model['backbone']
        self.conf_RPN = conf_model['RPN']
        self.conf_box = conf_model['box_head']

        self._build_backbone()
        self._build_RPN()
        self._build_box_head()

    def _build_backbone(self):
        if 'FPN' in self.conf_backbone:
            self.backbone = FPN(self.conf_backbone)
        else:
            self.backbone = BackBone(self.conf_backbone)

    def _build_RPN(self):
        self.RPN = RPN( 'FPN' in self.conf_backbone,
                        self.backbone.get_feature_channel(), 
                        self.backbone.get_feature_size(self.conf_backbone['input_size']),
                        self.conf_RPN)

    def _build_box_head(self):
        self.box_head = BoxHead('FPN' in self.conf_backbone,
                                self.backbone.get_feature_channel(),
                                self.conf_backbone['input_size'],
                                self.conf_box)

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

        proposals, rpn_losses = self.RPN(feature_map, data, mode)

        # training mode (return losses)
        if mode == 'train':
            losses = dict()

            # RPN loss update
            losses.update(rpn_losses)

            # RoI head forward
            box_head_losses = self.box_head(feature_map, proposals, data, mode)
            losses.update(box_head_losses)

            return losses

        # evaluation mode (return bboxes, scores ...)
        else:
            # RoI head forward
            bboxes, scores, img_id_map = self.box_head(feature_map, proposals, data, mode)

            # transform anchors into original image space 
            bboxes = transform_xywh_with_img_id(bboxes, img_id_map, data['inv_trans'], data['img_id']).cpu()

            return bboxes, scores, img_id_map

    def get_parameters(self):
        return  list(self.backbone.get_parameters()) + \
                list(self.RPN.parameters()) + \
                list(self.box_head.parameters())

