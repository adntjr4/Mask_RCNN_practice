
import torch
import torch.nn as nn

from src.model.backbone.backbone import BackBone
from src.model.backbone.fpn import FPN
from src.model.rpn.rpn import RPN
from src.model.rpn.rpn_head import RPNHead

class BaseModel(nn.Module):
    def __init__(self, conf_model):
        super().__init__()

        self.conf_backbone = conf_model['backbone']
        self.conf_RPN = conf_model['RPN']

        self._build_backbone()
        self._build_RPN()

    def _build_backbone(self):
        if 'FPN' in self.conf_backbone:
            self.backbone_model = FPN(self.conf_backbone)
        else:
            self.backbone_model = BackBone(self.conf_backbone)

    def _build_RPN(self):
        input_size = self.conf_backbone['input_size']
        self.RPN = RPN( 'FPN' in self.conf_backbone,
                        self.backbone_model.get_feature_channel(), 
                        self.backbone_model.get_feature_size(input_size),
                        self.conf_RPN)

    def forward(self, img):
        model_out = dict()

        feature_map = self.backbone_model(img)

        RPN_out = self.RPN(feature_map)
        model_out.update(RPN_out)


        return model_out
