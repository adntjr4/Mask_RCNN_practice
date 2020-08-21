
import torch
import torch.nn as nn

from src.model.backbone import BackBone
from src.model.rpn import RPN

class BaseModel(nn.Module):
    def __init__(self, conf_model):
        super().__init__()

        self.conf_backbone = conf_model['backbone']
        self.conf_RPN = conf_model['RPN']

        self._build_backbone()
        self._build_RPN()

    def _build_backbone(self):
        self.backbone_model = BackBone(self.conf_backbone['backbone_type'])

    def _build_RPN(self):
        self.RPN = RPN(self.backbone_model.get_channel(), 256, tr_thres=self.conf_RPN['train_region_threshold'])

    def forward(self, x):
        feature_map = self.backbone_model(x)
        RPN_score = self.RPN(feature_map)
        return RPN_score
