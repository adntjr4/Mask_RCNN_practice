
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
        self.backbone_model.requires_grad_ = False

    def _build_RPN(self):
        threshold = self.conf_RPN['positive_threshold'], self.conf_RPN['negative_threshold']
        self.RPN = RPN( self.backbone_model.get_channel(self.conf_RPN['input_size']),
                        256,
                        self.conf_RPN['input_size'],
                        threshold=threshold,
                        reg_weight=self.conf_RPN['reg_weight'],
                        nms_threshold=self.conf_RPN['nms_threshold'] )

    def forward(self, img):
        model_out = dict()

        feature_map = self.backbone_model(img)

        RPN_out = self.RPN(feature_map)
        model_out.update(RPN_out)


        return model_out
