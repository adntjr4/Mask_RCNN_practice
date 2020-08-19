
import torch
import torch.nn as nn

from backbone import BackBone
from rpn import RPN

class BaseModel(nn.Module):
    def __init__(self, backbone: str='R50'):
        super().__init__()

        self._build_backbone(backbone)
        self._build_RPN()

    def _build_backbone(self, backbone):
        self.backbone_model = BackBone(backbone)

    def _build_RPN(self):
        c, w, h = self.backbone_model.get_channel()
        self.RPN = RPN(c, 256)

    def forward(self, x):
        feature_map = self.backbone_model(x)
        RoIs = self.RPN(feature_map)
        return RoIs


if __name__ == "__main__":
    bm = BaseModel()
    rn = torch.randn(1, 3, 1024, 1024)
    cls_score, reg_coor = bm(rn)
    print(cls_score.size())
    print(reg_coor.size())



