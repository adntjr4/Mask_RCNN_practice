from pycocotools.coco import COCO
import torch

t = torch.randn((1,2))
tt = torch.randn((1,2))

print(torch.cat([t, tt], dim=1).size())