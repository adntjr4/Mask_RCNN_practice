import torch
from torchvision.ops import roi_align

feature = torch.arange(125).view(5,5,5).type(torch.float)
bbox = torch.Tensor([[0,0,3,3], [0,0,1,1]])

roi = roi_align(feature.unsqueeze(0), [bbox], output_size=2)

print(roi)