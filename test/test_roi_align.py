import torch
from torchvision.ops import roi_align

features = torch.arange(100).view(2,2,5,5).type(torch.float)

bbox1 = torch.Tensor([[0,0,0.25,0.25], [0,0,5,5]])
bbox2 = torch.Tensor([[0,0,12,12], [4,4,12,12]])

bbox = [bbox1, bbox2]
print(features)

# [N, C, H, W], List([L, 4])
roi = roi_align(features, [bbox1, bbox2], output_size=7, spatial_scale=1, sampling_ratio=-1, aligned=False)
roi = roi.view(2, -1, 2,7,7)
# [N,L,C,2,2]

print(roi.size())
print(roi)

# # [B, C, H, W], List([L, 4])
# roi = []
# for idx, feature in enumerate(features):
#     roi.append(roi_align(feature.unsqueeze(0), [bbox[idx]], output_size=2))
# roi = torch.stack(roi, dim=0)

# # [B,L,C,2,2]

# print(roi.size())
# print(roi)