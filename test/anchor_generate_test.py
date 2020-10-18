import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.model.anchor_func import generate_anchor_form
from src.util.debugger import debug_draw_bbox

anchor_size = [32, 64, 128, 256, 512]
anchor_ratio = [0.5, 1.0, 2.0]
feature_size = [(64, 64)]
image_size = (1024, 1024)

anchor = []
for size in anchor_size:
    for ratio in anchor_ratio:
        anchor.append([size, ratio])
anchor = [anchor]

anchors = generate_anchor_form(anchor, feature_size, image_size)

debug_draw_bbox('./data/coco/val2017/000000000139.jpg', anchors[0][:, 31, 31, :], 'center_anchor')
