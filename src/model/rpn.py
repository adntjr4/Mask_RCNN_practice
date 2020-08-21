
import torch
import torch.nn.functional as F
import torch.nn as nn


class RPN(nn.Module):
    def __init__(self, in_shape, inter_channel, tr_thres=0.7):
        super().__init__()

        in_channel, self.in_H, self.in_W = in_shape

        self._set_anchor()
        self.num_anchor = len(self.anchor)

        self.inter_conv = nn.Conv2d(in_channel, inter_channel, kernel_size=3, padding=1)
        self.cls_conv = nn.Conv2d(inter_channel, self.num_anchor, kernel_size=1)
        self.reg_conv = nn.Conv2d(inter_channel, 4*self.num_anchor, kernel_size=1)

    def _set_anchor(self):
        self.anchor = []
        for size in [128, 256, 512]:
            for ratio in [0.5, 1., 2.]:
                self.anchor.append([size, ratio])

    def forward(self, features):
        inter_feature = F.relu(self.inter_conv(features))
        cls_score = self.cls_conv(inter_feature)
        reg_coor = self.reg_conv(inter_feature)

        return cls_score, reg_coor

    #def get_gt_anchor(self, gt_instance)

    def get_anchor_label(self, gt, threshold:float=0.7):
        '''
        B: batch size
        N: number of object
        k: number of anchor
        W: width of anchor feature map
        H: height of anchor feature map

        Args:
            gt (Tuple) : (img_size, label, bbox)
                img_size (Tensor) : [B, 2] (HW)
                label (Tensor) : [B, N]
                bbox (Tensor) : [B, N, 4]
            threshold (float)
               
        Returns:
            anchor_label (Tensor) : [B, k, H, W]
            anchor_bbox (Tensor) : [B, k, H, W, 4] # xywh
        '''
        img_size, label, bbox = gt
        batch_size = img_size.size()[0]
        object_num = label.size()[1]

        # tranform anchor to origin image domain
        BkHW = batch_size * self.num_anchor * self.in_H * self.in_W
        anchor_label = torch.zeros((BkHW), dtype=torch.bool) # [B * k * H * W]
        anchor_bbox_list = []
        for anc in self.anchor:
            loc_y = torch.arange(self.in_H) # [H]
            loc_x = torch.arange(self.in_W) # [W]

            x = torch.stack([loc_x]*self.in_H, dim=0) # [H, W]
            x = torch.stack([x]*batch_size, dim=0) # [B, H, W]
            y = torch.stack([loc_y]*self.in_W, dim=1) # [H, W]
            y = torch.stack([y]*batch_size, dim=0) # [B, H, W]

            w = torch.ones((batch_size, self.in_H, self.in_W)) * anc[0]
            h = torch.ones((batch_size, self.in_H, self.in_W)) * anc[0] * anc[1]
                
            anchor_bbox_t = torch.stack((x,y,w,h), dim=3) # [B, H, W, 4]
            anchor_bbox_list.append(anchor_bbox_t)
        anchor_bbox = torch.stack(anchor_bbox_list) # [k, B, H, W, 4]
        anchor_bbox = anchor_bbox.permute((1,0,2,3,4)) # [B, k, H, W, 4]

        # calculate IoU
        anchor_bbox_ext = torch.stack([anchor_bbox]*object_num, dim=0) # [N, B, k, H, W, 4]
        bbox_ext = torch.stack([bbox]*self.in_W, dim=2) # [B, N, W, 4]
        bbox_ext = torch.stack([bbox_ext]*self.in_H, dim=2) # [B, N, H, W, 4]
        bbox_ext = torch.stack([bbox_ext]*self.num_anchor, dim=2) # [B, N, k, H, W, 4]
        bbox_ext = bbox_ext.permute((1,0,2,3,4,5)) # [N, B, k, H, W, 4]

        cross_IoU = IoU(bbox_ext, anchor_bbox_ext).view((object_num, BkHW)) # [N, B * k * H * W]

        # label highest anchor
        _, highest_indices = torch.max(cross_IoU, dim=1) # [N]
        one_hot_label = F.one_hot(highest_indices, num_classes=BkHW) # [N, B * k * H * W]
        anchor_label += torch.sum(one_hot_label, dim=0, dtype=torch.bool) # [B * k * H * W]

        # label over threshold anchor
        anchor_label += torch.sum((cross_IoU > threshold), dim=0, dtype=torch.bool) # [B * k * H * W]

        anchor_label = anchor_label.view((batch_size, self.num_anchor, self.in_H, self.in_W)) # [B, k, H, W]
        
        return anchor_label, anchor_bbox


def IoU(xywh0:torch.Tensor, xywh1:torch.Tensor):
    assert xywh0.size() == xywh1.size(), 'for calculate IoU, size of two tensor must be same.'

    x0, y0, w0, h0 = xywh0.split(1, dim = len(xywh0.size())-1)
    x1, y1, w1, h1 = xywh1.split(1, dim = len(xywh0.size())-1)

    x0_, y0_ = x0+w0, y0+h0
    x1_, y1_ = x1+w1, y1+h1
    
    U_x, U_y, U_x_, U_y_ =  torch.max(x0, x1),   \
                            torch.max(y0, y1),   \
                            torch.min(x0_, x1_), \
                            torch.min(y0_, y1_) 

    inter_area = (U_x_-U_x).clamp(min=0.) * (U_y_-U_y).clamp(min=0.)
    union_area = w0*h0 + w1*h1 - inter_area

    del x0, y0, w0, h0, x1, y1, w1, h1, x0_, y0_, x1_, y1_, U_x, U_x_, U_y, U_y_

    return inter_area / union_area

if __name__ == "__main__":
    xywh0 = torch.Tensor([2., 0., 1., 1.])
    xywh1 = torch.Tensor([0., 0., 1., 1.])

    print(xywh0)
    print(xywh1)
    print(IoU(xywh0, xywh1))