

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from src.model.anchor_func import ( anchor_labeling_per_batch,
                                    box_regression,
                                    nms_per_batch,
                                    invaild_bbox_cliping_per_batch,
                                    calculate_regression_parameter )


class BoxHead(nn.Module):
    def __init__(self, FPN_mode, rpn_channel, conf_box):
        super().__init__()

        # configs
        self.FPN_mode = FPN_mode
        self.rpn_channel = rpn_channel

        self.fc_channel         = conf_box['fc_channel']
        self.roi_resolution     = conf_box['roi_resolution']
        self.reg_weight         = conf_box['regression_weight']
        self.label_thres        = conf_box['labeling_threshold']
        self.sampling_number    = conf_box['sampling_number']
        self.positive_fraction  = conf_box['positive_fraction']
        self.test_score_thres   = conf_box['test_score_threshold']
        self.test_nms_thres     = conf_box['test_nms_threshold']

        # set criterion
        self._set_criterion()

        # init network
        self.roi_align = RoIAlign(self.roi_resolution, 1.0, -1, aligned=True)

        self.res5 = Res5(self.rpn_channel)

        self.fc = nn.Linear(self.rpn_channel * self.roi_resolution * self.roi_resolution, self.fc_channel)
        self.obj_fc = nn.Linear(self.fc_channel, 1)
        self.reg_fc = nn.Linear(self.fc_channel, 4)

    def _set_criterion(self):
        self.box_objectness_criterion = nn.BCELoss(reduction='mean')
        self.box_regression_criterion = nn.SmoothL1Loss(reduction='sum')

    def roi_head_layer_forward(self, roi, batch_size):
        res5_output = self.res5(roi)
        _, C, PH, PW = res5_output.size()
        reshaped_roi = res5_output.view(batch_size, -1, C*PH*PW)

        layer_out = F.relu_(self.fc(reshaped_roi))
        
        objectnesses = torch.sigmoid(self.obj_fc(layer_out))
        bbox_deltas = self.reg_fc(layer_out)

        return objectnesses, bbox_deltas

    def forward(self, feature_map, proposals, data, mode):
        '''
        Args:
            feature_map (List) : List([B, C, H, W])
            proposals (Tensor) : [B, topk, 4]
            data (dict)
            mode (bool)
        Returns(train mode):
            losses (Tensor)
        Returns(eval mode):
            detections (Tensor) : [N, 4]
            scores (Tensor)     : [N]
            img_id_map (Tensor) : [N]
        '''
        # detectron2 heuristic method (adding gt bbox)
        proposals = torch.cat([proposals, data['bbox']], dim=1)
        batch_size = proposals.size()[0]

        # FPN
        if self.FPN_mode:
            # mapping
            # roi align
            raise NotImplementedError
        # resnet
        else:
            # roi align
            roi = self.roi_align(feature_map[0], [bboxes for bboxes in proposals])
        
        # layer forward and get objectnesses, deltas
        objectnesses, bbox_deltas = self.roi_head_layer_forward(roi, batch_size)

        losses = dict()
        if mode == 'train':
            # anchor labeling
            proposals_label, closest_gt = anchor_labeling_per_batch(proposals, data['bbox'], self.label_thres, self.label_thres, closest=False)

            # objectness loss
            selected_cls_out, label = self.get_cls_output_target(objectnesses, proposals_label)
            losses['box_obj'] = self.box_objectness_criterion(selected_cls_out, label)

            # bbox regression loss
            if torch.sum(proposals_label > 0) != 0:
                predicted_t, calculated_t = self.get_box_output_target(data['bbox'], proposals, bbox_deltas, proposals_label, closest_gt)
                losses['box_reg'] = self.box_regression_criterion(predicted_t, calculated_t)
            else:
                losses['box_reg'] = data['bbox'].new_zeros(())

            return losses
        else:
            # bbox regression
            bboxes = box_regression(proposals, bbox_deltas, self.reg_weight)

            # bbox clipping
            invaild_bbox_cliping_per_batch(bboxes, data['img_size'])

            # nms
            nms_keep = nms_per_batch(bboxes, objectnesses, self.test_nms_thres)

            # pre threshold
            threshold_mask = torch.logical_and(objectnesses.squeeze(2) > self.test_score_thres, nms_keep)

            # corresponding image id
            img_id_map = torch.cat([one_id.repeat(threshold_mask[batch_idx].sum()) for batch_idx, one_id in enumerate(data['img_id'])])

            return bboxes[threshold_mask], objectnesses[threshold_mask], img_id_map

    def get_cls_output_target(self, objectness, proposals_label):
        '''
        random proposal sampling for training
        Args:
            objectness (Tensor) : [B, A, 1]
            proposals_label (Tensor) : [B, A] (1, 0, -1) (at box head layer neutual(0) label doesn't exist)
        returns:
            training_cls_score (Tensor) : [B, sampling_number]
            training_cls_gt : [B, sampling_number]
        '''
        batch_size = objectness.size()[0]

        training_cls_score_list = []
        training_cls_gt_list    = []

        # for each batch
        for b_idx in range(batch_size):
            # gather objectnesses
            pos_cls_out, neg_cls_out = objectness[b_idx][proposals_label[b_idx] > 0], objectness[b_idx][proposals_label[b_idx] < 0] # [P], [N]
            pos_num,     neg_num     = pos_cls_out.size()[0],                      neg_cls_out.size()[0]

            # random sampling
            pivot = int(self.sampling_number * self.positive_fraction)
            sampled_pos_num = min(pivot, pos_num)
            sampled_neg_num = self.sampling_number - sampled_pos_num
            sampled_pos_cls_out = pos_cls_out[torch.randperm(pos_num)[:sampled_pos_num]]
            sampled_neg_cls_out = neg_cls_out[torch.randperm(neg_num)[:sampled_neg_num]]

            # training samples concatnate
            one_cls_score = torch.cat([sampled_pos_cls_out, sampled_neg_cls_out]).squeeze(-1)
            one_cls_gt = torch.cat([pos_cls_out.new_ones((sampled_pos_num)), pos_cls_out.new_zeros((sampled_neg_num))])

            training_cls_score_list.append(one_cls_score)
            training_cls_gt_list.append(one_cls_gt)

        return torch.stack(training_cls_score_list), torch.stack(training_cls_gt_list)

    def get_box_output_target(self, gt_bbox, proposals, bbox_deltas, proposals_label, closest_gt):
        '''
        Args:
            gt_bbox (Tensor)      : [B, N, 4]
            proposals (Tensor)    : [B, A, 4]
            bbox_deltas (Tensor)  : [B, A, 4]
            proposals_label (Tensor) : [B, A] (1, 0, -1)
            closest_gt (Tensor)   : [P, 2] (0 ~ B-1), (0 ~ N-1)
        Returns:
            predicted_t  : Tensor[P, 4]
            calculated_t : Tensor[P, 4]
        '''
        # calculate target regression parameter
        positive_proposals = proposals[proposals_label > 0] # [P, 4]
        positive_gt = torch.stack([gt_bbox[batch_num][gt_num] for batch_num, gt_num in closest_gt]) # [P, 4]
        calculated_t = calculate_regression_parameter(positive_proposals, positive_gt, self.reg_weight) # [P, 4]

        # reshape output regression prediction
        predicted_t = bbox_deltas[proposals_label > 0] # [P, 4]

        return predicted_t, calculated_t

class Res5(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv11_1 = nn.Conv2d(channel   , channel//4, kernel_size=1, stride=1, bias=False)
        self.conv33   = nn.Conv2d(channel//4, channel//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv11_2 = nn.Conv2d(channel//4, channel   , kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        residual = x
        x = F.relu_(self.conv11_1(x))
        x = F.relu_(self.conv33(x))
        x = self.conv11_2(x)
        x += residual
        return F.relu_(x)
