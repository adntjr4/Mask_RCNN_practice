

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign

from src.model.anchor_func import ( anchor_labeling_per_batch,
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

        # set criterion
        self._set_criterion()

        # init network
        self.roi_align = RoIAlign(self.roi_resolution, 1.0, -1, aligned=True)

        self.fc1 = nn.Linear(self.rpn_channel * self.roi_resolution * self.roi_resolution, self.fc_channel)
        self.fc2 = nn.Linear(self.fc_channel, self.fc_channel)
        self.obj_fc = nn.Linear(self.fc_channel, 1)
        self.reg_fc = nn.Linear(self.fc_channel, 4)

    def _set_criterion(self):
        self.box_objectness_criterion = nn.BCELoss(reduction='mean')
        self.box_regression_criterion = nn.SmoothL1Loss(reduction='sum')

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
        # FPN
        if self.FPN_mode:
            # mapping
            # roi align
            raise NotImplementedError
        # resnet
        else:
            # roi align
            roi = self.roi_align(feature_map[0], [bboxes for bboxes in proposals])

        # reshape
        batch_size = proposals.size()[0]
        _, C, PH, PW = roi.size()
        reshaped_roi = roi.view(batch_size, -1, C*PH*PW)

        # forward
        layer_out = F.relu(self.fc1(reshaped_roi))
        layer_out = F.relu(self.fc2(layer_out))
        
        objectnesses = torch.sigmoid(self.obj_fc(layer_out))
        bbox_deltas = self.reg_fc(layer_out)

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
            # 여기는 rpn에서 쓰던 함수 붙여오기 threshold 사용하는 버젼이랑 top N 버젼
            # 근데 생각해보니 threshold랑 top N 할 필요 없이 pre thres가 있으니까 nms하고 뽑으면 될듯.

            # pre threshold

            # bbox regression

            # bbox clipping

            # nms

            return detections

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
