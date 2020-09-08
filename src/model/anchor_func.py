import torch
import torch.nn.functional as F

from src.util.util import IoU


def generate_anchor_form(anchor, feature_size, image_size):
    '''
    generate anchors before doing box regression
    Args:
        anchor (List)
        feature_size (List of Tuple) (HW)
        input_size (Tuple) (HW)
    Returns:
        anchor_bbox : List(Tensor[k, H, W, 4])
    '''
    anchor_bbox = []
    for idx, f_size in enumerate(feature_size):
        feature_size_h, feature_size_w = f_size
        image_size_h  , image_size_w   = image_size

        x_expand_ratio = image_size_w / feature_size_w
        y_expand_ratio = image_size_h / feature_size_h

        anchor_bbox_list = []
        for anc in anchor[idx]:
            # repeat the aranged tensor and expand
            y, x = torch.meshgrid(torch.arange(feature_size_h), torch.arange(feature_size_w))

            x = x * x_expand_ratio # [W] -> [H, W]
            y = y * y_expand_ratio # [H] -> [H, W]

            w = torch.ones((feature_size_h, feature_size_w)) * anc[0]          # [H, W]
            h = torch.ones((feature_size_h, feature_size_w)) * anc[0] * anc[1] # [H, W]
                
            anchor_bbox_t = torch.stack((x,y,w,h), dim=2) # [H, W, 4]
            anchor_bbox_list.append(anchor_bbox_t)
        anchor_bbox.append(torch.stack(anchor_bbox_list)) # [k, H, W, 4]
    return anchor_bbox

def anchor_preprocessing(anchors, cls_score, bbox_pred, top_k, bbox_weight, nms_threshold):
    '''
    Args:
        anchors (Tensor) : List([B, A, 4])
        cls_score (Tensor) : List([B, 1*k, H, W])
        bbox_pred (Tensor) : List([B, 4*k, H, W])
        top_k (int)
    Returns:
        post_anchors (Tensor) : [B, A, 4]
        post_cls_score (Tensor) : [B, A, 4]
        post_bbox_pred (Tensor) : [B, A, 4]
        post_keep_amp (Tensor) : [B, A, 4]
    '''
    # reshape tensor
    post_anchors = torch.cat(anchors, dim=1)
    post_cls_score = reshape_output(cls_score, n=1)
    post_bbox_pred = reshape_output(bbox_pred, n=4)

    # top k
    post_anchors, post_cls_score, post_bbox_pred = top_k_per_batch(post_anchors, post_cls_score, post_bbox_pred, top_k)

    # bbox regression
    origin_anchors = post_anchors
    post_anchors = box_regression(post_anchors, post_bbox_pred, bbox_weight)

    # NMS
    post_keep_map = nms_per_batch(post_anchors, post_cls_score, nms_threshold)

    return origin_anchors, post_anchors, post_cls_score, post_bbox_pred, post_keep_map


def box_regression(bbox, variables, weight):
    '''
    Args:
        bbox : Tensor[..., N, 4]
        variables : Tensor[..., N, 4]
        weight (Tuple) : (w_x, w_y, w_w, w_h)
    Returns:
        moved_bbox : Tensor[..., N, 4]
    '''
    cat_dim = len(bbox.size())-1

    a_x, a_y, a_w, a_h = bbox.split(1, dim=cat_dim)       # [..., N, 1]
    t_x, t_y, t_w, t_h = variables.split(1, dim=cat_dim)  # [..., N, 1]

    w_x, w_y, w_w, w_h = weight

    t_x *= w_x
    t_y *= w_y
    t_w *= w_w
    t_h *= w_h

    m_x = a_w * t_x + a_x
    m_y = a_h * t_y + a_y
    m_w = a_w * t_w.exp()
    m_h = a_h * t_h.exp()

    moved_bbox = torch.cat([m_x, m_y, m_w, m_h], dim=cat_dim) # [..., N, 4]

    return moved_bbox

def calculate_regression_parameter(anchor_bbox, gt_bbox, weight):
    '''
    Args:
        anchor_bbox (Tensor) : [..., N, 4]
        gt_bbox (Tensor) : [..., N, 4]
        weight (Tuple) : (wx, wy, ww, wh)
    Returns:
        regression_parameter (Tensor) : [..., N, 4]
    '''
    cat_dim = len(anchor_bbox.size())-1

    a_x, a_y, a_w, a_h = anchor_bbox.split(1, dim=cat_dim)    # [..., N, 1]
    g_x, g_y, g_w, g_h = gt_bbox.split(1, dim=cat_dim)        # [..., N, 1]

    t_x = (g_x - a_x) / a_w
    t_y = (g_y - a_y) / a_h
    t_w = (g_w / a_w).log()
    t_h = (g_h / a_h).log()

    w_x, w_y, w_w, w_h = weight

    t_x /= w_x
    t_y /= w_y
    t_w /= w_w
    t_h /= w_h

    regression_parameter = torch.cat([t_x, t_y, t_w, t_h], dim=cat_dim)   # [..., N, 4]

    return regression_parameter

def top_k_per_batch(anchors, cls_score, bbox_pred, k):
    '''
    select top k anchors based on objectness score per batch
    Args:
        anchors (Tensor) : [B, A, 4]
        cls_score (Tensor) : [B, A, 1]
        bbox_pred (Tensor) : [B, A, 4]
        k (int)
    Returns:
        top_k_anchors (Tensor) : [B, k, 4]   (sorted as follow cls_score)
        top_k_cls_score (Tensor) : [B, k, 1] (sorted)
        top_k_bbox_pred (Tensor) : [B, k, 4] (sorted as follow cls_score)
    '''
    top_k_cls_score, indices = torch.sort(cls_score, dim=1, descending=True)
    indices = indices.repeat(1,1,4) # [B, A, 4]
    top_k_anchors = anchors.gather(dim=1, index=indices)[:,:k,:]
    top_k_cls_score = top_k_cls_score[:,:k,:]
    top_k_bbox_pred = bbox_pred.gather(dim=1, index=indices)[:,:k,:]
    return top_k_anchors, top_k_cls_score, top_k_bbox_pred

def remove_invaild_bbox(anchors, bbox_pred, cls_score):
    '''
    remove invaild bboxes (e.g. outside image)
    Args:
        anchors (Tensor) : [N, 4]
        bbox_pred (Tensor) : [N, 4]
        cls_score (Tensor) : [N, 1]
    Returns:
        valid_anchors (Tensor) : [N', 4]
        valid_bbox_pred (Tensor) : [N', 4]
        valid_cls_score (Tensor) : [N', 1] 
    '''
    return anchors, bbox_pred, cls_score

def nms(bbox, score, threshold):
    '''
    non-maximum suppression
    Args:
        bbox (Tensor) : [N, 4]
        score (Tensor) : [N, 1]
        threshold (float)
    Returns:
        keep_map (Tensor) : [N]
    '''
    assert int(score.size()[0]) == int(bbox.size()[0]), "size of two arg tensor must be same"

    bbox_number = int(score.size()[0])

    cross_bbox0 = bbox.repeat(bbox_number, 1, 1)
    cross_bbox1 = cross_bbox0.permute(1,0,2)

    cross_IoU = IoU(cross_bbox0, cross_bbox1)

    cross_score0 = score.repeat(1, bbox_number)
    cross_score1 = cross_score0.permute(1,0)

    cross_score_comparision = cross_score0-cross_score1

    # over threshold IoU value
    cross_IoU_bool = cross_IoU > threshold
    
    # less score 
    cross_score_comparision_bool = cross_score_comparision < 0.

    keep_map = torch.logical_not(torch.logical_and(cross_IoU_bool, cross_score_comparision_bool).sum(dim=1, dtype=torch.bool))

    return keep_map

def nms_per_batch(bbox, score, threshold):
    '''
    non-maximum suppression
    Args:
        bbox (Tensor) : [B, N, 4]
        score (Tensor) : [B, N]
        threshold (float)
    Returns:
        total_keep_map (Tensor) : [B, N]
    '''
    batch_size, _, _ = bbox.size()
    total_keep_map = torch.stack([nms(bbox[idx], score[idx], threshold) for idx in range(batch_size)])
    return total_keep_map

def anchor_labeling_per_batch(anchor, keep, gt_bbox, pos_thres, neg_thres):
    '''
    labeling positive, neutral, negative anchor per batch
    Args:
        anchor (Tensor) : [B, A, 4]
        keep (Tensor) : [B, A]
        gt_bbox (Tensor) : [B, N, 4]
    Returns:
        anchor_label (Tensor) : [B, A] (1, 0, -1)
        closest_gt (Tensor) : [B, A] (0 ~ N-1)
    '''
    _        , anchor_num, _ = anchor.size()
    batch_size, object_num, _ = gt_bbox.size()

    # expand anchor and gt_bbox for cross IoU calculation
    expanded_anchor = anchor.repeat(object_num,1,1,1).permute(1,0,2,3)   # [B, A, 4] -> [N, B, A, 4] -> [B, N, A, 4]
    expanded_gt_bbox = gt_bbox.repeat(anchor_num,1,1,1).permute(1,2,0,3) # [B, N, 4] -> [A, B, N, 4] -> [B, N, A, 4]

    # IoU calculation
    cross_IoU = IoU(expanded_anchor, expanded_gt_bbox) # [B, N, A]

    # label positive and negative
    anchor_pos_label = (cross_IoU > pos_thres).any(1)                    # [B, A]
    anchor_neg_label = torch.logical_not((cross_IoU > neg_thres).any(1)) # [B, A]

    # find closest anchor for each gt_bbox
    closest_indices = torch.argmax(cross_IoU, dim=2) # [B, N]
    one_hot_indices = F.one_hot(closest_indices, num_classes=anchor_num).type(torch.bool) # [B, N, A]
    closest_label = one_hot_indices.any(1) # [B, A]
    anchor_pos_label += closest_label

    # find closest gt_bbox for each anchor
    closest_gt_indices = torch.argmax(cross_IoU, dim=1) # [B, A]
    closest_gt_object = closest_gt_indices[anchor_pos_label] # [P] (0 ~ N-1)
    closest_gt_batch  = torch.arange(0, batch_size).cuda().repeat(anchor_num, 1).permute(1,0)[anchor_pos_label] # [P] (0 ~ B-1)
    closest_gt = torch.stack([closest_gt_batch, closest_gt_object], dim=1)

    # merge anchor label
    anchor_label = anchor_pos_label * 1.0 # [B, A]
    anchor_label -= anchor_neg_label * 1.0 # [B, A]
    anchor_label += torch.logical_and(anchor_pos_label, anchor_neg_label) * 1.0 # [B, A]

    return anchor_label, closest_gt

def training_anchor_selection_per_batch(cls_score, anchor_label, sampling_number):
    '''
    random anchor sampling for training
    Args:
        cls_score (Tensor) : [B, A, 1]
        anchor_label (Tensor) : [B, A] (1, 0, -1)
        sampling_number (int)
    returns:
        training_cls_score (Tensor) : [B, sampling_number]
        training_cls_gt : [B, sampling_number]
    '''
    batch_size, _, _ = cls_score.size()

    training_cls_score_list = []
    training_cls_gt_list = []

    for b_idx in range(batch_size):
        pos_cls_out, neg_cls_out = cls_score[b_idx][anchor_label[b_idx] > 0], cls_score[b_idx][anchor_label[b_idx] < 0] # [P], [N]
        pos_num,     neg_num     = pos_cls_out.size()[0],                     neg_cls_out.size()[0]

        # random sampling
        pivot = int(sampling_number/2)
        if pos_num <= pivot:
            sampled_pos_cls_out = pos_cls_out
            sampled_neg_cls_out = neg_cls_out[torch.randperm(neg_num)[:sampling_number-pos_num]]
            sampled_pos_num = pos_num
            sampled_neg_num = sampling_number - pos_num
        else:
            sampled_pos_cls_out = pos_cls_out[torch.randperm(pos_num)[:pivot]]
            sampled_neg_cls_out = neg_cls_out[torch.randperm(neg_num)[:sampling_number-pivot]]
            sampled_pos_num = pivot
            sampled_neg_num = sampling_number - pivot

        one_cls_score = torch.cat([sampled_pos_cls_out, sampled_neg_cls_out]).squeeze(-1)
        one_cls_gt = torch.cat([pos_cls_out.new_ones((sampled_pos_num)), pos_cls_out.new_zeros((sampled_neg_num))])

        training_cls_score_list.append(one_cls_score)
        training_cls_gt_list.append(one_cls_gt)

    return torch.stack(training_cls_score_list), torch.stack(training_cls_gt_list)

def training_bbox_regression_calculation(gt_bbox, bbox_pred, anchor_label, highest_gt):
    '''
    Args:
        gt_bbox (Tensor) : [B, N, 4]
        default_anchors (Tensor) : [B, A, 4]
        bbox_pred (Tensor) : [B, A, 4]
        anchor_label (Tensor) : [B, A] (1, 0, -1)
        highest_gt (Tensor) : [B, A] (0 ~ B-1), (0 ~ N-1)
    Returns:
        predicted_t  : Tensor[P, 4]
        calculated_t : Tensor[P, 4]
    '''
    '''
    P : number of positive anchor
    Args:
        reg_out (Tensor) : List([B, 4*k, H, W])
        gt : {img, img_size, label, bbox}
            bbox (Tensor) : [B, N, 4]
        anchor_label : {anchor_bbox, anchor_pos_label, pos_indice}
            anchor_bbox (Tensor) : [B, k, H, W, 4]
            anchor_pos_label (Tensor) : [B, k, H, W] (bool)
            highest_gt (Tensor) : [P, 2] (batch, object number)
    Returns:
        predicted_t  : Tensor[P, 4]
        calculated_t : Tensor[P, 4]
    '''
    predicted_t = []
    calculated_t = []
    for idx, one_reg_out in enumerate(reg_out):
        anchor_bbox, pos_label, highest_gt = anchor_label['anchor_bbox'][idx], anchor_label['anchor_pos_label'][idx], anchor_label['highest_gt'][idx]

        B, k4, H, W = one_reg_out.size()
        k = int(k4/4)

        # reshape reg_out for predicted_t
        predicted_t.append(one_reg_out.view(B, k, 4, H, W).permute(0,1,3,4,2)[pos_label]) # [P, 4]

        # calculate box regression parameter
        pos_anchor_bbox = anchor_bbox[pos_label] # [P, 4] (xywh)
        pos_gt_bbox = torch.stack([gt['bbox'][batch_num][gt_num] for batch_num, gt_num in highest_gt]) # [P, 4] (xywh)

        calculated_t.append(calculate_regression_parameter(pos_anchor_bbox, pos_gt_bbox, self.reg_weight))

    return torch.cat(predicted_t), torch.cat(calculated_t)

def reshape_output(output, n):
    '''
    Args:
        output : [B, n*k, H, W] or List([B, n*k, H, W])
    Returns:
        reshaped_output : [B, kHW, n] or [B, sum(kHW), n]
    '''
    if isinstance(output, list):
        reshaped_output_list = []
        for one_output in output:
            B, nk, H, W = one_output.size()
            reshaped_output_list.append(one_output.view(B, n, int(nk/n), H, W).permute(0,2,3,4,1).view(B, -1, n)) # [B, kHW, n]
        return torch.cat(reshaped_output_list, dim=1)
    else:
        B, nk, H, W = output.size()
        return output.view(B, n, int(nk/n), H, W).permute(0,2,3,4,1).view(B, -1, n) # [B, kHW, n]