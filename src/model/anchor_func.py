import math

import torch
from torchvision.ops import nms
import torch.nn.functional as F

from src.util.util import IoU

@torch.no_grad()
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
            mesh_y, mesh_x = torch.meshgrid(torch.arange(feature_size_h), torch.arange(feature_size_w))

            center_x = (mesh_x+0.5) * x_expand_ratio
            center_y = (mesh_y+0.5) * y_expand_ratio

            w = torch.ones((feature_size_h, feature_size_w)) * math.sqrt(anc[0]*anc[0]/anc[1])
            h = torch.ones((feature_size_h, feature_size_w)) * math.sqrt(anc[0]*anc[0]*anc[1])

            x = center_x - 0.5*w
            y = center_y - 0.5*h
                
            anchor_bbox_t = torch.stack((x,y,w,h), dim=2) # [H, W, 4]
            anchor_bbox_list.append(anchor_bbox_t)
        anchor_bbox.append(torch.stack(anchor_bbox_list).cuda()) # [k, H, W, 4]
    return anchor_bbox

def anchor_preprocessing(anchors, image_size, cls_score, bbox_pred, pre_top_k, post_top_k, nms_thres, reg_weight):
    '''
    Args:
        anchors     (Tensor) : List([B, A, 4])
        image_size  (Tensor) : [B, 2]
        cls_score   (Tensor) : List([B, 1*k, H, W])
        bbox_pred   (Tensor) : List([B, 4*k, H, W])
        pre_top_k   (int)    
        post_top_k  (int)    
        nms_thres   (float) 
        reg_weight  (list)
    Returns:
        origin_anchors   (Tensor) : [B, A, 4]
        post_cls_score (Tensor) : [B, A, 1]
        post_bbox_pred (Tensor) : [B, A, 4]
        post_keep_map  (Tensor) : [B, A]
    '''
    # for each feature level do {pre-nms-top-k, nms}
    origin_anchors, concat_cls_score = [], []
    concat_bbox_pred, concat_keep_map = [], []
    for lvl_anchors, lvl_cls_score, lvl_bbox_pred in zip(anchors, cls_score, bbox_pred):
        # reshape
        lvl_cls_score = reshape_output(lvl_cls_score, n=1)
        lvl_bbox_pred = reshape_output(lvl_bbox_pred, n=4)

        # pre nms top k (per level)
        indices = sort_per_batch(lvl_cls_score)
        lvl_anchors_top   = top_k_from_indices(lvl_anchors,   indices, pre_top_k)
        lvl_cls_score_top = top_k_from_indices(lvl_cls_score, indices, pre_top_k)
        lvl_bbox_pred_top = top_k_from_indices(lvl_bbox_pred, indices, pre_top_k)

        # bbox regression
        lvl_anchors_top = box_regression(lvl_anchors_top, lvl_bbox_pred_top, reg_weight)

        # invaild bbox clipping
        invaild_bbox_cliping_per_batch(lvl_anchors_top, image_size)

        # remove empty bbox
        non_empty_keep = torch.logical_and( lvl_anchors_top[:,:,2] > 0 , lvl_anchors_top[:,:,3] > 0 )

        # NMS
        nms_keep = nms_per_batch(lvl_anchors_top, lvl_cls_score_top, nms_thres)
        concat_keep_map.append(torch.logical_and(non_empty_keep, nms_keep))

        # append
        origin_anchors.append(lvl_anchors_top)
        concat_cls_score.append(lvl_cls_score_top)
        concat_bbox_pred.append(lvl_bbox_pred_top)

    # concatenate all level anchors
    origin_anchors   = torch.cat(origin_anchors,   dim=1)
    concat_cls_score = torch.cat(concat_cls_score, dim=1)
    concat_bbox_pred = torch.cat(concat_bbox_pred, dim=1)
    keep  = torch.cat(concat_keep_map,  dim=1)

    # for each batch do {post-nms-top-k}
    post_origin_anchors, post_cls_score, post_bbox_pred = [], [], []
    for idx in range(concat_cls_score.size()[0]):
        # post nms top k
        indices = sort_per_batch(concat_cls_score[idx][keep[idx]])
        post_origin_anchors.append(top_k_from_indices(origin_anchors[idx][keep[idx]], indices, post_top_k))
        post_cls_score.append(top_k_from_indices(concat_cls_score[idx][keep[idx]], indices, post_top_k))
        post_bbox_pred.append(top_k_from_indices(concat_bbox_pred[idx][keep[idx]], indices, post_top_k))

    # stack all batch
    post_origin_anchors = torch.stack(post_origin_anchors)
    post_cls_score = torch.stack(post_cls_score)
    post_bbox_pred = torch.stack(post_bbox_pred)

    return post_origin_anchors, post_cls_score, post_bbox_pred

@torch.no_grad()
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

    a_x_c = a_x+0.5*a_w
    a_y_c = a_y+0.5*a_h

    w_x, w_y, w_w, w_h = weight

    t_x /= w_x
    t_y /= w_y
    t_w /= w_w
    t_h /= w_h

    m_x_c = a_x_c + a_w * t_x
    m_y_c = a_y_c + a_h * t_y
    m_w = a_w * t_w.exp()
    m_h = a_h * t_h.exp()
    m_x = m_x_c - 0.5*m_w
    m_y = m_y_c - 0.5*m_h

    moved_bbox = torch.cat([m_x, m_y, m_w, m_h], dim=cat_dim) # [..., N, 4]

    return moved_bbox

@torch.no_grad()
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

    a_x_c = a_x+0.5*a_w
    a_y_c = a_y+0.5*a_h

    g_x_c = g_x+0.5*g_w
    g_y_c = g_y+0.5*g_h

    t_x = (g_x_c - a_x_c) / a_w
    t_y = (g_y_c - a_y_c) / a_h
    t_w = (g_w / a_w).log()
    t_h = (g_h / a_h).log()

    w_x, w_y, w_w, w_h = weight
    t_x *= w_x
    t_y *= w_y
    t_w *= w_w
    t_h *= w_h

    regression_parameter = torch.cat([t_x, t_y, t_w, t_h], dim=cat_dim)   # [..., N, 4]

    return regression_parameter

def top_k_from_indices(source, indices, k):
    '''
    select top k from source with indices
    Args:
        source  (Tensor) : [B, A, ?] or [A, ?]
        indices (Tensor) : [B, A] or [A]
        k (int)
    Returns:
        top_k_source (Tensor) : [B, k, ?] or [k, ?]
    '''
    if len(source.size()) == 3:
        _, _, last_dimension = source.size()
        repeated_indices = indices.repeat(1, 1, last_dimension)
        return source.gather(dim=1, index=repeated_indices)[:,:k,:]
    elif len(source.size()) == 2:
        _, last_dimension = source.size()
        repeated_indices = indices.repeat(1, last_dimension)
        return source.gather(dim=0, index=repeated_indices)[:k,:]

def sort_per_batch(cls_score):
    '''
    select top k anchors based on objectness score per batch
    Args:
        cls_score (Tensor) : [B, A, 1] or [A, 1]
    Returns:
        sorted_cls_socre (Tensor) : [B, A, 1] or [A, 1]
        indices (Tensor)
    '''
    assert len(cls_score.size()) in [2,3]
    _, indices = torch.sort(cls_score, dim=len(cls_score.size())-2, descending=True)
    return indices

@torch.no_grad()
def invaild_bbox_cliping_per_batch(anchors, image_size):
    '''
    remove invaild bboxes per batch (e.g. outside image)
    Args:
        anchors (Tensor) : [B, A, 4]
        image_size (Tensor) : [B, 2]
    '''
    for anchor, one_img_size in zip(anchors, image_size):
        H, W = one_img_size[0], one_img_size[1]
        x, y, w, h = anchor.split(1, dim=1)
        w += x
        h += y
        x.clamp_(min=0, max=W)
        y.clamp_(min=0, max=H)
        w.clamp_(min=0, max=W)
        h.clamp_(min=0, max=H)
        w -= x
        h -= y

def remove_invaild_bbox_per_batch(anchors, image_size):
    '''
    remove invaild bboxes per batch (e.g. outside image)
    Args:
        anchors (Tensor) : [B, A, 4]
        image_size (Tensor) : [B, 2]
    Returns:
        vaild_map (Tensor) : [B, A]
    '''
    vaild_map = []
    for batch_anchors, batch_image_size in zip(anchors, image_size):
        vaild_map.append(remove_invaild_bbox(batch_anchors, batch_image_size))
    vaild_map = torch.stack(vaild_map)
    return vaild_map

def remove_invaild_bbox(anchors, image_size):
    '''
    remove invaild bboxes (e.g. outside image)
    Args:
        anchors (Tensor) : [A, 4]
        image_size (Tensor) : [2]
    Returns:
        vaild_map (Tensor) : [A]
    '''
    x, y, w, h = anchors.split(1, dim=1)

    tolerance = 128.

    left = x >= -1 * tolerance
    up = y >= -1 * tolerance
    right = x+w <= image_size[1].item() + tolerance
    down = y+h <= image_size[0].item() + tolerance

    vaild_map = torch.logical_and(torch.logical_and(left, up), torch.logical_and(right, down)).squeeze(1)
    return vaild_map

@torch.no_grad()
def nms_made(bbox, score, threshold):
    '''
    non-maximum suppression
    Args:
        bbox (Tensor) : [N, 4]
        score (Tensor) : [N, 1]
        threshold (float)
    Returns:
        keep_map (Tensor) : [N]
    '''
    with torch.no_grad():
        assert int(score.size()[0]) == int(bbox.size()[0]), "size of two arg tensor must be same"

        bbox_number = int(score.size()[0])

        cross_bbox0 = bbox.repeat(bbox_number, 1, 1)
        cross_bbox1 = cross_bbox0.permute(1,0,2)

        cross_IoU = IoU(cross_bbox0, cross_bbox1)

        cross_score0 = score.squeeze(-1).repeat(bbox_number, 1)
        cross_score1 = cross_score0.permute(1,0)

        cross_score_comparision = cross_score0 - cross_score1

        # over threshold IoU value
        cross_IoU_bool = cross_IoU > threshold
        
        # less score 
        cross_score_comparision_bool = cross_score_comparision > 0.

        keep_map = torch.logical_not(torch.logical_and(cross_IoU_bool, cross_score_comparision_bool).any(1))

        return keep_map

@torch.no_grad()
def nms_per_batch(bbox, score, threshold):
    '''
    non-maximum suppression
    Args:
        bbox (Tensor) : [B, N, 4]
        score (Tensor) : [B, N, 1]
        threshold (float)
    Returns:
        total_keep_map (Tensor) : [B, N]
    '''
    batch_size, N, _ = bbox.size()
    total_keep_map = []
    for idx in range(batch_size):
        x, y, w, h = bbox[idx].split(1, dim=1)
        one_xyxy_bboxes = torch.cat([x, y, x+w, y+h], dim=1)
        int_keep = nms(one_xyxy_bboxes, score[idx].squeeze(1), threshold)
        one_keep_map = bbox.new_zeros(N).type(torch.bool)
        one_keep_map[int_keep] = True
        total_keep_map.append(one_keep_map)
    total_keep_map = torch.stack(total_keep_map)
    return total_keep_map

@torch.no_grad()
def anchor_labeling_per_batch(anchor, gt_bbox, pos_thres:float, neg_thres:float, closest:bool=True):
    '''
    labeling positive, neutral, negative anchor per batch
    Args:
        anchor (Tensor) : [B, A, 4]
        gt_bbox (Tensor) : [B, N, 4]
        pos_thres (float) : positive IoU threshold
        neg_thres (float) : negative IoU threshold
        closest (bool) : if True, the closest anchor to any gt will labeled as positvie. 
    Returns:
        anchor_label (Tensor) : [B, A] (1, 0, -1)
        closest_gt (Tensor) : [P, 2] (0 ~ B-1), (0 ~ N-1)
    '''
    _         , anchor_num, _ = anchor.size()
    batch_size, object_num, _ = gt_bbox.size()

    # expand anchor and gt_bbox for cross IoU calculation
    expanded_anchor  = anchor.unsqueeze(0).expand(object_num,-1,-1,-1).permute(1,0,2,3)   # [B, A, 4] -> [N, B, A, 4] -> [B, N, A, 4]
    expanded_gt_bbox = gt_bbox.unsqueeze(0).expand(anchor_num,-1,-1,-1).permute(1,2,0,3) # [B, N, 4] -> [A, B, N, 4] -> [B, N, A, 4]

    # IoU calculation
    cross_IoU = IoU(expanded_anchor, expanded_gt_bbox) # [B, N, A]

    # label positive and negative
    anchor_pos_label = (cross_IoU > pos_thres).any(1)                    # [B, A]
    anchor_neg_label = torch.logical_not((cross_IoU > neg_thres).any(1)) # [B, A]

    # find closest anchor for each gt_bbox
    if closest:
        closest_indices = torch.argmax(cross_IoU, dim=2) # [B, N] (0 ~ A-1)
        one_hot_indices = F.one_hot(closest_indices, num_classes=anchor_num) # [B, N, A]
        one_hot_indices = torch.logical_and(one_hot_indices, cross_IoU > 0.)
        one_hot_indices = (one_hot_indices * gt_bbox[:,:,2].unsqueeze(-1).repeat(1,1,anchor_num)).type(torch.bool) # [B, N, A] (for select real anchor by multiply width of gt bbox)
        closest_label = one_hot_indices.any(1) # [B, A]
        anchor_pos_label += closest_label

    # find closest gt_bbox for each anchor
    closest_gt_indices = torch.argmax(cross_IoU, dim=1) # [B, A]
    closest_gt_object = closest_gt_indices[anchor_pos_label] # [P] (0 ~ N-1)
    closest_gt_batch  = torch.arange(0, batch_size).cuda().repeat(anchor_num, 1).permute(1,0)[anchor_pos_label] # [P] (0 ~ B-1)
    closest_gt = torch.stack([closest_gt_batch, closest_gt_object], dim=1) # [P, 2]

    # merge anchor label
    anchor_label = anchor_neg_label * -1.0
    anchor_label[anchor_pos_label] = 1.0

    # anchor_label = anchor_pos_label * 1.0 # [B, A]
    # anchor_label -= anchor_neg_label * 1.0 # [B, A]
    # anchor_label += torch.logical_and(anchor_pos_label, anchor_neg_label) * 1.0 # [B, A]

    return anchor_label, closest_gt

def anchor_labeling_no_gt(anchor, keep):
    '''
    labeling neutral, negative anchor per batch when if there is no gt
    Args:
        anchor (Tensor) : [B, A, 4]
        keep (Tensor) : [B, A]
    Returns:
        anchor_label (Tensor) : [B, A] (1, 0, -1)
        closest_gt (Tensor) : [P, 2] (0 ~ B-1), (0 ~ N-1)
    '''
    anchor_label = keep * -1.0
    closest_gt = keep.new_zeros(0,2)
    return anchor_label, closest_gt

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
