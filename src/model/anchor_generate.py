import torch


def generate_anchor_form(anchor, feature_size, image_size):
    '''
    generate anchors before doing box regression
    Args:
        anchor (List)
        feature_size (Tuple) (HW)
        input_size (Tuple) (HW)
    Returns:
        anchor_bbox : Tensor[k, H, W, 4]
    '''
    feature_size_h, feature_size_w = feature_size
    image_size_h  , image_size_w   = image_size

    x_expand_ratio = image_size_w / feature_size_w
    y_expand_ratio = image_size_h / feature_size_h

    anchor_bbox_list = []
    for anc in anchor:
        # repeat the aranged tensor and expand
        x = torch.arange(feature_size_w).repeat((feature_size_h, 1))              * x_expand_ratio # [W] -> [H, W]
        y = torch.arange(feature_size_h).repeat((feature_size_w, 1)).permute(1,0) * y_expand_ratio # [H] -> [H, W]

        w = torch.ones((feature_size_h, feature_size_w)) * anc[0]          # [H, W]
        h = torch.ones((feature_size_h, feature_size_w)) * anc[0] * anc[1] # [H, W]
            
        anchor_bbox_t = torch.stack((x,y,w,h), dim=2) # [H, W, 4]
        anchor_bbox_list.append(anchor_bbox_t)
    anchor_bbox = torch.stack(anchor_bbox_list) # [k, H, W, 4]
    return anchor_bbox

def box_regression(bbox, variables):
    '''
    Args:
        bbox : Tensor[B, k, H, W, 4]
        variables : Tensor[B, 4*k, H, W]
    Returns:
        moved_bbox : Tensor[B, k, H, W, 4]
    '''
    a_x, a_y, a_w, a_h = bbox.split(1, dim=4)      # [B, k, H, W, 1]
    t_x, t_y, t_w, t_h = variables.split(1, dim=1) # [B, k, H, W, 1]

    m_x = a_w * t_x + a_x
    m_y = a_h * t_y + a_y
    m_w = a_w * t_w.exp()
    m_h = a_h * t_h.exp()

    moved_bbox = torch.cat([m_x, m_y, m_w, m_h], dim=4)

    return moved_bbox
