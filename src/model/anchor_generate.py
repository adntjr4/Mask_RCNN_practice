import torch


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

def box_regression(bbox, variables, weight):
    '''
    Args:
        bbox : Tensor[B, k, H, W, 4]
        variables : Tensor[B, 4*k, H, W]
        weight : Tuple(w_x, w_y, w_w, w_h)
    Returns:
        moved_bbox : Tensor[B, k, H, W, 4]
    '''
    _, k, _, _, _ = bbox.size()

    a_x, a_y, a_w, a_h = bbox.split(1, dim=4) # [B, k, H, W, 1]
    t_x, t_y, t_w, t_h = [tmp.unsqueeze(4) for tmp in variables.split(k, dim=1)] # [B, k, H, W, 1]

    w_x, w_y, w_w, w_h = weight

    t_x *= w_x
    t_y *= w_y
    t_w *= w_w
    t_h *= w_h

    m_x = a_w * t_x + a_x
    m_y = a_h * t_y + a_y
    m_w = a_w * t_w.exp()
    m_h = a_h * t_h.exp()

    moved_bbox = torch.cat([m_x, m_y, m_w, m_h], dim=4)

    return moved_bbox

def calculate_regression_parameter(anchor_bbox, gt_bbox, weight):
    '''
    Args:
        anchor_bbox (Tensor) : [P, 4]
        gt_bbox (Tensor) : [P, 4]
        weight : Tuple(wx, wy, ww, wh)
    Returns:
        regression_parameter (Tensor) : [P, 4]
    '''

    a_x, a_y, a_w, a_h = anchor_bbox.split(1, dim=1) # [P, 1]
    g_x, g_y, g_w, g_h = gt_bbox.split(1, dim=1) # [P, 1]

    t_x = (g_x - a_x) / a_w
    t_y = (g_y - a_y) / a_h
    t_w = (g_w / a_w).log()
    t_h = (g_h / a_h).log()

    w_x, w_y, w_w, w_h = weight

    t_x /= w_x
    t_y /= w_y
    t_w /= w_w
    t_h /= w_h

    regression_parameter = torch.cat([t_x, t_y, t_w, t_h], dim=1)

    return regression_parameter
