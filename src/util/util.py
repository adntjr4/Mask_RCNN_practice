import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np


def load_model(file_name):
    saved_model = torch.load(file_name)
    return saved_model['model_weight']

def draw_boxes(img, boxes, color=(0,255,0)):
    if type(img) == torch.Tensor:
        img = img.cpu()
        img = img.permute(1,2,0).numpy()[:,:,[2,1,0]]

    if type(boxes) == torch.Tensor:
        boxes = boxes.cpu()
        boxes = boxes.tolist()

    if type(boxes) == np.array:
        boxes = boxes.tolist()

    for box in boxes:
        x, y, w, h = box
        pt1 = (int(x), int(y))
        pt2 = (int(x+w), int(y+h))
        img = cv2.rectangle(cv2.UMat(img), pt1, pt2, color, 1)

    return img

def transform_xywh_with_img_id(xywh, img_id_map, inv_trans, img_id):
    '''
    Args:
        xywh (Tensor) : [N, 4]
        img_id_map (Tensor) : [N]
        inv_trans (Tensor) : [B, 2, 3]
        img_id (Tensor) : [B]
    Returns:
        tranformed_xywh (Tensor) : [N, 4]
    '''
    N = xywh.size()[0]

    # matching inverse transforms for each xywh
    # (couldn't think the way with gpu tensor)
    matching_batch = xywh.new_zeros(N).type(torch.int64)
    for batch_idx, one_img_id in enumerate(img_id):
        matching_batch[matching_batch==one_img_id] = batch_idx
    matching_batch = matching_batch.repeat(2,3,1).permute(2,0,1)

    matching_inv_trans = torch.gather(input=inv_trans, dim=0, index=matching_batch)

    # inverse transfomation
    
    xy, wh = xywh[:, 0:2], xywh[:, 2:4] # [N,2]
    ones = xywh.new_ones((N, 1)) # [N,1]

    xy0 = torch.matmul(matching_inv_trans, torch.cat([xy   , ones], dim=1).unsqueeze(2)).squeeze(2) # [N,2,3]x[N,3,1] = [N,2,1] -> [N,2]
    xy1 = torch.matmul(matching_inv_trans, torch.cat([xy+wh, ones], dim=1).unsqueeze(2)).squeeze(2) # [N,2,3]x[N,3,1] = [N,2,1] -> [N,2]

    stack_xy = torch.stack([xy0, xy1]) # [N,2,2]
    max_xy, _ = torch.max(stack_xy, dim=2)
    min_xy, _ = torch.min(stack_xy, dim=2)

    transformed_wh = max_xy - min_xy

    return torch.cat([min_xy, transformed_wh], dim=1)

def transform_xywh(xywh, trans):
    '''
    Args:
        xywh (Tensor) : [N, 4]
        trans : affine transformation matrix
    Returns:
        tranformed_xywh (np.array) : [N, 4]
    '''
    if len(xywh) != 0:
        xywh = np.array(xywh).transpose(1,0) # [4,N]
        xy, wh = xywh[0:2,:], xywh[2:4,:] # [2,N]

        ones = np.ones((1, xy.shape[1])) # [1,N]

        xy0 = np.dot(trans, np.concatenate((xy, ones))).transpose(1,0)      # [2,3]x[3,N] = [2,N] -> [N,2]
        xy1 = np.dot(trans, np.concatenate((xy+wh, ones))).transpose(1,0)

        max_xy = np.max(np.stack((xy0, xy1)), axis=0)
        min_xy = np.min(np.stack((xy0, xy1)), axis=0)

        transformed_wh = max_xy - min_xy

        return np.concatenate((min_xy, transformed_wh), axis=1)
    return np.zeros((0, 4))

image_mean = np.array([103.530, 116.280, 123.675]) # BGR

def img_process(img, resize):
    '''
    pre-processing of image from cv image to tensor
    Args:
        img (np.ndarray)
        resize (tuple) : height, width
    Returns:
        padded_img (Tensor) : [C, H, W]
        size (Tensor) : [2]
        transformation_matrix
        post_size (Tensor) : [2] (H, W)
    '''
    h, w, _ = img.shape

    if h > w:
        size = (resize[0], int(resize[0]*w/h))
    else:
        size = (int(resize[0]*h/w), resize[0])

    trans, inv_trans = get_trans_matrix((h,w), size)
    resized_img = cv2.warpAffine(img, trans, resize)
    resized_img = resized_img - image_mean
    img_tensor = torch.Tensor(resized_img[:,:,[2,1,0]].transpose(2,0,1))

    return img_tensor, trans, inv_trans, size

def get_trans_matrix(size, dst_size, hor_flip=False):
    '''
    Args:
        size (Tuple) : (h,w) of original image
        dst_size (Tuple) : (h,w) of destination image
        hor_flip (bool) : True when horizontal flip
    Returns:
        trans
        inv_trans
    '''
    src = np.zeros((3,2), dtype=np.float32)
    src[0, :] = np.array([0.,0.], dtype=np.float32)
    src[1, :] = np.array([0, size[0]], dtype=np.float32)
    src[2, :] = np.array([size[1], 0.], dtype=np.float32)

    dst = np.zeros((3,2), dtype=np.float32)
    if hor_flip:
        dst[0, :] = np.array([dst_size[1], 0.], dtype=np.float32)
        dst[1, :] = np.array([dst_size[1], dst_size[0]], dtype=np.float32)
        dst[2, :] = np.array([0., 0.], dtype=np.float32)
    else:
        dst[0, :] = np.array([0.,0.], dtype=np.float32)
        dst[1, :] = np.array([0, dst_size[0]], dtype=np.float32)
        dst[2, :] = np.array([dst_size[1], 0.], dtype=np.float32)

    return cv2.getAffineTransform(src, dst), cv2.getAffineTransform(dst, src)

def IoU(xywh0:torch.Tensor, xywh1:torch.Tensor):
    '''
    calculate IoUs using tensor
    Args:
        xywh0 (Tensor) : [..., 4]
        xywh1 (Tensor) : [..., 4]
    Returns:
        IoUs (Tensor) : [...]
    '''
    with torch.no_grad():
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

        return (inter_area / union_area).squeeze(-1)
