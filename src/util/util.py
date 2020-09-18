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
        img = img.permute(1,2,0).numpy()

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

def transform_xywh(xywh, trans):
    '''
    Args:
        xywh (list) : [N, 4]
        trans : affine transformation matrix
    Returns:
        tranformed_xywh (np.array) : [N, 4]
    '''
    if len(xywh) != 0:
        xywh = np.array(xywh).transpose(1,0)
        xy, wh = xywh[0:2,:], xywh[2:4,:]

        ones = np.ones((1, xy.shape[1]))

        xy0 = np.dot(trans, np.concatenate((xy, ones))).transpose(1,0)
        xy1 = np.dot(trans, np.concatenate((xy+wh, ones))).transpose(1,0)

        max_xy = np.max(np.stack((xy0, xy1)), axis=0)
        min_xy = np.min(np.stack((xy0, xy1)), axis=0)

        transformed_wh = max_xy - min_xy

        return np.concatenate((min_xy, transformed_wh), axis=1)
    return np.zeros((0, 4))

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
    '''
    h, w, _ = img.shape

    if h > w:
        size = (resize[0], int(resize[0]*w/h))
    else:
        size = (int(resize[0]*h/w), resize[0])

    trans, inv_trans = get_trans_matrix((h,w), size)
    resized_img = cv2.warpAffine(img, trans, resize)
    img_tensor = torch.Tensor(resized_img.transpose(2,0,1))

    return img_tensor, trans, inv_trans

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
