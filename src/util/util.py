import torch
import cv2
import numpy as np


def draw_boxes(img, boxes, color=(0,255,0)):
    if type(img) == torch.Tensor:
        img = img.permute(1,2,0).numpy()

    if type(boxes) == torch.Tensor:
        boxes = boxes.tolist()

    for box in boxes:
        x, y, w, h = box
        pt1 = (int(x), int(y))
        pt2 = (int(x+w), int(y+h))
        img = cv2.rectangle(cv2.UMat(img), pt1, pt2, color, 2)

    return img

def resize_xywh(xywh, input_size, target_size):
    x_ratio = target_size[1] / input_size[1]
    y_ratio = target_size[0] / input_size[0]

    after = []
    for before in xywh:
        x,y,w,h = before
        x *= x_ratio
        y *= y_ratio
        w *= x_ratio
        h *= y_ratio
        after.append([x,y,w,h])
    return after

def img_process(img, resize):
    '''
    pre-processing of image from cv image to tensor
    Args:
        img (np.ndarray)
        resize (tuple) : height, width
    '''
    resized_img = cv2.resize(img, dsize=resize, interpolation=cv2.INTER_CUBIC)
    return torch.Tensor(resized_img.transpose(2,0,1))

def IoU(xywh0:torch.Tensor, xywh1:torch.Tensor):
    '''
    calculate IoUs using tensor
    '''
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

    return inter_area / union_area
