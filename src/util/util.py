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
    Returns:
        padded_img (Tensor) : [C, H, W]
        size (Tensor) : [2] 
    '''
    h, w, _ = img.shape

    if h > w:
        size = (int(resize[0]*w/h), resize[0])
        num_pad = resize[0] - int(resize[0]*w/h)
    else:
        size = (resize[0], int(resize[0]*h/w))
        num_pad = resize[0] - int(resize[0]*h/w)

    resized_img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.Tensor(resized_img.transpose(2,0,1))

    if img_tensor.size()[0] > 3:
        print('t')

    if h > w:
        padded_img = F.pad(img_tensor, (0,num_pad,0,0,0,0))
    else:
        padded_img = F.pad(img_tensor, (0,0,0,num_pad,0,0))

    return padded_img, torch.Tensor([size[1], size[0]])

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

def nms(bbox, score, threshold):
    '''
    non-maximum suppression
    bbox (Tensor) : [N, 4]
    score (Tensor) : [N]
    threshold (float)
    '''
    assert int(score.size()[0]) == int(bbox.size()[0]), "size of two arg tensor must be same"
    bbox_number = int(score.size()[0])

    cross_bbox0 = bbox.repeat(bbox_number, 1, 1)
    cross_bbox1 = cross_bbox0.permute(1,0,2)

    cross_IoU = IoU(cross_bbox0, cross_bbox1).squeeze()

    cross_score0 = score.repeat(bbox_number, 1)
    cross_score1 = cross_score0.permute(1,0)

    cross_score_comparision = cross_score1-cross_score0

    # over threshold IoU value
    cross_IoU_bool = cross_IoU > threshold
    
    # less score 
    cross_score_comparision_bool = cross_score_comparision < 0.

    remain_bool = torch.logical_not(torch.logical_and(cross_IoU_bool, cross_score_comparision_bool).sum(dim=1, dtype=torch.bool))

    return bbox[remain_bool]

if __name__ == "__main__":
    testbbox = torch.Tensor([[0., 0., 1., 1.], [0., 0., 1., 1.2], [0., 0., 1.2, 1.], [1., 1., 1., 1.]])
    testscore = torch.Tensor([0.5, 0.7, 0.6, 0.4])
    print(nms(testbbox, testscore, 0.5))


