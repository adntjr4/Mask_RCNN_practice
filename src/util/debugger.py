
import cv2

from .util import img_process, draw_boxes

def debug_draw_bbox(bbox, name):
    img_dir = 'data/coco/val2017/000000397133.jpg'
    img = cv2.imread(img_dir)
    input_size = (1024, 1024)
    img_tensor, _ = img_process(img, input_size)

    boxed_img = draw_boxes(img_tensor, bbox)
    cv2.imwrite('data/tmp/%s.jpg'%name, boxed_img)

def debug_draw_bbox2(bbox1, bbox2, name):
    img_dir = 'data/coco/val2017/000000397133.jpg'
    img = cv2.imread(img_dir)
    input_size = (1024, 1024)
    img_tensor, _ = img_process(img, input_size)

    boxed_img = draw_boxes(img_tensor, bbox1)
    boxed_img = draw_boxes(boxed_img, bbox2, (0,0,255))
    cv2.imwrite('data/tmp/%s.jpg'%name, boxed_img)