
import cv2

from .util import img_process, draw_boxes

def debug_draw_bbox(img_dir, bbox, name):
    img = cv2.imread(img_dir)
    input_size = (1024, 1024)
    img_tensor, _ = img_process(img, input_size)

    boxed_img = draw_boxes(img_tensor, bbox)
    cv2.imwrite('data/tmp/%s.jpg'%name, boxed_img)

def debug_draw_bbox2(img_dir, bbox1, bbox2, name):
    img = cv2.imread(img_dir)
    input_size = (1024, 1024)
    img_tensor, _ = img_process(img, input_size)

    boxed_img = draw_boxes(img_tensor, bbox1)
    boxed_img = draw_boxes(boxed_img, bbox2, (0,0,255))
    cv2.imwrite('data/tmp/%s.jpg'%name, boxed_img)

def debug_draw_bbox_cv_img(img, bbox, name):
    boxed_img = draw_boxes(img, bbox)
    cv2.imwrite('data/saved/result/%s.jpg'%name, boxed_img)