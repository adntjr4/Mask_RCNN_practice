from PIL import Image, ImageDraw 

def draw_boxes(img, boxes):
    box_draw = ImageDraw.Draw(img)
    for box in boxes:
        x, y, w, h = box
        box_draw.rectangle([x, y, x+w, y+h])
