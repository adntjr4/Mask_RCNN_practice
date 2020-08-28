# Mask_RCNN_practice

[2020 SNU CVlab freshman tutorial]  
Mask R-CNN implementation  


## week 2

### last code review

- train.py에서 coco라는 말이 보이지 않게  
[train.py](./train.py)

- COCO API 사용  
[data_set.py](./data_set/data_set.py)  
이미지 정보와 annotation을 불러올 때 coco api를 사용하고  
이미지를 불러올 때는 opencv를 사용함.  

__getitem__()
```
img_object = self.coco.loadImgs(self.img_id_list[index])[0]
img = cv2.imread('%s/%s/%s'%(self.data_dir, self.data_type, img_object['file_name']))
img_tensor, resize = img_process(img, self.input_size)

img_size = torch.Tensor([img_object['height'], img_object['width']])
anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=self.img_id_list[index]))
```

- file, variable naming  
최대한 노력했지만...  

- 시간관련된 log 출력 구현을 아직 못 함.  

### This week work
- [trainer.py](./trainer/trainer.py)  
train()함수는 다음과 같은 순서로 진행함  
    - resume flag를 읽고 last checkpoint를 불러옴
    - epoch ~ max epoch 각각의 epoch를 training
    - 10 epoch마다 checkpoint save

train_1epoch()  
    - data_loader의 모든 batch에 대해서
        - forward
        - get losses (self.criterion())
        - backward

RPN에서 cls loss와 reg loss를 구함  
- 
