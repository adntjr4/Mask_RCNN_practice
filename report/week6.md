# week 6

## Detectron 2 code copy

---
### 세부 알고리즘 수정
- pre-top k, post-top k
    - Before : pre-nms top k
    - After : pre-nms top k & post-nsm top k
- box cliping
    - Before : terminate invaild bbox
    - After : remove out of image part (cliping)

### 저번 주 사진

images | images
:-------:|:-----------:
![gt0](./img/week5/eval_2.jpg) | ![eval0](./img/week5/eval_4.jpg)
![gt1](./img/week5/eval_6.jpg) | ![eval1](./img/week5/eval_8.jpg)
![gt2](./img/week5/eval_10.jpg) | ![eval2](./img/week5/eval_12.jpg)
![gt3](./img/week5/eval_13.jpg) | ![eval3](./img/week5/eval_15.jpg)
![gt4](./img/week5/eval_16.jpg) | ![eval4](./img/week5/eval_18.jpg)

---

### 이번 주 사진

images | images
:-------:|:-----------:
![gt0](./img/week6/test16.jpg) | ![eval0](./img/week6/test19.jpg)
![gt1](./img/week6/test20.jpg) | ![eval1](./img/week6/test26.jpg)
![gt2](./img/week6/test19.jpg) | ![eval2](./img/week6/test30.jpg)
![gt3](./img/week6/test31.jpg) | ![eval3](./img/week6/test32.jpg)
![gt4](./img/week6/test40.jpg) | ![eval4](./img/week6/test43.jpg)
![gt4](./img/week6/test53.jpg) | ![eval4](./img/week6/test57.jpg)
![gt4](./img/week6/test60.jpg) | ![eval4](./img/week6/test65.jpg)
![gt4](./img/week6/test66.jpg) | ![eval4](./img/week6/test72.jpg)
![gt4](./img/week6/test73.jpg) | ![eval4](./img/week6/test74.jpg)
![gt4](./img/week6/test75.jpg) | ![eval4](./img/week6/test78.jpg)
![gt4](./img/week6/test79.jpg) | ![eval4](./img/week6/test89.jpg)
![gt4](./img/week6/test12.jpg) | ![eval4](./img/week6/test21.jpg)
![gt4](./img/week6/test27.jpg) | ![eval4](./img/week6/test28.jpg)


### 현재

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.044  
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.177  
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.008  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.008  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.072  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.097  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.041  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.088  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.090  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.098  
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.203  

### 기존
score_threshold : 0.5, nms_threshold : 0.7  

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.036  
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.140  
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.005  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.030  
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.060  
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.043  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.036  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.112  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087  
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.238  
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.235  


### 문제점
- 성능은 더 낮게 나옴.




## 질문 사항

---

## 현재 버그
- anchor와 loss를 구할 gt를 정하는 곳에 버그가 있음.(가장 높은 IoU)

## TODO

- 이번 주까지 목표
    - [x] opencv transformation matrix
    - [x] get prediction on original images
    - [x] result out as coco format file

    - [x] human detection precision 뽑아내기

    - [ ] RoIPooling 추가
    - [ ] classification network (box head)

    - [ ] Timer, logger 만들기
    - [x] evaluation 함수 만들기
    - [x] human detection AP 구하기 (coco api)

---


3 : sample 256, weight 0.1, constrain : 0.3, 2