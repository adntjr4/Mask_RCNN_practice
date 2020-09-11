# week 4

## 이번 주 작업

### ● rpn에서의 process를 바꿈.
  

### ● training
```
- 기존 : anchor → bbox regression → labeling → sampling → loss → backward
- 바꾼 : anchor → select top k → bbox regression → NMS → labeling → sampling → loss → backward
```

### ● evaluation
```
- 기존 : anchor → bbox regression → score threshold → NMS → RoI
- 바꾼 : anchor → select top k → bbox regression → NMS → "score threshold" or "select top N" → RoI
```

### ● 바꾸면서 코드 구조를 좀 더 체계적으로 바꿈.

### ● critical한 bug 수정
- 정해진 수의 training sample 고를 때 image 별로 안 고르고 batch에서 고름.
- box regression의 loss를 구할 때 box regression을 하고 난 anchor로 구했었음.
- labeling할 때 nms에서 걸러진 anchor에서 가장 IoU가 높은 것으로 구해야하는데 nms로 거르기 전 anchor들도 포함되어 있었음.
- 하나의 gt_bbox와 가장 IoU가 높은 anchor를 구할 때, gt_bbox tensor의 pad도 실제 anchor로 생각하여 계산 했었음.

## Region proposal (val2017 image 120개)

Ground Truth | Region Proposal
:-------:|:-----------:
![gt0](./img/week4/gt0.jpg) | ![eval0](./img/week4/RPN_evaluation0_120.jpg)
![gt1](./img/week4/gt1.jpg) | ![eval1](./img/week4/RPN_evaluation1_120.jpg)
![gt2](./img/week4/gt2.jpg) | ![eval2](./img/week4/RPN_evaluation2_120.jpg)
![gt3](./img/week4/gt3.jpg) | ![eval3](./img/week4/RPN_evaluation3_120.jpg)
![gt4](./img/week4/gt4.jpg) | ![eval4](./img/week4/RPN_evaluation4_120.jpg)
![gt5](./img/week4/gt5.jpg) | ![eval5](./img/week4/RPN_evaluation5_120.jpg)
![gt6](./img/week4/gt6.jpg) | ![eval6](./img/week4/RPN_evaluation6_120.jpg)
![gt7](./img/week4/gt7.jpg) | ![eval7](./img/week4/RPN_evaluation7_120.jpg)
![gt8](./img/week4/gt8.jpg) | ![eval8](./img/week4/RPN_evaluation8_120.jpg)
![gt9](./img/week4/gt9.jpg) | ![eval9](./img/week4/RPN_evaluation9_120.jpg)

---

## Human detection (train2017)

- 어제 22:30 ~ 오늘 13:30 까지 (15h) 6epoch를 돌린 결과를 val2017 앞의 10개 이미지 테스트

Ground Truth(human) | Region Proposal(human)
:-------:|:-----------:
![human_gt0](./img/week4/human_gt0.jpg) | ![human_eval0](./img/week4/RPN_human_evaluation0.jpg)
![human_gt1](./img/week4/human_gt1.jpg) | ![human_eval1](./img/week4/RPN_human_evaluation1.jpg)
![human_gt2](./img/week4/human_gt2.jpg) | ![human_eval2](./img/week4/RPN_human_evaluation2.jpg)
![human_gt3](./img/week4/human_gt3.jpg) | ![human_eval3](./img/week4/RPN_human_evaluation3.jpg)
![human_gt4](./img/week4/human_gt4.jpg) | ![human_eval4](./img/week4/RPN_human_evaluation4.jpg)
![human_gt5](./img/week4/human_gt5.jpg) | ![human_eval5](./img/week4/RPN_human_evaluation5.jpg)
![human_gt6](./img/week4/human_gt6.jpg) | ![human_eval6](./img/week4/RPN_human_evaluation6.jpg)
![human_gt7](./img/week4/human_gt7.jpg) | ![human_eval7](./img/week4/RPN_human_evaluation7.jpg)
![human_gt8](./img/week4/human_gt8.jpg) | ![human_eval8](./img/week4/RPN_human_evaluation8.jpg)
![human_gt9](./img/week4/human_gt9.jpg) | ![human_eval9](./img/week4/RPN_human_evaluation9.jpg)

---

## 질문 사항

- smooth_L1_loss

DataParallel을 쓰는 경우 gt가 없는 이미지가 발생 -> 이런 경우는 loss를 0.0을 채움  
smooth_L1_loss은 loss의 최소값이 -0.5인데 어떻게 해야하는가.

![week4_img0](./img/week4/week4_img0.jpg)

- training log 출력. 어떤 내용?

- evaluation 함수: precision만? batch 별로?

---

## 현재 버그
- anchor와 loss를 구할 gt를 정하는 곳에 버그가 있음.(가장 높은 IoU)

## TODO

- 이번 주까지 목표
    - [ ] human detection precision 뽑아내기

    - [ ] RoIPooling 추가
    - [ ] classification network (box head)

    - [ ] Timer, logger 만들기
    - [ ] evaluation 함수 만들기
    - [ ] human detection AP 구하기 (coco api)

    - [ ] opencv affine transformation 만들기

    - [ ] RoIPooling 추가
    - [ ] classification network (box head)

- 다음 주까지 목표

---

- 차후
    - [ ] fpn debug w/o box regression
    - [x] fpn을 위한 rpn 함수 구조 체계화
    - [ ] 시간 측정
    - [ ] fpn debug
    - [ ] fpn에서 region proposal의 heuristic method 적용
    - [ ] RoIAlign 추가
    - [ ] box head의 box regression loss와 class loss
    - [ ] box head에서의 heuristic method를 적용
    - [ ] training 




