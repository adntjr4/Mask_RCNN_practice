# week 8

## 수정 사항

### 중요 수정 사항
- roi_head layer 만듬
- roi align은 torchvision의 roi_align() 함수를 사용. (detectron2에서 사용하고 있는 RoIAlignV2(위치가 정확하게 보정된)를 사용)
- loss 구하는 logic 함수들은 RPN에서 사용하던 함수를 수정하여 사용.

### 안중요 수정 사항
- backbone에서 학습시키지 않는 영역의 require_grad를 False를 안 시켜줌 (메모리 낭비)
- warmpup scheduler를 만들었는데, 아직 사용 X
- RPN에서 재탕할 때, labeling 함수에서 가장 IOU를 높은 것은 무시하고 무조건 threshold로 함.
- detectron2의 heuristic 방법인 proposal에 gt_bbox를 넣어줌.
- roi_align에서의 중요한 버그들을 수정함
- 1 proposals를 feature map에 맞게 mapping을 안 시켜줬었음.
- 2 xyxy좌표를 넣어줘야하는데, xywh 좌표를 넣어줌.
- 3 box clipping 후 width or height가 0인 proposal들에 대한 처리가 없었음.

## Result

- 일단 epoch 4에서는 regression의 효과는 매우 잘 되는 것 같다. 근데 AP 성능이 그냥 떨어짐.
- 실제 human이 아닌 것이나 주변 hard negative들도 human으로 인식함.
- loss balance가 잘 안 맞는 것이 원인으로 생각하여 roi_head regression loss weight을 조금 줄이고 다시 실험 중
- detectron2의 코드를 좀 더 보거나 실제로 학습을 돌려봐서 loss 값을 비교해볼 예정.

| 종류 |    IoU    |  area  |maxDets| w/o roi_head (last week) | w/ roi_head |
|:----:|:---------:|:------:|:-----:|:------------------------:|:-----------:|
|  AP  | 0.50:0.95 |  all   |  100  |           0.311          |    0.245    |
|  AP  | 0.50      |  all   |  100  |           0.657          |    0.434    |
|  AP  | 0.75      |  all   |  100  |           0.254          |  __0.250__  |
|  AP  | 0.50:0.95 | small  |  100  |           0.218          |    0.105    |
|  AP  | 0.50:0.95 | medium |  100  |           0.474          |    0.335    |
|  AP  | 0.50:0.95 | large  |  100  |           0.329          |  __0.372__  |
|  AR  | 0.50:0.95 |  all   |   1   |           0.119          |  __0.125__  |
|  AR  | 0.50:0.95 |  all   |   10  |           0.363          |    0.327    |
|  AR  | 0.50:0.95 |  all   |  100  |           0.407          |    0.350    |
|  AR  | 0.50:0.95 | small  |  100  |           0.268          |    0.148    |
|  AR  | 0.50:0.95 | medium |  100  |           0.535          |    0.439    |
|  AR  | 0.50:0.95 | large  |  100  |           0.451          |  __0.548__  |

## Result

| 종류 |    IoU    |  area  |maxDets| value |
|:----:|:---------:|:------:|:-----:|:-----:|
|  AP  | 0.50:0.95 |  all   |  100  | 0.395 |
|  AP  | 0.50      |  all   |  100  | 0.630 |
|  AP  | 0.75      |  all   |  100  | 0.424 |
|  AP  | 0.50:0.95 | small  |  100  | 0.165 |
|  AP  | 0.50:0.95 | medium |  100  | 0.508 |
|  AP  | 0.50:0.95 | large  |  100  | 0.606 |
|  AR  | 0.50:0.95 |  all   |   1   | 0.166 |
|  AR  | 0.50:0.95 |  all   |   10  | 0.421 |
|  AR  | 0.50:0.95 |  all   |  100  | 0.439 |
|  AR  | 0.50:0.95 | small  |  100  | 0.184 |
|  AR  | 0.50:0.95 | medium |  100  | 0.557 |
|  AR  | 0.50:0.95 | large  |  100  | 0.682 |

## Proposal images

| img | img |
|:---:|:---:|
| ![img](./img/week8/test0.jpg) | ![img](./img/week8/test1.jpg) |
| ![img](./img/week8/test2.jpg) | ![img](./img/week8/test3.jpg) |
| ![img](./img/week8/test4.jpg) | ![img](./img/week8/test5.jpg) |
| ![img](./img/week8/test6.jpg) | ![img](./img/week8/test7.jpg) |
| ![img](./img/week8/test8.jpg) | ![img](./img/week8/test9.jpg) |
| ![img](./img/week8/test10.jpg) | ![img](./img/week8/test11.jpg) |
| ![img](./img/week8/test12.jpg) | ![img](./img/week8/test13.jpg) |
| ![img](./img/week8/test14.jpg) | ![img](./img/week8/test15.jpg) |
| ![img](./img/week8/test16.jpg) | ![img](./img/week8/test17.jpg) |
| ![img](./img/week8/test18.jpg) | ![img](./img/week8/test19.jpg) |
| ![img](./img/week8/test20.jpg) | ![img](./img/week8/test21.jpg) |
| ![img](./img/week8/test22.jpg) | ![img](./img/week8/test23.jpg) |
| ![img](./img/week8/test24.jpg) | ![img](./img/week8/test25.jpg) |
| ![img](./img/week8/test26.jpg) | ![img](./img/week8/test27.jpg) |
| ![img](./img/week8/test28.jpg) | ![img](./img/week8/test29.jpg) |
| ![img](./img/week8/test30.jpg) | ![img](./img/week8/test31.jpg) |
| ![img](./img/week8/test32.jpg) | ![img](./img/week8/test33.jpg) |
| ![img](./img/week8/test34.jpg) | ![img](./img/week8/test35.jpg) |
| ![img](./img/week8/test36.jpg) | ![img](./img/week8/test37.jpg) |
| ![img](./img/week8/test38.jpg) | ![img](./img/week8/test39.jpg) |
| ![img](./img/week8/test40.jpg) | ![img](./img/week8/test41.jpg) |
| ![img](./img/week8/test42.jpg) | ![img](./img/week8/test43.jpg) |
| ![img](./img/week8/test44.jpg) | ![img](./img/week8/test45.jpg) |
| ![img](./img/week8/test46.jpg) | ![img](./img/week8/test47.jpg) |
| ![img](./img/week8/test48.jpg) | ![img](./img/week8/test49.jpg) |
| ![img](./img/week8/test50.jpg) | ![img](./img/week8/test51.jpg) |
| ![img](./img/week8/test52.jpg) | ![img](./img/week8/test53.jpg) |
| ![img](./img/week8/test54.jpg) | ![img](./img/week8/test55.jpg) |
| ![img](./img/week8/test56.jpg) | ![img](./img/week8/test57.jpg) |
| ![img](./img/week8/test58.jpg) | ![img](./img/week8/test59.jpg) |
| ![img](./img/week8/test60.jpg) | ![img](./img/week8/test61.jpg) |
| ![img](./img/week8/test62.jpg) | ![img](./img/week8/test63.jpg) |
| ![img](./img/week8/test64.jpg) | ![img](./img/week8/test65.jpg) |
| ![img](./img/week8/test66.jpg) | ![img](./img/week8/test67.jpg) |
| ![img](./img/week8/test68.jpg) | ![img](./img/week8/test69.jpg) |
| ![img](./img/week8/test70.jpg) | ![img](./img/week8/test71.jpg) |
| ![img](./img/week8/test72.jpg) | ![img](./img/week8/test73.jpg) |
| ![img](./img/week8/test74.jpg) | ![img](./img/week8/test75.jpg) |
| ![img](./img/week8/test76.jpg) | ![img](./img/week8/test77.jpg) |
| ![img](./img/week8/test78.jpg) | ![img](./img/week8/test79.jpg) |

# Question
