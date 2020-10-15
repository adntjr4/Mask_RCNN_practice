# week 7

## 수정 사항

- post top-k 값 조정 : 2000 -> 1000
- regression loss : mean -> sum
- image pre-processing : subtract mean
- freeze 2 front layer of backbone
- more training : 16 epoch -> 24 epoch

## Best result

- proposal score threshold : 0.4
- proposal nms threshold : 0.5

| 종류 |    IoU    |  area  |maxDets| value |
|:----:|:---------:|:------:|:-----:|:-----:|
|  AP  | 0.50:0.95 |  all   |  100  | 0.311 |
|  AP  | 0.50      |  all   |  100  | 0.657 |
|  AP  | 0.75      |  all   |  100  | 0.254 |
|  AP  | 0.50:0.95 | small  |  100  | 0.218 |
|  AP  | 0.50:0.95 | medium |  100  | 0.474 |
|  AP  | 0.50:0.95 | large  |  100  | 0.329 |
|  AR  | 0.50:0.95 |  all   |   1   | 0.119 |
|  AR  | 0.50:0.95 |  all   |   10  | 0.363 |
|  AR  | 0.50:0.95 |  all   |  100  | 0.407 |
|  AR  | 0.50:0.95 | small  |  100  | 0.268 |
|  AR  | 0.50:0.95 | medium |  100  | 0.535 |
|  AR  | 0.50:0.95 | large  |  100  | 0.451 |


## Proposal images

- proposal score threshold : 0.7
- proposal nms threshold : 0.2

| img | img |
|:---:|:---:|
| ![img](./img/week7/test0.jpg) | ![img](./img/week7/test1.jpg) |
| ![img](./img/week7/test2.jpg) | ![img](./img/week7/test3.jpg) |
| ![img](./img/week7/test4.jpg) | ![img](./img/week7/test5.jpg) |
| ![img](./img/week7/test6.jpg) | ![img](./img/week7/test7.jpg) |
| ![img](./img/week7/test8.jpg) | ![img](./img/week7/test9.jpg) |
| ![img](./img/week7/test10.jpg) | ![img](./img/week7/test11.jpg) |
| ![img](./img/week7/test12.jpg) | ![img](./img/week7/test13.jpg) |
| ![img](./img/week7/test14.jpg) | ![img](./img/week7/test15.jpg) |
| ![img](./img/week7/test16.jpg) | ![img](./img/week7/test17.jpg) |
| ![img](./img/week7/test18.jpg) | ![img](./img/week7/test19.jpg) |
| ![img](./img/week7/test20.jpg) | ![img](./img/week7/test21.jpg) |
| ![img](./img/week7/test22.jpg) | ![img](./img/week7/test23.jpg) |
| ![img](./img/week7/test24.jpg) | ![img](./img/week7/test25.jpg) |
| ![img](./img/week7/test26.jpg) | ![img](./img/week7/test27.jpg) |
| ![img](./img/week7/test28.jpg) | ![img](./img/week7/test29.jpg) |
| ![img](./img/week7/test30.jpg) | ![img](./img/week7/test31.jpg) |
| ![img](./img/week7/test32.jpg) | ![img](./img/week7/test33.jpg) |
| ![img](./img/week7/test34.jpg) | ![img](./img/week7/test35.jpg) |
| ![img](./img/week7/test36.jpg) | ![img](./img/week7/test37.jpg) |
| ![img](./img/week7/test38.jpg) | ![img](./img/week7/test39.jpg) |
| ![img](./img/week7/test40.jpg) | ![img](./img/week7/test41.jpg) |
| ![img](./img/week7/test42.jpg) | ![img](./img/week7/test43.jpg) |
| ![img](./img/week7/test44.jpg) | ![img](./img/week7/test45.jpg) |
| ![img](./img/week7/test46.jpg) | ![img](./img/week7/test47.jpg) |
| ![img](./img/week7/test48.jpg) | ![img](./img/week7/test49.jpg) |
| ![img](./img/week7/test50.jpg) | ![img](./img/week7/test51.jpg) |
| ![img](./img/week7/test52.jpg) | ![img](./img/week7/test53.jpg) |
| ![img](./img/week7/test54.jpg) | ![img](./img/week7/test55.jpg) |
| ![img](./img/week7/test56.jpg) | ![img](./img/week7/test57.jpg) |
| ![img](./img/week7/test58.jpg) | ![img](./img/week7/test59.jpg) |
| ![img](./img/week7/test60.jpg) | ![img](./img/week7/test61.jpg) |
| ![img](./img/week7/test62.jpg) | ![img](./img/week7/test63.jpg) |
| ![img](./img/week7/test64.jpg) | ![img](./img/week7/test65.jpg) |
| ![img](./img/week7/test66.jpg) | ![img](./img/week7/test67.jpg) |
| ![img](./img/week7/test68.jpg) | ![img](./img/week7/test69.jpg) |
| ![img](./img/week7/test70.jpg) | ![img](./img/week7/test71.jpg) |
| ![img](./img/week7/test72.jpg) | ![img](./img/week7/test73.jpg) |
| ![img](./img/week7/test74.jpg) | ![img](./img/week7/test75.jpg) |
| ![img](./img/week7/test76.jpg) | ![img](./img/week7/test77.jpg) |
| ![img](./img/week7/test78.jpg) | ![img](./img/week7/test79.jpg) |