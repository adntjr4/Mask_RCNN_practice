# week 4

## 이번 주 작업

### Full epoch training

FPN : RPN training (6 epoch)  
-> 8GPU, 2 image per GPU, 40k iters -> 640k images -> 640k / 118k ~ 5.4 epoch  

FPN : Object detection training (11 epoch)  
-> 8GPU, 2 image per GPU, 80k iters -> 1280k images -> 128k / 118k ~ 10.8 epoch  

### Evaluation function 작성



## 질문 사항

---

## 현재 버그
- anchor와 loss를 구할 gt를 정하는 곳에 버그가 있음.(가장 높은 IoU)

## TODO

- 이번 주까지 목표
    - [ ] opencv transformation matrix
    - [ ] get prediction on original images
    - [ ] result out as coco format file

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




