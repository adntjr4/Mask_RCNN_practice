# week 4

## 이번 주 작업

- rpn에서의 process를 바꿈.
- training
    - 기존 : anchor → bbox regression → labeling → sampling → loss → backward
    - 바꾼 : anchor → select top k → bbox regression → NMS → labeling → sampling → loss → backward
- evaluation
    - 기존 : anchor → bbox regression → score threshold → NMS → RoI
    - 바꾼 : anchor → select top k → bbox regression → NMS → "score threshold" or "select top N" → RoI
- 바꾸면서 코드 구조를 좀 더 체계적으로 바꿈.

## 질문 사항

---

## TODO

- 이번 주까지 목표
    - [x] config file 구조 수정하기
    - [x] trainer에 기능 on/off 넣기

    - [ ] rpn process 재정립
    - [ ] basemodel 테스트
    - [ ] rpn for human dectection
    - [ ] multiGPU setting

    - [ ] fpn debug w/o box regression
    - [ ] fpn을 위한 rpn 함수 구조 체계화
    - [ ] 시간 측정
    - [ ] fpn debug

    - [ ] opencv affine transformation 만들기

    - [ ] RoIPooling 추가
    - [ ] classification network (box head)

- 다음 주까지 목표

- 차후
    - [ ] fpn에서 region proposal의 heuristic method 적용
    - [ ] RoIAlign 추가
    - [ ] box head의 box regression loss와 class loss
    - [ ] box head에서의 heuristic method를 적용
    - [ ] training 




