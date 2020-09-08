# week 3

## 저번 주 리뷰 수정

- loss 합으로 바꾸기, iteration마다 loss 출력  
    ```
    trainer.py

    for loss_name in losses:
        losses[loss_name].backward(retain_graph=True)
        
                ▼

    total_loss = sum(v for v in losses.values())
    total_loss.backward()
    ```
- imread os.path
    ```
    data_set.py

    img = cv2.imread('%s/%s/%s'%(self.data_dir, self.data_type, img_object['file_name']))

                ▼

    img = cv2.imread(path.join(self.data_dir, self.data_type, img_object['file_name']))
    ```
- etc (변수명 수정)

---

## 추가된 내용

- FPN
    - 기존 backbone은 두고 fpn.py를 만듬.
    - 기존 backbone에만 호환되던 rpn도 수정함.
    - 하지만 아직 작동이 안 됨...

        - rpn의 conv layer는 feature 마다? 공유?
        - box regression loss가 일정 수치에서 멈추고 학습이 안 되는 경우가 있음
        - box regression을 disable 시키고 debug

- rpn에서 일부 연산 수정
    ```
    _, b = torch.max(var, dim=1)
                ▼
    b = torch.argmax(var, dim=1)
    ```
    ```
    axis_or = torch.sum(bool_tensor, dim=1, dtype=torch.bool)
                ▼
    axis_or = bool_tensor.any(1)
    ```
    ```
    zeros((h,w)).cuda()
                ▼
    new_zeros((h,w))
    ```

- [질문] config file 구조
    - 현재 한 file에 data_set, training, evaluation, model configuration 정보가 함께 있음.
    - 다 나누는 구조? 다른 파일을 불러오는 구조?

- [질문] 시간 측정
    - 다른 코드를 보니 time.time() 말고 다른 것을 쓰던데...
    - 다른 시간 측정 라이브러리?
    - 그냥 time 쓰면 됨.

---

## TODO

- 다음 주까지 목표
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


- 차후
    - [ ] fpn에서 region proposal의 heuristic method 적용
    - [ ] RoIAlign 추가
    - [ ] box head의 box regression loss와 class loss
    - [ ] box head에서의 heuristic method를 적용
    - [ ] training 




