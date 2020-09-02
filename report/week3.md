# week 3

## 수정 내용

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

## 추가된 내용

- multiGPU setting
- rpn에서 연산 수정
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
- rpn에서 처음에 생성하는 tensor 바꾸기
