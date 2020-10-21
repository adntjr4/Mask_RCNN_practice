import torch
import torch.optim as optim


def main():
    a = torch.Tensor([0])
    b = torch.Tensor([0])
    init_lr = 0.1
    otm = optim.SGD([a,b], init_lr)
    sd = optim.lr_scheduler.MultiStepLR(otm, [5,10,15], gamma=0.1)

    for i in range(20):
        print(sd.get_last_lr())
        sd.step()

def save():
    a = torch.Tensor([0])
    b = torch.Tensor([0])
    init_lr = 0.1
    otm = optim.SGD([a,b], init_lr)
    sd = optim.lr_scheduler.MultiStepLR(otm, [5,10,15], gamma=0.1)

    for i in range(2):
        print(sd.get_last_lr())
        sd.step()
    
    torch.save({'epoch': 2,
                'optimizer': otm}, './test_scheduler.pth')

def load():
    a = torch.Tensor([0])
    b = torch.Tensor([0])

    load = torch.load('./test_scheduler.pth')

    epoch = load['epoch']
    otm = load['optimizer']
    sd = optim.lr_scheduler.MultiStepLR(otm, [5,10,15], gamma=0.1, last_epoch=1)

    for i in range(epoch, 20):
        print(sd.get_last_lr())
        sd.step() 

if __name__ == "__main__":
    main()