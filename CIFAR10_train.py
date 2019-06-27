if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torch.utils.data import DataLoader
    from CIFAR10_models import PlainNetwork, ResidualNetwork
    from torch.optim.lr_scheduler import MultiStepLR
    from CIFAR10_pipeline import CustomCIFAR10

    BATCH_SIZE = 16

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    # transform = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # dataset = CIFAR10(root='\\datasets', train=True, transform=transform, download=True)
    dataset = CustomCIFAR10(train=True)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=True)

    pnet = PlainNetwork()
    resnet = ResidualNetwork()

    criterion = nn.CrossEntropyLoss()

    optim_resnet = torch.optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # weight_decay: Loss의 weight가 너무 커지면 1e-4로 줄여버림
    scheduler_resnet = MultiStepLR(optim_resnet, milestones=[320000, 480000], gamma=0.1)
    # MultiStepLR: LR을 변화시키는 함수. milestone 안에 수치들 만큼 학습을 했으면, lr = lr*gamma 로 러닝레이트를 변화시킴

    iter_total = 0
    ITER=5
    while iter_total < ITER:
        for input, label in data_loader:
            iter_total += 1
            input, label = input.to(device), label.to(device)

            output = resnet(input)

            loss = criterion(output, label)

            optim_resnet.zero_grad()
            loss.backward()
            optim_resnet.step()

            scheduler_resnet.step()

            n_correct_answers = torch.eq(output.argmax(dim=1), label).sum()

            print("Loss: {:.{prec}}, Acc: {:.{prec}}"
                 .format(loss.detach().item(), (float(n_correct_answers.item()) / BATCH_SIZE) *100, prec=4))

            if iter_total == ITER:
                break