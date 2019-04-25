if __name__ == '__main__':
    import os
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torchvision.utils import save_image
    from cGAN_models import Generator, Discriminator
    import datetime

    BATCH_SIZE = 16
    EPOCHS = 10
    IMAGE_DIR = 'C:\\Users\\KangwooYi\\PycharmProjects\\PytorchLearning\\GANs\\checkpoints\\MNIST\\Images\\Training'
    IMAGE_SIZE = 28
    ITER_DISPLAY = 100
    ITER_REPORT = 100
    LATENT_DIM = 100
    MODEL_DIR = 'C:\\Users\\KangwooYi\\PycharmProjects\\PytorchLearning\\GANs\\checkpoints\\MNIST\\Models'
    OUT_CHANNEL = 1
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    transforms = Compose([ToTensor(), Normalize(mean=[0.5], std=[0.5])])
    dataset = MNIST(root='datasets', train=True, transform=transforms, download=True)
    # ToTensor(): 데이터를 0~1 사이로 Normalization. Output에서의 activation이 이 SIgmoid 이므로(0~1)
    # Discriminator 에서 인풋 데이터와 Generated 데이터의 공정한 비교를 위하여 같은 범위로 Normaliation 해야함.
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    D = Discriminator()
    G = Generator()
    print(D)  # Discriminator의 내부 구조와 웨이트를 알 수 있음
    print(G)  # Genertor의 내부 구조와 웨이트를 알 수 있음

    # 모델 선언 후 LOSS 정의하는게 좋으나, 여기서는 pytorch가 제공하는 LOSS가 아닌 직접 제작한 LOSS를 쓸 것이므로 훈련 과정중에 LOSS 정의

    # D와 G가 쓸 optimizer 선언
    optim_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9))
    optim_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9))

    st = datetime.datetime.now()
    iter = 0
    for epoch in range(EPOCHS):
        for i, data in enumerate(data_loader):
            iter += 1

            real = data[0].view(BATCH_SIZE, -1)
            label = data[1].view(BATCH_SIZE, 1)
            one_hot = torch.zeros(BATCH_SIZE, 10)
            one_hot.scatter_(dim=1, index=label, src=torch.ones(BATCH_SIZE, 1))

            z = torch.rand(BATCH_SIZE, LATENT_DIM)

            fake = G(z, one_hot)
            real = data[0].view(BATCH_SIZE, -1)

            # LOSS 직접 지정. BInary Cross Entropy.  // fake.detach(): fake에게는 loss를 이용한 업데이트가 가지 않음
            # 어차피 optim_D는 D.parameter만 받기 때문에 fake=G()를 업데이트 시키지 못하지만, 계산 낭비가 일어남
            loss_D = -torch.mean(torch.log(D(real, one_hot)) + torch.log(1 - D(fake.detach(), one_hot)))
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

            loss_G = -torch.mean(torch.log(D(fake, one_hot)))  # Non saturating loss
            # For saturaing loss, loss_G = torch.mean(torch.log(1-D(fake)))
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if iter % ITER_DISPLAY == 0:
                fake = fake.view(BATCH_SIZE, OUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
                real = real.view(BATCH_SIZE, OUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
                # nrows = 한 행 당 뽑을 이미지 수. 배치사이즈가 16이니, nrows가 4면 4*4 = 16개의 이미지 생성저장
                # normalize = 데이터를 -1~+1 ==> 0~255로 바꿔줌.
                save_image(fake, IMAGE_DIR + '/{}_fake.png'.format(epoch + 1), nrow=4, normalize=True)
                save_image(real, IMAGE_DIR + '/{}_real.png'.format(epoch + 1), nrow=4, normalize=True)

            if iter % ITER_REPORT == 0:
                print("Epoch: {} Iter: {} Loss D: {:.{prec}} Loss G: {:.{prec}}"
                      .format(epoch + 1, iter, loss_D.detach().item(), loss_G.detach().item(), prec=4))

    torch.save(D, join(MODEL_DIR, 'Latest_D.pt'))
    torch.save(G, join(MODEL_DIR, 'Latest_G.pt'))
    print("Total time taken: ", datetime.datetime.now() - st)
