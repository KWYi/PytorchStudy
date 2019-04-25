if __name__ == '__main__':
    import os
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    from Pix2Pix_models import Discriminator, Generator
    from pipeline import CustomDataset

    DEVICE = torch.device("gpu:0" if torch.cuda.is_available() else "cpu:0")
    DIR_IMAGE = 'pix2pix\\checkpoint\\IMAGE\\'
    DIR_MODEL = 'pix2pix\\checkpoint\\MODEL\\'
    EPOCHS = 10
    EPOCH_SAVE = 5
    ITER_DISPLAY = 10

    os.makedirs(DIR_IMAGE) if not os.path.isdir(DIR_IMAGE) else None
    os.makedirs(DIR_MODEL) if not os.path.isdir(DIR_MODEL) else None

    dataset = CustomDataset(root='dataset\\images\\', crop_size=128, flip=True)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=1)

    D = Discriminator()
    G = Generator()

    GAN_Loss = nn.BCELoss()
    L1_Loss = nn.L1Loss()

    D_optim = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))

    total_iter = 0
    for epoch in range(EPOCHS):
        for input, target in data_loader:
            total_iter += 1
            input, target = input.to(DEVICE), target.to(DEVICE)
            fake = G(input)

            valid_fake = D(torch.cat((fake.detach(), input), dim=1))
            valid_real = D(torch.cat((target, input), dim=1))
            fake_loss = GAN_Loss(valid_fake, torch.zeros_like(valid_fake).to(DEVICE))
            real_loss = GAN_Loss(valid_real, torch.ones_like(valid_real).to(DEVICE))
            D_loss = (fake_loss + real_loss) * 0.5

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            valid_fake = D(torch.cat((fake, input), dim=1))
            G_loss = GAN_Loss(valid_fake, torch.ones_like(valid_fake).to(DEVICE)) + L1_Loss(fake, target)

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            print("Epoch: {}, Iter: {}, D_loss: {:.{prec}}, G_loss: {:.{prec}}"
                  .format(epoch + 1, total_iter, D_loss.detach().item(), G_loss.detach().item(), prec=4))

            if total_iter % ITER_DISPLAY == 0:
                save_image(fake.detach(), os.path.join(DIR_IMAGE, '{}_fake.png'.format(epoch + 1)), normalize=True,
                           nrow=1)
                save_image(target.detach(), os.path.join(DIR_IMAGE, '{}_real.png'.format(epoch + 1)), normalize=True,
                           nrow=1)

        if (epoch + 1) % EPOCH_SAVE == 0:
            torch.save(G.state_dict(), os.path.join(DIR_MODEL, '{}_G.pt'.format(epoch + 1)))
            torch.save(D.state_dict(), os.path.join(DIR_MODEL, '{}_D.pt'.format(epoch + 1)))