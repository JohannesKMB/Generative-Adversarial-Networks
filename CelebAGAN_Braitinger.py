import torch.nn as nn
import torch
import os
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mpl_toolkits.axes_grid1 import ImageGrid

PATH = r'/work/hertrich/braitinger'

img_shape = (32, 32, 3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_folder = r'img_align_celeba'
data_folder = os.path.join(PATH, data_folder)

for _, _, files in os.walk(data_folder):
    files = files


def get_x_batch(batch_size):
    imgs = []
    for _ in range(batch_size):
        rand_no = int(len(files) * np.random.rand(1))
        x = load_img(os.path.join(data_folder, files[rand_no]), target_size=img_shape[:2])
        x = img_to_array(x).reshape(3, 32, 32) / 255.
        imgs += [x]
    return torch.tensor(imgs, dtype=torch.float32)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 8 * 8),
            nn.ReLU()
        )
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear_layers(x)
        x = x.view(-1, 128, 8, 8)
        return self.conv_layers(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.cnn_layers = nn.Sequential(
            self._cnn_block(in_channels=3, out_channels=32, kernel_size=3, stride=2),
            self._cnn_block(32, 64, 3, 2),
            self._cnn_block(64, 128, 3, 1),
            nn.Flatten()
        )
        self.linear_layers = nn.Sequential(nn.Linear(6272, 1024),
                                           nn.ReLU(),
                                           nn.Linear(1024, 1),
                                           nn.Sigmoid())

    def _cnn_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=stride)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.linear_layers(x)
        return x


z_dim = 400
lr = 7e-5
batch_size = 128

gen = Generator(z_dim).to(device)
optimizer = optim.Adam(list(gen.parameters()), lr=lr, betas=(0.5, 0.999))

discr = Discriminator().to(device)
optimizer2 = optim.Adam(discr.parameters(), lr=lr, betas=(0.5, 0.999))


def ms_criterion(noise, noise2, fake, fake2):
    return 1 / (torch.mean(torch.abs(fake - fake2)) / torch.mean(torch.abs(noise - noise2)) + 1e-5)


def train_SNF_epoch_new(epoch):
    for _ in range(5):
        z = torch.randn((batch_size, z_dim)).to(device)
        z2 = torch.randn((batch_size, z_dim)).to(device)

        loss = 0
        out_for = gen(z).to(device)
        out_for2 = gen(z2).to(device)
        disc_fake = discr(out_for).to(device)
        loss_gen = (-torch.mean(torch.log(disc_fake + 1e-10)) + 1 * ms_criterion(z, z2, out_for, out_for2)).to(device)

        optimizer.zero_grad()
        loss_gen.backward()
        optimizer.step()

        for _ in range(1):
            x = get_x_batch(batch_size).to(device)
            z = torch.randn((batch_size, z_dim)).to(device)

            out_for = gen(z).to(device)

            disc_real = discr(x).to(device)
            disc_fake = discr(out_for).to(device)
            loss_disc = (-torch.mean(torch.log(disc_real + 1e-10) + torch.log(1 - disc_fake + 1e-10))).to(device)
            optimizer2.zero_grad()
            loss_disc.backward()
            optimizer2.step()
        # print('disc classification of fake samples')
        # print(torch.mean(discr(out_for)))
        # print('disc classification of correct samples')
        # print(torch.mean(discr(x)))
        if epoch and not epoch % 1000:
            torch.save(discr.state_dict(), os.path.join(PATH, f'celebdisc_{epoch}_epochs.pt'))
            torch.save(gen.state_dict(), os.path.join(PATH, f'celebgen_{epoch}_epochs.pt'))
            torch.save(optimizer.state_dict(), os.path.join(PATH, f'celebgen_opt_{epoch}_epochs.pt'))
            torch.save(optimizer2.state_dict(), os.path.join(PATH, f'celebdisc_opt_{epoch}_epochs.pt'))
    return loss


def main():
    for i in range(40001):
        #print(f'epoch: {i}')
        train_SNF_epoch_new(i)


if __name__ == '__main__':
    main()


'''PLOTTING'''
'''
gene = gen.eval()

bs = 20
x = get_x_batch(bs)[0].cpu().data.numpy()

plt.figure()
plt.imshow(x.reshape(32,32,3))

plt.figure()
with torch.no_grad():
    z = torch.randn((bs,z_dim))

    out_for = gene(z).cpu().data.numpy()    
    out_for1 = []
    for i in range(4):
            for j in range(5):
                out_for1 += [out_for[5*i+j].reshape(3,32,32).reshape(32,32,3)]
    fig = plt.figure(figsize=(20, 20))

    grid = ImageGrid(fig,111,  # similar to subplot(111)
                 nrows_ncols=(4, 5),  # creates 2x2 grid of axes
                 axes_pad=0,  # pad between axes in inch.
                 )
    for ax, im in zip(grid, out_for1):
        # Iterating over the grid returns the Axes.
        ax.tick_params(
    axis='both',          # changes apply to the axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,# ticks along the top edge are off
    labelbottom=False,
    labelleft=False)
        ax.imshow(im)
    #plt.savefig('gen_1kepochs.png')
    plt.show()
'''
