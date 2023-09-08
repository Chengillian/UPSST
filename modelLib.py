import torch.nn as nn
import torch.optim as optim
from environment import *


# Input: Grayscale images: 64 x 64 x 1

# Reduce image size to 1/4 x 1/4
class encodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(out_channels)
        self.last = last

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        return x


# Expand image size to 4 x 4
class decodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        self.convTrans1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.convTrans2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.last = last

    def forward(self, x):
        x = self.convTrans1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.convTrans2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x) if not self.last else x
        return x


# Autoencoder
class autoencoder_minist(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = encodeBlock(1, 20)  # 64x64 > 16x16
        # self.down2 = encodeBlock(20,40) # 16x16 > 4x4
        # self.down3 = encodeBlock(40,20) # 4x4 > 1x1

        self.fcn1 = nn.Linear(49, LSSIZE)
        self.fcn2 = nn.Linear(LSSIZE, 49)
        self.relu = nn.ReLU()
        self.sft = nn.Softmax()
        # self.up1 = decodeBlock(20,40)
        # self.up2 = decodeBlock(40,20)
        self.up3 = decodeBlock(20, 1, last=True)

    def forward(self, x):
        row = x
        x = self.down1(x)
        # x = self.down2(x)
        # x = self.down3(x)

        x = x.view(-1, 7*7)
        ls = self.fcn1(x)
        ls = self.sft(ls)

        x = self.fcn2(ls)

        x = x.view(-1, 20, 7, 7)

        # x = self.up1(x)
        # x = self.up2(x)
        x = self.up3(x)
        return ls, x



# Autoencoder
class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = encodeBlock(3, 20)  # 64x64 > 16x16
        # self.down2 = encodeBlock(20,40) # 16x16 > 4x4
        # self.down3 = encodeBlock(40,20) # 4x4 > 1x1

        self.fcn1 = nn.Linear(256, LSSIZE)
        self.fcn2 = nn.Linear(LSSIZE, 256)
        self.relu = nn.ReLU()
        self.sft = nn.Softmax()
        # self.up1 = decodeBlock(20,40)
        self.up2 = decodeBlock(40,20)
        self.up3 = decodeBlock(20, 3, last=True)

    def forward(self, x):
        row = x
        x = self.down1(x)
        # x = self.down2(x)
        # x = self.down3(x)

        x = x.view(-1, 256)
        ls = self.fcn1(x)
        ls = self.sft(ls)

        x = self.fcn2(ls)

        x = x.view(-1, 20, 16, 16)

        # x = self.up1(x)
        # x = self.up2(x)
        x = self.up3(x)
        return ls, x


class autoencoder_linear(nn.Module):
    def __init__(self):
        super(autoencoder_linear, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(3 * 64 * 64, 4096),
                                     nn.ReLU(True),
                                     nn.Linear(4096, 1024),
                                     nn.ReLU(True),
                                     nn.Linear(1024, 256),
                                     nn.ReLU(True),
                                     nn.Linear(256, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     # nn.ReLU(True),
                                     # nn.Linear(64, 12)
                                     # nn.ReLU(True),
                                     # nn.Linear(12, 3)
                                     )
        self.decoder = nn.Sequential(
            # nn.Linear(3, 12),
            # nn.ReLU(True),
            # nn.Linear(12, 64),
            # nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 3 * 64 * 64),
            nn.Tanh())

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode



class autoencoder_minist1(nn.Module):
    def __init__(self):
        super(autoencoder_minist1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


class autoencoder_t1(nn.Module):
    def __init__(self):
        super(autoencoder_t1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode
