import torch
from unet_parts import *

class CCANet(torch.nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(CCANet, self).__init__()
        #  u net

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64 // factor, bilinear)
        self.outc = OutConv(32, n_classes)

    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # print(x.shape)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':

    model = CCANet(n_channels=3, n_classes=2)
    d_input = torch.randn((1, 3, 128, 128))
    pred = model(d_input)
    print(pred.shape)
