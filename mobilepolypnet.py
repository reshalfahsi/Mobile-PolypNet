import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileBottleneck(nn.Module):
    def __init__(
        self, in_channels, expand_filters, contract_filters, stride=1, add=True
    ):
        super(MobileBottleneck, self).__init__()
        self.stride = stride
        self.add = add

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, expand_filters, 1, padding=0),
            nn.BatchNorm2d(expand_filters),
            nn.ReLU6(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                expand_filters,
                expand_filters,
                3,
                padding=1,
                stride=stride,
                groups=expand_filters,
            ),
            nn.BatchNorm2d(expand_filters),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expand_filters, contract_filters, 1, padding=0, groups=1),
            nn.BatchNorm2d(contract_filters),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if (self.stride == 1) and (self.add):
            out = out + x
        return out


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        skip_channels,
        expand_filters,
        contract_filters,
    ):
        super(Up, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, hidden_channels, 3, padding=1, stride=2
        )
        self.conv = nn.Sequential(
            MobileBottleneck(
                hidden_channels + skip_channels,
                expand_filters,
                contract_filters,
                add=False,
            ),
            MobileBottleneck(
                contract_filters, expand_filters, contract_filters, stride=1
            ),
        )

    def forward(self, x, y):
        x = self.upsample(x)

        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, y], 1)
        x = self.conv(x)
        return x


class MobilePolypNet(nn.Module):
    def __init__(self):
        super(MobilePolypNet, self).__init__()

        self.down0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 1, padding=0, groups=1),
            nn.BatchNorm2d(32),
        )
        self.down2 = nn.Sequential(
            MobileBottleneck(32, 48, 8, stride=2),
            MobileBottleneck(8, 48, 8, stride=1),
            MobileBottleneck(8, 48, 8, stride=1),
        )

        self.down3 = nn.Sequential(
            MobileBottleneck(8, 96, 16, stride=2),
            MobileBottleneck(16, 96, 16, stride=1),
            MobileBottleneck(16, 96, 16, stride=1),
        )

        self.down4 = nn.Sequential(
            MobileBottleneck(16, 144, 32, stride=2),
            MobileBottleneck(32, 144, 32, stride=1),
            MobileBottleneck(32, 144, 32, stride=1),
        )

        self.down5 = nn.Sequential(
            MobileBottleneck(32, 144, 32, stride=2),
            MobileBottleneck(32, 144, 32, stride=1),
            MobileBottleneck(32, 144, 32, stride=1),
        )

        self.down6 = nn.Sequential(
            MobileBottleneck(32, 144, 32, stride=2),
            MobileBottleneck(32, 144, 32, stride=1),
            MobileBottleneck(32, 144, 32, stride=1),
            MobileBottleneck(32, 144, 32, stride=1),
            MobileBottleneck(32, 144, 32, stride=1),
            MobileBottleneck(32, 144, 32, stride=1),
        )

        self.up0 = Up(32, 16, 32, 144, 32)
        self.up1 = Up(32, 16, 32, 96, 16)
        self.up2 = Up(16, 16, 16, 96, 16)
        self.up3 = Up(16, 8, 8, 48, 8)
        self.up4 = Up(8, 8, 32, 48, 8)

        self.out = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        x0 = self.down0(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x7 = self.up0(x6, x5)
        x8 = self.up1(x7, x4)
        x9 = self.up2(x8, x3)
        x10 = self.up3(x9, x2)
        x11 = self.up4(x10, x1)

        out = self.out(x11)

        return out


if __name__ == "__main__":
    from ptflops import get_model_complexity_info

    model = MobilePolypNet()

    macs, params = get_model_complexity_info(
        model, (3, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
