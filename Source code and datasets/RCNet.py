import torch
from torch import nn
import torch.nn.functional as F


# ================== ConvBlock ==================
class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel // 4, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel // 4)
        self.conv2 = nn.Conv2d(output_channel // 4, output_channel // 4, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel // 4)
        self.conv3 = nn.Conv2d(output_channel // 4, output_channel, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=stride, bias=False)
        self.bn4 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        residual = self.bn4(self.conv4(residual))
        out = x + residual
        out = self.relu(out)
        return out


# ================== SEAttention ==================
class SEAttention(nn.Module):
    def __init__(self, channel=12, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ================== SEResidualBlock ==================
class SEResidualBlock(nn.Module):
    def __init__(self, channels=12, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEAttention(channel=channels, reduction=reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


# ================== MultiScaleConvModule ==================
class MultiScaleConvModule(nn.Module):
    def __init__(self, in_channels, out_channels_per_branch=32, num_classes=5):
        super(MultiScaleConvModule, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=7, padding=3)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(in_channels, out_channels_per_branch, kernel_size=1)

        total_out_channels = 6 * out_channels_per_branch
        self.output_conv = nn.Conv2d(total_out_channels, in_channels, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(total_out_channels, num_classes)
        self.total_out_channels = total_out_channels

    def forward(self, x, classify=True):
        branch_3x3 = self.conv3x3(x)
        branch_5x5 = self.conv5x5(x)
        branch_7x7 = self.conv7x7(x)
        branch_1x1_1 = self.conv1x1_1(x)
        branch_1x1_2 = self.conv1x1_2(x)
        branch_1x1_3 = self.conv1x1_3(x)

        combined = torch.cat([
            branch_3x3, branch_5x5, branch_7x7,
            branch_1x1_1, branch_1x1_2, branch_1x1_3
        ], dim=1)

        features = self.output_conv(combined)

        if classify:
            pooled = self.global_pool(combined).view(combined.size(0), -1)
            output = self.fc(pooled)
            return output
        else:
            return features


# ================== CSEBlock ==================
class CSEBlock(nn.Module):
    def __init__(self, in_channels=12, reduction=4, out_channels_per_branch=32, num_classes=5, is_last=False):
        super(CSEBlock, self).__init__()
        self.se_res_block = SEResidualBlock(channels=in_channels, reduction=reduction)
        self.multi_scale_module = MultiScaleConvModule(
            in_channels=in_channels,
            out_channels_per_branch=out_channels_per_branch,
            num_classes=num_classes
        )
        self.is_last = is_last

    def forward(self, x):
        x = self.se_res_block(x)
        x = self.multi_scale_module(x, classify=self.is_last)
        return x


# ================== FCSEBlock ==================
class FCSEBlock(nn.Module):
    def __init__(self, in_channels=12, reduction=4, out_channels_per_branch=32, num_classes=5):
        super(FCSEBlock, self).__init__()
        self.block1 = CSEBlock(in_channels, reduction, out_channels_per_branch, num_classes, is_last=False)
        self.block2 = CSEBlock(in_channels, reduction, out_channels_per_branch, num_classes, is_last=False)
        self.block3 = CSEBlock(in_channels, reduction, out_channels_per_branch, num_classes, is_last=False)
        self.block4 = CSEBlock(in_channels, reduction, out_channels_per_branch, num_classes, is_last=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
# ================== CSEA_ViT ==================
class CSEA_ViT(nn.Module):
    def __init__(self, in_channels=12, hidden_channels=12, reduction=4, out_channels_per_branch=32, num_classes=5):
        super(CSEA_ViT, self).__init__()
        self.stem = ConvBlock(in_channels, hidden_channels)
        self.encoder = FCSEBlock(
            in_channels=hidden_channels,
            reduction=reduction,
            out_channels_per_branch=out_channels_per_branch,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        return x

