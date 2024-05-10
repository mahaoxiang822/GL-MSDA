import torch.nn as nn
from torch.nn import init

def conv_norm_relu(dim_in, dim_out, kernel_size=3, norm=nn.BatchNorm2d, stride=1, padding=1,
                   use_leakyRelu=False, use_bias=False, is_Sequential=True):
    if use_leakyRelu:
        act = nn.LeakyReLU(0.2, True)
    else:
        act = nn.ReLU(True)

    if is_Sequential:
        result = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=use_bias),
            norm(dim_out, affine=True),
            act
        )
        return result
    return [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=False),
            norm(dim_out, affine=True),
            act]

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

class UpsampleBasicBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, norm=nn.BatchNorm2d, scale=2, mode='bilinear', upsample=True):
        super(UpsampleBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm(planes)

        if upsample:
            if inplanes != planes:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 3, 1

            self.upsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                norm(planes))
        else:
            self.upsample = None

        self.scale = scale
        self.mode = mode

    def forward(self, x):

        if self.upsample is not None:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=True)
            residual = self.upsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        dims = [32, 64, 128, 256, 512, 1024, 2048]

        norm = nn.InstanceNorm2d

        self.up1 = UpsampleBasicBlock(dims[4], dims[3], kernel_size=1, padding=0, norm=norm)
        self.up2 = UpsampleBasicBlock(dims[3], dims[2], kernel_size=1, padding=0, norm=norm)
        self.up3 = UpsampleBasicBlock(dims[2], dims[1], kernel_size=1, padding=0, norm=norm)
        self.up4 = UpsampleBasicBlock(dims[1], dims[1], kernel_size=3, padding=1, norm=norm)

        self.skip_3 = conv_norm_relu(dims[3], dims[3], kernel_size=1, padding=0, norm=norm)
        self.skip_2 = conv_norm_relu(dims[2], dims[2], kernel_size=1, padding=0, norm=norm)
        self.skip_1 = conv_norm_relu(dims[1], dims[1], kernel_size=1, padding=0, norm=norm)

        self.up_image = nn.Sequential(
            nn.Conv2d(64, 3, 7, 1, 3, bias=False),
            nn.Tanh()
        )

        init_weights(self.up1, 'normal')
        init_weights(self.up2, 'normal')
        init_weights(self.up3, 'normal')
        init_weights(self.up4, 'normal')
        init_weights(self.skip_3, 'normal')
        init_weights(self.skip_2, 'normal')
        init_weights(self.skip_1, 'normal')
        init_weights(self.up_image, 'normal')

    def forward(self, input):
        skip1 = self.skip_1(self.encoder.out['1'])
        skip2 = self.skip_2(self.encoder.out['2'])
        skip3 = self.skip_3(self.encoder.out['3'])

        upconv4 = self.up1(input)  # input = self.encoder.out['4']
        upconv3 = self.up2(upconv4 + skip3)
        upconv2 = self.up3(upconv3 + skip2)
        upconv1 = self.up4(upconv2 + skip1)

        generated_images = self.up_image(upconv1)

        return generated_images