import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, zeros_
from torch.fft import rfft2, irfft2, fftshift, ifftshift


def WieNer(blur, psf, lamda):
    blur_fft = rfft2(blur)
    psf_fft = rfft2(psf)
    psf_fft = torch.conj(psf_fft) / (torch.abs(psf_fft) ** 2 + lamda + 1e-6)
    img = ifftshift(irfft2(psf_fft * blur_fft), (-2, -1))
    return img


class wienernet(nn.Module):
    def __init__(self, psf, im_shape=(140, 140), device='cpu'):
        super(wienernet, self).__init__()
        self.device = device
        self.psf = psf
        D, C, H, W = psf.shape

        self.lamda = nn.Parameter(torch.tensor(100., dtype=torch.float32, device=device))
        self.w = nn.Parameter(torch.tensor(1., dtype=torch.float32, device=device))

        self.left = (H - im_shape[0]) // 2
        self.top = (W - im_shape[1]) // 2

    def forward(self, blur):            # b, 3, H, W
        out = WieNer(blur.unsqueeze(1), self.w * self.psf, self.lamda)
        out = out[:, :, :, self.left:-self.left, self.top:-self.top]
        B, S, C, H, W = out.shape
        out = out / out.reshape([B, S, -1]).max(-1)[0].reshape([B, S, 1, 1, 1])
        return out


class conv3d_bn(nn.Module):
    def __init__(self, in_ch, out_ch, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0)):
        super(conv3d_bn, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv3d(x)


class trans3d_bn(nn.Module):
    def __init__(self, in_ch, out_ch, k=(1, 1, 1), s=(1, 1, 1), p=(0, 0, 0), op=(0, 0, 0)):
        super(trans3d_bn, self).__init__()
        '''  
        nn.ConvTranspose3d  
        Hout = (Hin - 1) * s -2 * p + k + op
        '''
        self.trans3d = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=op),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.trans3d(x)


class Mixed(nn.Module):
    def __init__(self, in_ch=192, out_ch=(64, 96, 128, 16, 32, 32)):
        super(Mixed, self).__init__()
        self.branch0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[0])

        self.branch1_0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[1])
        self.branch1_1 = conv3d_bn(in_ch=out_ch[1], out_ch=out_ch[2], k=(3, 3, 3), p=(1, 1, 1))

        self.branch2_0 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[3])
        self.branch2_1 = conv3d_bn(in_ch=out_ch[3], out_ch=out_ch[4], k=(3, 3, 3), p=(1, 1, 1))

        self.branch3_0 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.branch3_1 = conv3d_bn(in_ch=in_ch, out_ch=out_ch[5])

        self.output_channels = out_ch[0] + out_ch[2] + out_ch[4] + in_ch  # conv1, conv2, conv3, max1

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1_1(self.branch1_0(x))
        b2 = self.branch2_1(self.branch2_0(x))
        b3 = self.branch3_1(self.branch3_0(x))
        return torch.cat([b0, b1, b2, b3], 1)


class Pre_out(nn.Module):
    def __init__(self, in_ch, out_ch, d_layers):
        super(Pre_out, self).__init__()
        self.d_layers = d_layers
        self.out = nn.Sequential(
            conv3d_bn(in_ch, out_ch, k=(3, 3, 3), p=(1, 1, 1)),
            nn.Conv3d(out_ch, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        )

    def forward(self, x):
        x = self.out(x)
        x = F.softplus(x)
        x = x / torch.sum(x, dim=-3, keepdim=True)
        d_out = torch.sum(x * self.d_layers, dim=2, keepdim=False)
        return d_out


class depthnet_s(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, n_stack=12, psf=None, device='cpu'):
        super(depthnet_s, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_stack = n_stack
        self.device = device
        self.psf = psf
        d_layers = np.array([36, 39, 43, 47, 52, 58, 66, 77, 91, 111, 143, 200])
        self.d_layers = torch.from_numpy(d_layers.reshape((1, 1, n_stack, 1, 1))).float().to(self.device)
        """
        Conv3d_1a_7x7
        """
        self.conv3d_1a = conv3d_bn(n_channels, out_ch=64, k=(7, 7, 7), s=(1, 2, 2), p=(3, 3, 3))   # 70
        """
        MaxPool3d_2a_3X3, Conv3d_2b_1x1, Conv3d_2c_3x3
        """
        self.max3d_2a = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))   # 35
        self.conv3d_2b = conv3d_bn(in_ch=64, out_ch=96, k=(3, 3, 3), p=(1, 1, 1))
        self.conv3d_2c = conv3d_bn(in_ch=96, out_ch=128, k=(3, 3, 3), p=(1, 1, 1))
        """
        MaxPool3d_3a_3X3, Mixed_3b, Mixed_3c
        """
        self.max3d_3a = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))   # 18
        self.conv3d_3b = conv3d_bn(in_ch=128, out_ch=192, k=(3, 3, 3), p=(1, 1, 1))
        self.Mixed_3c = Mixed(in_ch=192, out_ch=(64, 96, 128, 16, 32, 32))  # out_ch = 64+128+32+32 = 256
        """
        Mixed 4a, 4b ,4c, 4d, 4e, 4f
        """
        self.max3d_4a = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))   # 9
        self.Mixed_4b = Mixed(in_ch=256, out_ch=(80, 120, 208, 24, 48, 48))  # out_ch = 384
        self.Mixed_4e = Mixed(in_ch=384, out_ch=(128, 192, 256, 32, 64, 64))  # out_ch = 512
        # ---------
        # Decoder
        # ---------
        """
        Upconv2
        """
        self.up_4h = trans3d_bn(in_ch=512, out_ch=64, k=(3, 4, 4), s=(1, 2, 2), p=(1, 1, 1))
        self.up_3g = conv3d_bn(256, 64, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))
        self.up_3i = conv3d_bn(128, 128, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))
        # self.pred_l4 = Pre_out(128, 32, self.d_layers)
        """
        Upconv3   #size,k=(3, 4, 4)
        """
        self.up_3h = trans3d_bn(in_ch=128, out_ch=32, k=(3, 3, 3), s=(1, 2, 2), p=(1, 1, 1))
        self.up_2g = conv3d_bn(128, 32, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))
        self.up_2i = conv3d_bn(64, 64, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))
        # self.pred_l3 = Pre_out(64, 32, self.d_layers)
        """
        Upconv4
        """
        self.up_2h = trans3d_bn(in_ch=64, out_ch=16, k=(3, 4, 4), s=(1, 2, 2), p=(1, 1, 1))
        self.up_1g = conv3d_bn(64, 16, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))
        self.up_1i = conv3d_bn(32, 32, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1))
        # self.pred_l2 = Pre_out(32, 32, self.d_layers)
        """
        Upconv5  #size,k=(3, 3, 3)
        """
        self.final_up = nn.ConvTranspose3d(32, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1))
        self.out = nn.Conv3d(32, self.n_classes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        # Encoder
        conv1a = self.conv3d_1a(x)
        conv2c = self.conv3d_2c(self.conv3d_2b(self.max3d_2a(conv1a)))
        Mix3c = self.Mixed_3c(self.conv3d_3b(self.max3d_3a(conv2c)))
        Mix4f = self.Mixed_4e(self.Mixed_4b(self.max3d_4a(Mix3c)))
        # Decoder
        Up_3 = self.up_3i(torch.cat([self.up_3g(Mix3c), self.up_4h(Mix4f)], 1))
        # pre4 = self.pred_l4(Up_3)
        Up_2 = self.up_2i(torch.cat([self.up_2g(conv2c), self.up_3h(Up_3)], 1))
        # pre3 = self.pred_l3(Up_2)
        Up_1 = self.up_1i(torch.cat([self.up_1g(conv1a), self.up_2h(Up_2)], 1))
        # pre2 = self.pred_l2(Up_1)
        out = self.out(self.final_up(Up_1))
        # result
        d_attention = F.softplus(out)
        d_attention = d_attention / torch.sum(d_attention, dim=-3, keepdim=True)
        pre1 = torch.sum(d_attention * self.d_layers, dim=2, keepdim=False)
        # rgb
        aif_attention = F.softmax(out, dim=2)
        # mask = (aif_attention == aif_attention.max(dim=2, keepdim=True)[0]).to(dtype=torch.int32)
        rgb_out = torch.sum(aif_attention * x, dim=2)
        # # output
        pre_out = []
        pre_out.append(x)
        pre_out.append(pre1)
        # # pre_out.append(pre2)
        # # pre_out.append(pre3)
        # # pre_out.append(pre4)
        pre_out.append(rgb_out)
        return pre_out


class cnn(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(cnn, self).__init__()
        num = 16

        self.uint1 = nn.Sequential(
            nn.Conv2d(in_ch, num, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num, num, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.BatchNorm2d(num),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num, num, kernel_size=(3, 3), padding=3, dilation=(3, 3)),
            nn.BatchNorm2d(num),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num, num, kernel_size=(3, 3), padding=4, dilation=(4, 4)),
            nn.BatchNorm2d(num),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num, num, kernel_size=(3, 3), padding=3, dilation=(3, 3)),
            nn.BatchNorm2d(num),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num, num, kernel_size=(3, 3), padding=2, dilation=(2, 2)),
            nn.BatchNorm2d(num),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num, num, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num, out_ch, kernel_size=(3, 3), padding=1)
        )

        self.uint2 = nn.Sequential(
            nn.ConvTranspose2d(out_ch, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_ch, kernel_size=(3, 3), padding=(1, 1))
        )

    def forward(self, x):
        if x.size()[0] == 1:
            img = x[0]
            y = self.uint2(self.uint1(img) + img)
            return y
        else:
            img_stack = torch.zeros_like(x)
            for i in range(x.size()[1]):
                img = x[:, i]
                y = self.uint2(self.uint1(img) + img)
                img_stack[:, i] = y
            return img_stack


class depthnet_c(nn.Module):
    def __init__(self, psfs, device):
        super(depthnet_c, self).__init__()
        # self.device = device
        self.step1 = wienernet(psfs, (140, 140), device)

        self.step2 = cnn(3, 3)
        # al2 = torch.load('./results/Wiener_CNN_16.tar', map_location=device)
        # self.step2.load_state_dict(al2['net_state_dict'], strict=False)

        self.step3 = depthnet_s(n_channels=3, n_classes=1, n_stack=12, psf=psfs, device=device)
        # al3 = torch.load('./results/DepthNet_S.tar', map_location=device)
        # self.step3.load_state_dict(al3['net_state_dict'], strict=False)

    def forward(self, x):
        # x = x.to(self.device)
        x = self.step1(x)  # bxDx3xHxW
        x = self.step2(x)  # Dx3xHxW
        if len(x.size()) == 4:
            x = x.unsqueeze(0)
        x = self.step3(x.transpose(1, 2))
        return x



