import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from utils import Whiten2d, PONO, MS
import torch.nn.functional as F
from utils_curl import NEW_ImageProcessing
from color_change import *
from ssim_loss import SSIM
from attention_module import CBAM
import torchvision


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
    def conv3x3(self,in_channels,out_channels, stride=1):
        return nn.Conv2d(in_channels,out_channels,kernel_size=3,
                         stride=stride,padding=1,bias=True)

class GlobalPoolingBlock(Block,nn.Module):
    def __init__(self,receptive_field):
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        out = self.avg_pool(x)
        return out

class MaxPoolBlock(Block,nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        img_out =self.max_pool(x)
        return img_out

class ConvBlock_curl(nn.Module):
    def __init__(self, in_chans, out_chans, stride=2):
        super(ConvBlock_curl, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=3,
                              stride=stride, padding=1, bias=True)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        return out


class  curl_hsv(nn.Module):
    def __init__(self):
        super(curl_hsv, self).__init__()
        self.hsv1 = ConvBlock_curl(3,64)
        self.hsv2 = MaxPoolBlock()
        self.hsv3 = ConvBlock_curl(64, 64)
        self.hsv4 = MaxPoolBlock()
        self.hsv5 = ConvBlock_curl(64, 64)
        self.hsv6 = MaxPoolBlock()
        self.hsv7 = ConvBlock_curl(64,64)
        self.hsv8 = GlobalPoolingBlock(2)
        self.hsv_fc = torch.nn.Linear(64, 64)
        self.dp1 = nn.Dropout(0.5)

    def forward(self, x):

        x.contiguous()  # remove memory holes
        feat = x[:, 3:64, :, :]
        img = x[:, 0:3, :, :]

        torch.cuda.empty_cache()
        shape = x.shape

        img_rgb = torch.clamp(img, 0, 1)

        img_hsv = NEW_ImageProcessing.new_rgb_to_hsv(img_rgb)
        img_hsv = torch.clamp(img_hsv,0,1)
        feat_hsv = torch.cat((feat,img_hsv),1)

        x= self.hsv1(feat_hsv)
        del feat_hsv

        x = self.hsv2(x)
        x = self.hsv3(x)
        x = self.hsv4(x)
        x = self.hsv5(x)
        x = self.hsv6(x)
        x = self.hsv7(x)
        x = self.hsv8(x)
        x = x.view(x.size()[0],-1)
        x = self.dp1(x)

        H = self.hsv_fc(x)

        img_hsv, gradient_regulariser_hsv = NEW_ImageProcessing.new_adjust_hsv(
            img_hsv,H[:,0:64])
        img_hsv = torch.clamp(img_hsv,0,1)

        img_residual = torch.clamp(NEW_ImageProcessing.new_hsv_to_rgb(
            img_hsv),0,1)
        img = torch.clamp(img + img_residual,0,1)

        return img

class SELayer(torch.nn.Module):
    def __init__(self, num_filter):
        super(SELayer, self).__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv_double = torch.nn.Sequential(
            torch.nn.Conv2d(num_filter, num_filter // 16, 1, 1, 0, bias=True),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(num_filter // 16, num_filter, 1, 1, 0, bias=True),
            torch.nn.Sigmoid())

    def forward(self, x):
        mask = self.global_pool(x)
        mask = self.conv_double(mask)
        x = x * mask
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=0),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DenseConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(DenseConvBlock, self).__init__()
        num_filter = ch_in * 3
        self.conv1 = nn.Conv2d(ch_in, ch_in, kernel_size=3,stride=1 ,padding=1,dilation=1)

        self.conv2_1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv2_2 = nn.Conv2d(ch_in,ch_in,kernel_size=3,stride=1,padding=2,dilation=2)

        self.conv3_1 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1,padding=1,dilation=1)
        self.conv3_2 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=2,dilation=2)
        self.conv3_3 = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=3,dilation=3)

        self.adjust = nn.Conv2d(num_filter,ch_out,kernel_size=1,stride=1,padding=0)
        self.res = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2_2(self.relu(self.conv2_1(x))))
        x3 = self.relu(self.conv3_3(self.relu(self.conv3_2(self.relu(self.conv3_1(x))))))
        out = torch.cat((x1,x2,x3),1)
        adjust = self.adjust(out)
        res = self.res(x)
        return res + adjust

class ResCA(nn.Module):
    def __init__(self, num_filter):
        super(ResCA, self).__init__()
        self.conv1 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1)
        self.cbamlayer = CBAM(num_filter)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        res = self.cbamlayer(x2)
        out = res + x
        return out



class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_in = ConvBlock(ch_in=3, ch_out=32)
        self.conv1 = DenseConvBlock(ch_in=32, ch_out=64)
        self.conv2 = DenseConvBlock(ch_in=64, ch_out=128)
        self.conv3 = DenseConvBlock(ch_in=128, ch_out=256)
        self.conv4 = DenseConvBlock(ch_in=256, ch_out=512)
        self.IW1 = Whiten2d(64)
        self.IW2 = Whiten2d(128)
        self.IW3 = Whiten2d(256)
        self.IW4 = Whiten2d(512)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_in(x)

        x1, x1_mean, x1_std = PONO(x)
        x1 = self.conv1(x)
        x2 = self.pool(x1)

        x2, x2_mean, x2_std = PONO(x2)
        x2 = self.conv2(x2)
        x3 = self.pool(x2)

        x3, x3_mean, x3_std = PONO(x3)
        x3 = self.conv3(x3)
        x4 = self.pool(x3)

        x4, x4_mean, x4_std = PONO(x4)
        x4 = self.conv4(x4)

        x4_iw = self.IW4(x4)
        x3_iw = self.IW3(x3)
        x2_iw = self.IW2(x2)
        x1_iw = self.IW1(x1)

        return x1_iw, x2_iw, x3_iw, x4_iw, x1_mean, x2_mean, x3_mean, x4_mean, x1_std, x2_std, x3_std, x4_std


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()


        self.encoder = Encoder()
        self.UpConv4 = DenseConvBlock(ch_in=512, ch_out=256)
        self.Up3 = Up()
        self.UpConv3 = DenseConvBlock(ch_in=512, ch_out=128)
        self.Up2 = Up()
        self.UpConv2 = DenseConvBlock(ch_in=256, ch_out=64)
        self.Up1 = Up()
        self.UpConv1 = DenseConvBlock(ch_in=128, ch_out=32)

        self.conv_u4 = nn.Conv2d(1, 512, kernel_size=1, padding=0)
        self.conv_s4 = nn.Conv2d(1, 512, kernel_size=1, padding=0)
        self.conv_u3 = nn.Conv2d(1, 128, kernel_size=1, padding=0)
        self.conv_s3 = nn.Conv2d(1, 128, kernel_size=1, padding=0)
        self.conv_u2 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_s2 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.conv_u1 = nn.Conv2d(1, 32, kernel_size=1, padding=0)
        self.conv_s1 = nn.Conv2d(1, 32, kernel_size=1, padding=0)

    def forward(self, Input):
        x1, x2, x3, x4, x1_mean, x2_mean, x3_mean, x4_mean, x1_std, x2_std, x3_std, x4_std = self.encoder.forward(Input)

        # x4->x3
        x4_mean = self.conv_u4(x4_mean)#576
        x4_std = self.conv_s4(x4_std)#576
        x4 = MS(x4, x4_mean, x4_std)#575
        x4 = self.UpConv4(x4)#448
        d3 = self.Up3(x4)
        # x3->x2
        d3 = torch.cat((x3, d3), dim=1)#896
        d3 = self.UpConv3(d3)#320
        x3_mean = self.conv_u3(x3_mean)#320
        x3_std = self.conv_s3(x3_std)#320
        d3 = MS(d3, x3_mean, x3_std)
        d2 = self.Up2(d3)
        # x2->x1
        d2 = torch.cat((x2, d2), dim=1)#640
        d2 = self.UpConv2(d2)#192
        x2_mean = self.conv_u2(x2_mean)#192
        x2_std = self.conv_s2(x2_std)#192
        d2 = MS(d2, x2_mean, x2_std)
        d1 = self.Up1(d2)
        # x1->out
        d1 = torch.cat((x1, d1), dim=1)#384
        d1 = self.UpConv1(d1)#64
        x1_mean = self.conv_u1(x1_mean)#64
        x1_std = self.conv_s1(x1_std)#64
        d1 = MS(d1, x1_mean, x1_std)
        return d1


class scnet(nn.Module):
    def __init__(self, device):
        super(scnet, self).__init__()
        self.device = device
        self.step1 = curl_hsv()
        self.step2 = Decoder()
        self.step3 = ResCA(32)
        self.adjust_conv = nn.Conv2d(32,3,kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        out1 = self.step1(x)
        out2 = self.step2(out1)
        out_rgb = torch.Sigmoid(out2)
        out3 = self.step3(out2)
        out = self.adjust_conv(out)
        hsv_feature = out[:, :3, :, :]
        rgb_feature = out[:, 3:, :, :]
        out = hsv_feature * out1 + rgb_feature * out_rgb
        return out


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.device = torch.device(opt.device)
        self.decoder = scnet(device=self.device).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.VGG16 = PerceptionLoss().to(self.device)
        # self.ssim = SSIM().to(self.device)

    def forward(self, Input):
        return self.decoder.forward(Input)

    def loss(self, outputs, labels):
        reconstruction_loss = self.criterion(outputs, labels)
        vgg16_loss = self.VGG16(outputs, labels)
        # ssim_loss = self.ssim(outputs,labels)

        loss = reconstruction_loss + 0.1 * vgg16_loss
        return loss
