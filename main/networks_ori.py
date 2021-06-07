import torch
from torch.functional import norm
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import functools
import vg
import numpy as np
import math
import scipy.stats as st

EPS = 1e-7


# --------------------------Beny Start--------------------------------------
class inputDown(nn.Module):

    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc,
                              kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        # norm = functools.partial(
        #     nn.BatchNorm2d, affine=True, track_running_stats=True)
        # norm = functools.partial(
        #     nn.InstanceNorm2d, affine=True, track_running_stats=True)
        norm_gn = functools.partial(nn.GroupNorm, affine=True)
        self.downConv = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_nc, output_nc, kernel_size=4,
                      stride=2, padding=1, bias=False),
            # norm(output_nc)
            norm_gn(16*(output_nc//64), output_nc)
        )

    def forward(self, x):
        return self.downConv(x)


class downBottleNeck(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.downBN = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(input_nc, output_nc, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.ReLU(True)
            # nn.LeakyReLU(0.2,True)
        )

    def forward(self, x):
        return self.downBN(x)


class upBottleNeck(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        # norm = functools.partial(
            # nn.BatchNorm2d, affine=True, track_running_stats=True)
        # norm = functools.partial(
        #     nn.InstanceNorm2d, affine=True, track_running_stats=True)
        norm_gn = functools.partial(nn.GroupNorm, affine=True)
        self.upBN = nn.Sequential(
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4,
                               stride=2, padding=1, bias=False),
            # norm(output_nc)
            norm_gn(16*(output_nc//64), output_nc)
        )

    def forward(self, x1, x2):
        x1 = self.upBN(x1)
        return torch.cat([x2, x1], dim=1)


class up(nn.Module):
    def __init__(self, input_nc, output_nc, dropOut=False):
        super().__init__()
        # norm = functools.partial(
        #     nn.BatchNorm2d, affine=True, track_running_stats=True)
        # norm = functools.partial(
        #     nn.InstanceNorm2d, affine=True, track_running_stats=True)
        norm_gn = functools.partial(nn.GroupNorm, affine=True)
        if dropOut:
            self.upConv = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    input_nc * 2, output_nc, kernel_size=4, stride=2, padding=1, bias=False),
                # norm(output_nc),
                norm_gn(16*(output_nc//64), output_nc),
                nn.Dropout(0.5)
            )
        else:
            self.upConv = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(
                    input_nc * 2, output_nc, kernel_size=4, stride=2, padding=1, bias=False),
                # norm(output_nc)
                norm_gn(16*(output_nc//64), output_nc)
            )

    def forward(self, x1, x2):
        x1 = self.upConv(x1)
        return torch.cat([x2, x1], dim=1)


class outUp(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(input_nc * 2, output_nc,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.out(x)


class outConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.Tanh):
        super(outConv, self).__init__()
        if activation is not None:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1),
                nn.ReLU(True),
                nn.Conv2d(out_channels*4, out_channels, kernel_size=1),
                nn.Tanh()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1),
                nn.ReLU(True),
                nn.Conv2d(out_channels*4, out_channels, kernel_size=1)
            )

    def forward(self, x):
        return self.conv(x)


class inputEmbedding(nn.Module):
    def __init__(self, input_nc, output_nc, middle=2048):
        super().__init__()
        self.outputNc = output_nc
        self.embed = nn.Sequential(
            nn.Linear(input_nc, middle),
            nn.LeakyReLU(0.2, True),
            nn.Linear(middle, output_nc**2)
        )

    def forward(self, x1):
        # print('self.outputNc',self.outputNc)
        x = self.embed(x1)
        return x.reshape(-1, 1, self.outputNc, self.outputNc)
    
    
class inputEmbeddingImage(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        
    def forward(self,x):
        x = x.reshape(-1,self.input_nc).unsqueeze(-1).unsqueeze(-1)
        x = x.repeat(1,1,self.output_nc,self.output_nc)
        return x
        

class UNet_En(nn.Module):
    def __init__(self, img_channels_in, embeddingNum, inputSize):
        super(UNet_En, self).__init__()
        self.img_channels_in = img_channels_in
        self.embeddingNum = embeddingNum
        self.inputSize = inputSize

        # input(input_nc,256,256)
        # self.inputEmbedding = inputEmbedding(self.embeddingNum, self.inputSize)
        # self.down1 = inputDown(self.img_channels_in + 1, 64)  # (64,128,128)
        
        # self.inputEmbedding = inputEmbeddingImage(self.embeddingNum, self.inputSize)
        # self.down1 = inputDown(self.img_channels_in + 3, 64)  # (64,128,128)
        
        self.down1 = inputDown(self.img_channels_in, 64)  # (64,128,128)
        self.down2 = down(64, 128)  # (128,64,64)
        self.down3 = down(128, 256)  # (256,32,32)
        self.down4 = down(256, 512)  # (512,16,16)
        self.down5 = down(512, 512)  # (512,8,8)
        self.down6 = down(512, 512)  # (512,4,4)
        self.down7 = down(512, 512)  # (512,2,2)
        self.down8 = downBottleNeck(512, 512)  # (512,1,1)

    # def forward(self, x, v1, v2):
    #     v1 = self.inputEmbedding(v1)
    #     v2 = self.inputEmbedding(v2)
    #     x = torch.cat([x, v1, v2], dim=1)
        
    # def forward(self, x, v1):
    #     v1 = self.inputEmbedding(v1)
    #     x = torch.cat([x, v1], dim=1)
        
    def forward(self, x):

        dx1 = self.down1(x)
        dx2 = self.down2(dx1)
        dx3 = self.down3(dx2)
        dx4 = self.down4(dx3)
        dx5 = self.down5(dx4)
        dx6 = self.down6(dx5)
        dx7 = self.down7(dx6)
        dx8 = self.down8(dx7)
       
        return dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8


class UNet_De(nn.Module):
    def __init__(self, img_channels_out):
        super(UNet_De, self).__init__()
        self.img_channels_out = img_channels_out

        self.up1 = upBottleNeck(512, 512)  # (512,2,2)
        self.up2 = up(512, 512, dropOut=True)  # (512,4,4)
        self.up3 = up(512, 512, dropOut=True)  # (512,8,8)
        self.up4 = up(512, 512, dropOut=True)  # (512,16,16)
        self.up5 = up(512, 256)  # (256,32,32)
        self.up6 = up(256, 128)  # (128,64,64)
        self.up7 = up(128, 64)  # (64,128,128)
        self.up8 = outUp(64, self.img_channels_out)  # (output_nc,256,256)

    def forward(self, x):
        dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8 = x

        ux1 = self.up1(dx8, dx7)
        ux2 = self.up2(ux1, dx6)
        ux3 = self.up3(ux2, dx5)
        ux4 = self.up4(ux3, dx4)
        ux5 = self.up5(ux4, dx3)
        ux6 = self.up6(ux5, dx2)
        ux7 = self.up7(ux6, dx1)
        out = self.up8(ux7)

        return out


class UNet_DeV(nn.Module):
    def __init__(self, vec_channels_out):
        super(UNet_DeV, self).__init__()
        self.vec_channels_out = vec_channels_out

        self.out_vec = outConv(512, self.vec_channels_out, activation=nn.Tanh)

    def forward(self, x):
        ## -------------------------256------------
        # dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8 = x
        dx8 = x[-1]
        out = self.out_vec(dx8).reshape(dx8.shape[0], -1)

        return out


class netG(nn.Module):
    def __init__(self, input_nc, input_vc, inputSize, output_nc,output_vc=11):
        super(netG, self).__init__()
        self.netS0_00E = UNet_En(input_nc, input_vc, inputSize)
        self.netS1_01D = UNet_De(output_nc)
        self.netS1_02N = UNet_De(output_nc)
        self.netS2_01SS_SP_DTV = UNet_De(output_nc)
        self.netS2_02DTL = UNet_De(1)
        self.netS2_02L = UNet_DeV(output_vc)
        
    # def forward(self, x1, v1, v2):
    #     en_S0 = self.netS0_00E(x1, v1, v2)
    
    # def forward(self, x1, v1):
    #     en_S0 = self.netS0_00E(x1, v1)
    
    def forward(self, x1):
        en_S0 = self.netS0_00E(x1)
        y_d = self.netS1_01D(en_S0)
        y_n = self.netS1_02N(en_S0)
        y_ss_sp_dtV = self.netS2_01SS_SP_DTV(en_S0)
        y_dtL = self.netS2_02DTL(en_S0)
        
        ## without lightinfo
        # return y_d, y_n, y_ss_sp_dtV, y_dtL
        
        # with lightinfo
        y_l = self.netS2_02L(en_S0)
        return y_d, y_n, y_ss_sp_dtV, y_dtL, y_l


class netGs(nn.Module):
    def __init__(self, input_nc, input_vc, inputSize, output_nc,output_vc=11):
        super(netGs, self).__init__()
        self.netS0_00E = UNet_En(input_nc, input_vc, inputSize)
        self.netS1_01D = UNet_De(output_nc*3)
        self.netS2_02DTL = UNet_De(1)
        self.netS2_02L = UNet_DeV(output_vc)
    
    # def forward(self, x1, v1, v2):
    #     en_S0 = self.netS0_00E(x1, v1, v2)
    
    # def forward(self, x1, v1):
    #     en_S0 = self.netS0_00E(x1, v1)
    
    def forward(self, x1):
        en_S0 = self.netS0_00E(x1)
        
        y_all = self.netS1_01D(en_S0)
        y_dtL = self.netS2_02DTL(en_S0)
        y_d = y_all[:,0:3,:,:]
        y_n = y_all[:,3:6,:,:]
        y_ss_sp_dtV = y_all[:,6:9,:,:]

        ## without lightinfo
        # return y_d, y_n, y_ss_sp_dtV, y_dtL
    
        ## with lightinfo
        y_l = self.netS2_02L(en_S0)
        return y_d, y_n, y_ss_sp_dtV, y_dtL, y_l
    
    
class netGr(nn.Module):
    def __init__(self, input_nc, input_vc, inputSize, output_nc):
        super(netGr, self).__init__()
        # input(input_nc,256,256)
        
        self.inputEmbedding = inputEmbedding(input_vc, inputSize)
        self.down1 = inputDown(input_nc + 1, 64)  # (64,128,128)
        
        # self.inputEmbedding = inputEmbeddingImage(input_vc, inputSize)
        # self.down1 = inputDown(input_nc + 11, 64)  # (64,128,128)
        
        self.down2 = down(64, 128)  # (128,64,64)
        self.down3 = down(128, 256)  # (256,32,32)
        self.down4 = down(256, 512)  # (512,16,16)
        self.down5 = down(512, 512)  # (512,8,8)
        self.down6 = down(512, 512)  # (512,4,4)
        self.down7 = down(512, 512)  # (512,2,2)
        self.down8 = downBottleNeck(512, 512)  # (512,1,1)
        
        self.up1 = upBottleNeck(512, 512)  # (512,2,2)
        self.up2 = up(512, 512, dropOut=True)  # (512,4,4)
        self.up3 = up(512, 512, dropOut=True)  # (512,8,8)
        self.up4 = up(512, 512, dropOut=True)  # (512,16,16)
        self.up5 = up(512, 256)  # (256,32,32)
        self.up6 = up(256, 128)  # (128,64,64)
        self.up7 = up(128, 64)  # (64,128,128)
        self.up8 = outUp(64, output_nc)  # (output_nc,256,256)
        
        
    def forward(self, x, v1):
        v1 = self.inputEmbedding(v1)
        x = torch.cat([x, v1], dim=1)

        dx1 = self.down1(x)
        dx2 = self.down2(dx1)
        dx3 = self.down3(dx2)
        dx4 = self.down4(dx3)
        dx5 = self.down5(dx4)
        dx6 = self.down6(dx5)
        dx7 = self.down7(dx6)
        dx8 = self.down8(dx7)
    
        ux1 = self.up1(dx8, dx7)
        ux2 = self.up2(ux1, dx6)
        ux3 = self.up3(ux2, dx5)
        ux4 = self.up4(ux3, dx4)
        ux5 = self.up5(ux4, dx3)
        ux6 = self.up6(ux5, dx2)
        ux7 = self.up7(ux6, dx1)
        out = self.up8(ux7)
        
        return out
    

class EDs(nn.Module):
    def __init__(self, cin, vin, im_size, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDs, self).__init__()
        # self.inputEmbedding = inputEmbedding(vin, im_size)
        num_groups = 2
        ## downsampling
        # https://www.cnblogs.com/wanghui-garcia/p/10877700.html
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 256x256 -> 128x128
            nn.GroupNorm(num_groups, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(num_groups*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(num_groups*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(num_groups*4, nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups*4, nf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        self.network = nn.Sequential(*network)
        
        out_net_vec = [nn.Conv2d(zdim, 11, kernel_size=1, stride=1, padding=0, bias=False),
                       nn.Tanh()]
        self.out_net_vec = nn.Sequential(*out_net_vec)
        
        out_net_maps = [nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(num_groups*4, nf*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups*4, nf*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(num_groups*4, nf*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups*4, nf*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(num_groups*4, nf*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(num_groups*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 128x128
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 128x128 -> 256x256
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout*3, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()]
        
        self.out_net_maps = nn.Sequential(*out_net_maps)
        
        out_net_dt = [nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(num_groups*4, nf*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(num_groups*4, nf*8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(num_groups*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(num_groups*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 128x128
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 128x128 -> 256x256
            nn.Conv2d(nf, 1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()]
        
        self.out_net_dt = nn.Sequential(*out_net_dt)
        

    # def forward(self, x, v1):
    #     v1 = self.inputEmbedding(v1)
    #     x = torch.cat([x, v1], dim=1)
    
    def forward(self, x):
        
        feature_out = self.network(x)
        y_l = self.out_net_vec(feature_out)
        y_all = self.out_net_maps(feature_out)
        y_dtL = self.out_net_dt(feature_out)
        
        y_d = y_all[:,0:3,:,:]
        y_n = y_all[:,3:6,:,:]
        y_ss_sp_dtV = y_all[:,6:9,:,:]
        y_l = y_l.reshape(x.shape[0],-1)
        return y_d, y_n, y_ss_sp_dtV, y_dtL, y_l
    

class EDr(nn.Module):
    def __init__(self, cin, vin, im_size, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDr, self).__init__()
        self.inputEmbedding = inputEmbedding(vin, im_size)
        num_groups = 2
        ## downsampling
        # https://www.cnblogs.com/wanghui-garcia/p/10877700.html
        network = [
            nn.Conv2d(cin+1, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(num_groups, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(num_groups*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(num_groups*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(num_groups*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(num_groups*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(num_groups, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, x, v1):
        v1 = self.inputEmbedding(v1)
        # print('v1.shape',v1.shape)
        x = torch.cat([x, v1], dim=1)
        return self.network(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class netD(nn.Module):
    def __init__(self, input_nc):
        super(netD, self).__init__()

        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True)

        self.net = NLayerDiscriminator(input_nc, norm_layer=norm_layer)

    def forward(self, x):

        y = self.net(x)
        return y


# --------------------------Beny End--------------------------------------

class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(
            pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x / 2 + 0.5
        out = (out - self.mean_rgb.view(1, 3, 1, 1)) / \
            self.std_rgb.view(1, 3, 1, 1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1, im2], 0)
        im = self.normalize(im)  # normalize input

        # compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1 - f2) ** 2
            if conf_sigma is not None:
                loss = loss / (2 * conf_sigma ** 2 + EPS) + \
                    (conf_sigma + EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm // h, wm // w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(
                    sh, sw), stride=(sh, sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss
    
    

if __name__ == '__main__':
    # demo = EDs(3,3)
    demo = EDr(10,11,256,3)
    print('demo',demo)
    x = torch.rand(4,3,256,256)
    v1 = torch.rand(4,11)
    # y_d, y_n, y_ss_sp_dtV, y_dtL, y_l = demo.forward(x)
    
    # print('y',y_d.shape, y_n.shape, y_ss_sp_dtV.shape, y_dtL.shape, y_l.shape)
    
    y = demo.forward(x, v1)
    print('y',y.shape)