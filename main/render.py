import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import functools
import vg
import cv2
import numpy as np
import math
import scipy.stats as st
import kornia


EPS = 1e-7


# class GaussianBlur(nn.Module):
class GaussianBlur():
    def __init__(self, gauss_kernel_size, gauss_sigma, device):
        super(GaussianBlur, self).__init__()
        self.gauss_kernel_padding = int((gauss_kernel_size-1)/2)
        # kernel = self.get_kernel(gauss_kernel_size, gauss_sigma) #获得高斯卷积核
        # print('kernel',kernel)
        # kernel = self.creat_gauss_kernel(gauss_kernel_size, gauss_sigma) #获得高斯卷积核
        # print('kernel',kernel)
        # kernel = self.gaussian_2d_kernel(gauss_kernel_size, gauss_sigma) #获得高斯卷积核
        # print('kernel',kernel)
        kernel = self.gaussian_kernel_2d_opencv(gauss_kernel_size, gauss_sigma)  # 获得高斯卷积核
        # print('kernel',kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(device)  # 扩展两个维度
        # self.weight = nn.Parameter(data=kernel, requires_grad=False)  # 原始权重，发生变化
        self.weight = nn.Parameter(data=kernel, requires_grad=True)  # 原始权重，不发生变化
        # print('self.weight.shape',self.weight.shape)

    def get_kernel(self, kernlen=3, nsig=1):     # nsig 标准差 ，kernlen核尺寸
        interval = (2*nsig+1.)/kernlen  # 计算间隔
        x = np.linspace(-nsig-interval/2., nsig+interval / 2., kernlen+1)
        # 在前两者之间均匀产生数据

        # 高斯函数其实就是正态分布的密度函数
        kern1d = np.diff(st.norm.cdf(x))  # 先积分在求导是为啥？得到一个维度上的高斯函数值
        '''st.norm.cdf(x):计算正态分布累计分布函数指定点的函数值
            累计分布函数：概率分布函数的积分'''
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        # np.outer计算外积，再开平方，从1维高斯参数到2维高斯参数
        kernel = kernel_raw/kernel_raw.sum()  # 确保均值为1
        return kernel

    def creat_gauss_kernel(self, kernel_size=3, sigma=1, k=1):
        '''高斯核生成函数'''
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        X = np.linspace(-k, k, kernel_size)
        Y = np.linspace(-k, k, kernel_size)
        x, y = np.meshgrid(X, Y)
        x0 = 0
        y0 = 0
        gauss = 1/(2*np.pi*sigma**2) * \
            np.exp(- ((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
        return gauss

    def gaussian_2d_kernel(self, kernel_size=3, sigma=1):
        '''kernel_size set (n,n) default'''
        kernel = np.zeros([kernel_size, kernel_size])
        center = kernel_size//2

        if sigma == 0:
            sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8

        s = 2*(sigma**2)
        sum_val = 0
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                x = i-center
                y = j-center
                kernel[i, j] = np.exp(-(x**2+y**2) / s)
                sum_val += kernel[i, j]
                # /(np.pi * s)
        sum_val = 1/sum_val
        return kernel*sum_val

    def gaussian_kernel_2d_opencv(self, kernel_size=3, sigma=1):
        kx = cv2.getGaussianKernel(kernel_size, sigma)
        ky = cv2.getGaussianKernel(kernel_size, sigma)
        return np.multiply(kx, np.transpose(ky))

    def forward(self, x):
        # bs,i_c,H1,W1---->bs,1,H1,W1
        # print('x.shape',x.shape)
        x = F.conv2d(x, self.weight, stride=1, padding=self.gauss_kernel_padding)
        # print('x.shape',x.shape)
        return x


class Render():
    "Render functions of single scatter and specular reflection"

    # def __init__(self, cfgs, normal, lightinfo, map_ss, map_sp):
    def __init__(self, cfgs):
        self.device = cfgs.get('device', 'cpu')
        self.PI = torch.from_numpy(np.asarray(math.pi)).to(self.device)
        self.E = torch.from_numpy(np.asarray(math.e)).to(self.device)

        self.batch_size = cfgs.get('batch_size', 64)
        self.gauss_kernel_size = cfgs.get('gauss_kernel_size', 13)
        self.sigma_magnification = cfgs.get('sigma_magnification', 10)
        self.r0 = cfgs.get('r0', 0.02549)
        self.k_rho_dt = cfgs.get('k_rho_dt', 0.027)
        self.k_diffuseIntensity = cfgs.get('k_diffuseIntensity', 0.7)
        self.k_specularIntensity = cfgs.get('k_specularIntensity', 180)
        self.k_ss = cfgs.get('k_ss', 0.1)
        self.weight_lambert = cfgs.get('weight_lambert', 0.7)
        self.mix = cfgs.get('mix', 0.5)
        self.lobe_max = cfgs.get('lobe_max', 1.7)
        self.wb_max = cfgs.get('wb_max', 1.5)
        self.lobeParam0 = cfgs.get('lobeParam0', 1.5)
        self.lobeParam1 = cfgs.get('lobeParam1', 1.5)
        self.wbParam0 = cfgs.get('wbParam0', 1.5)
        self.wbParam1 = cfgs.get('wbParam1', 1.5)
        self.wbParam2 = cfgs.get('wbParam2', 1.5)
        
        self.cfgs = cfgs

    def func_F(self, r0, k2, h):
        '''
        r0:[1, None], k2:[b, c], h:[b,c]
        return:[b, None]
        '''
        val = r0 + (1 - r0) * torch.pow(1 - (k2 * h).sum(1), 5)
        # print('val.shape',val.shape)
        return val

    def blinnPhong(self, cos_angle, shininess1):
        '''
        cos_angle:[b,h,w], shininess1:[1,]
        return:[b,h,w]
        '''
        a2 = shininess1 **2
        n = 2 / a2 -2
        val = (self.E+2) / (2*self.PI) * torch.pow(cos_angle,n)
        return val
    
    def blinnPhong_2(self, cos_angle, alpha, intensity):
        '''
        cos_angle:[b,h,w], alpha:[b,], intensity:[b,]
        return:[1,]
        '''
        b, h, w = cos_angle.shape
        alpha = alpha.reshape(b,1,1).repeat(1,h,w)
        intensity = intensity.reshape(b,1,1).repeat(1,h,w)
        val = (1-alpha) * self.blinnPhong(cos_angle, 0.378) + alpha * self.blinnPhong(cos_angle, 0.20)
        val = val * intensity
        return val

    def HG_SScatterPhase(self,cos_angle,g):
        '''
        cos_angle:[b,1],g:[1,]
        return:[b,1]
        '''
        b,c=cos_angle.shape
        # g=(g).reshape(1,-1).repeat(b,1)
        numerator = 1-g**2
        denominator = torch.pow(1+g**2-2*g*cos_angle, 1.5) *self.PI *4
        val = numerator / denominator
        return val
    
    def HG_SScatterTerm(self, n, k1, k2):
        '''
        n:[b,h,w,c], k1:[b,c], k2:[b,c]
        return:[b,h,w]
        '''        
        b, h, w, c = n.shape
        k1 = k1.reshape(b,1,1,c).repeat(1,h,w,1)
        k2 = k2.reshape(b,1,1,c).repeat(1,h,w,1)
        
        term_third=((n*k1).sum(-1)+(n*k2).sum(-1)).clamp(0.,1.)
        term = torch.where(term_third.eq(0.0),torch.full_like(term_third,0.0),torch.div(1, term_third))
        return term
    
    def func_G(self, n, half, k1, k2):
        '''
        n:[b,h,w,c],half:[b,h,w,c],k1:[b,c],k2:[b,c],
        '''
        b, h, w, c = n.shape
        # print('half.shape',half.shape)
        # print('k2.shape',k2.shape)

        # ---------------------vector Start------------------------------------
        k1_re = k1.reshape(b, 1, 1, -1).repeat(1, h, w, 1)
        k2_re = k2.reshape(b, 1, 1, -1).repeat(1, h, w, 1)
        molecule1 = (2 * (n * half).sum(-1) * (n * k2_re).sum(-1))
        molecule2 = (2 * (n * half).sum(-1) * (n * k1_re).sum(-1))
        denominator = (k2_re * half).sum(-1)
        denominator_without_zero = torch.where(denominator.eq(
            0.0), torch.full_like(denominator, 1.), denominator)
        temp1 = torch.div(molecule1, denominator_without_zero)
        temp2 = torch.div(molecule2, denominator_without_zero)
        # https://blog.csdn.net/gyt15663668337/article/details/95882646
        val = torch.where(torch.gt(temp1, temp2), temp1, temp2).clamp(0., 1.)

        matrix = torch.where(denominator.eq(
            0.0), torch.full_like(denominator, 0.0), val)

        return matrix.to(self.device)

    def Norm_vec(self, normal, vector):
        '''
        normal:(b,h,w,c), vector:(b,c)
        return:(b,h,w)
        '''
        b, h, w, c = normal.shape
        vector = vector.reshape(b, 1, 1, -1).repeat(1, h, w, 1)
        val = (normal * vector).sum(-1)
        return val

    def specularIntegralFunction(self, k1, k2, n, r0, alpha, intensity):
        '''
        k1 表示光源，k2表示视点，n表示法线(多维矩阵)，r0表示皮肤特性，alpha, intensity表示bline-phong高光参数
        k1:[b,c], k2:[b,c], n:[b,h,w,c], r0:[b,none], alpha:[b,none], intensity:[b,none]
        return:[b,h,w]
        '''
        b, h, w, c = n.shape
        k1_re = k1.flatten().reshape(b, 1, 1, -1).repeat(1, h, w, 1)  # (b,h,w,c)
        NdotL = (n * k1_re).sum(-1).clamp(0., 1.)  # (b,h,w)

        half = k1 + k2
        # print('h.shape', h.shape)
        half = F.normalize(half, p=2, dim=1)  # (b,3)
        half_re = half.flatten().reshape(b, 1, 1, -1).repeat(1, h, w, 1)  # (b,h,w,c)
        
        Fresnel = (self.func_F(r0, k2, half)).reshape(b, 1, 1).repeat(1, h, w)
        
        NdotH = (n * half_re).sum(-1).clamp(0., 1.)  # (b,h,w)
        Phong = self.blinnPhong_2(NdotH, alpha, intensity)  # (b,h,w)
        Geo = self.func_G(n, half_re, k1, k2)  # (b,h,w)
        sp_melocule = Phong * Fresnel * Geo
        sp_denominator = self.Norm_vec(n, k1) + self.Norm_vec(n, k2) - self.Norm_vec(n, k1) * self.Norm_vec(n, k2)

        rho_sp = torch.where(sp_denominator.eq(0.0), torch.full_like(
            sp_denominator, 0.), torch.div(sp_melocule, sp_denominator))
        tem_sp = torch.where(rho_sp.gt(0.0), rho_sp, torch.full_like(rho_sp, 0.))
        
        val = NdotL * tem_sp
        return val

    def tensor2numpy(self, img):
        '''
        img:gpu,tensor
        return:cpu,array
        '''
        # return img.cpu().numpy()
        return img.detach().cpu().numpy()

    def gaussianBlur_sum1(self, imgs_input, gauss_kernel_size, sigma_magnification):
        '''
        img_input:[b,h,w,c], gauss_kernel_size:[1,none], sigma_magnification:[1,none]
        return:[b,h,w,c]
        '''
        b, h, w, c = imgs_input.shape

        imgs_input = self.tensor2numpy(imgs_input)
        # gauss_kernel_size = self.tensor2numpy(gauss_kernel_size)
        # sigma_magnification = self.tensor2numpy(sigma_magnification)
        results = np.zeros_like(imgs_input)
        for k in range(b):
            img_input = imgs_input[k]

            # 6个高斯拟合皮肤的三层dipole profile
            sigma = np.array([0.0064, 0.0484, 0.1870, 0.5670,
                              1.9900, 7.4100]) * sigma_magnification
            weights_red = np.array([0.233, 0.100, 0.118, 0.113, 0.358, 0.078])
            weights_green = np.array(
                [0.455, 0.336, 0.198, 0.007, 0.004, 0.000])
            weights_blue = np.array([0.649, 0.344, 0.000, 0.007, 0.000, 0.000])
            h, w, c = img_input.shape
            Len = len(sigma)
            img_blur = np.zeros([Len, h, w, c], np.float32)  # [Len,h,w,c]
            for i in range(Len):
                img_blur[i, :, :, 0] = cv2.GaussianBlur(
                    img_input[:, :, 0], (gauss_kernel_size, gauss_kernel_size), sigmaX=sigma[i])
                img_blur[i, :, :, 1] = cv2.GaussianBlur(
                    img_input[:, :, 1], (gauss_kernel_size, gauss_kernel_size), sigmaX=sigma[i])
                img_blur[i, :, :, 2] = cv2.GaussianBlur(
                    img_input[:, :, 2], (gauss_kernel_size, gauss_kernel_size), sigmaX=sigma[i])

            result_r = np.zeros([h, w], np.float32)
            result_g = np.zeros([h, w], np.float32)
            result_b = np.zeros([h, w], np.float32)
            img_blur_r = img_blur[:, :, :, 0]  # [Len,h,w]
            img_blur_g = img_blur[:, :, :, 1]  # [Len,h,w]
            img_blur_b = img_blur[:, :, :, 2]  # [Len,h,w]

            for i in range(Len):
                # result_r = addWeighted(result_r, img_blur_r[i], weights_red[i])
                # result_g = addWeighted(result_g, img_blur_g[i], weights_green[i])
                # result_b = addWeighted(result_b, img_blur_b[i], weights_blue[i])

                result_r = cv2.addWeighted(
                    result_r, 1, img_blur_r[i], weights_red[i], 0)
                result_g = cv2.addWeighted(
                    result_g, 1, img_blur_g[i], weights_green[i], 0)
                result_b = cv2.addWeighted(
                    result_b, 1, img_blur_b[i], weights_blue[i], 0)

            results[k] = cv2.merge([result_r, result_g, result_b])

        results = torch.from_numpy(results).to(self.device)
        return results.clamp(0., 1.)
    
    def gaussianBlur_2d(self, input, kernel_size, sigma):
        gauss = kornia.filters.GaussianBlur2d((kernel_size,kernel_size), (sigma,sigma))
        val = gauss(input)
        return val
    
    def medianBlur_2d(self, input, kernel_size):
        return kornia.median_blur(input, (kernel_size,kernel_size))

    def gaussianBlur_sum2(self, imgs_input, gauss_kernel_size, sigma_magnification):
        '''
        img_input:[b,h,w,c], gauss_kernel_size:[1,none], sigma_magnification:[1,none]
        return:[b,h,w,c]
        '''
        b, h, w, c = imgs_input.shape
        results = torch.zeros_like(imgs_input)
        imgs = imgs_input.permute(0, 3, 1, 2)

        # 6个高斯拟合皮肤的三层dipole profile
        sigma = torch.FloatTensor([0.0064, 0.0484, 0.1870, 0.5670, 1.9900, 7.4100]) * sigma_magnification
        weights_r = torch.FloatTensor([0.233, 0.100, 0.118, 0.113, 0.358, 0.078]).to(self.device)
        weights_g = torch.FloatTensor([0.455, 0.336, 0.198, 0.007, 0.004, 0.000]).to(self.device)
        weights_b = torch.FloatTensor([0.649, 0.344, 0.000, 0.007, 0.000, 0.000]).to(self.device)

        Len = len(sigma)
        img_blur = torch.zeros([Len, b, c, h, w])  # [Len,c,h,w]
        for i in range(Len):
            img_blur[i] = self.gaussianBlur_2d(imgs, gauss_kernel_size, sigma[i]).to(self.device)

        result_r = torch.zeros([b, h, w]).to(self.device)
        result_g = torch.zeros([b, h, w]).to(self.device)
        result_b = torch.zeros([b, h, w]).to(self.device)
        img_blur_r = img_blur[:, :, 0, :, :].to(self.device)  # [Len,b,h,w]
        img_blur_g = img_blur[:, :, 1, :, :].to(self.device)  # [Len,b,h,w]
        img_blur_b = img_blur[:, :, 2, :, :].to(self.device)  # [Len,b,h,w]

        for i in range(Len):
            result_r = result_r + weights_r[i] * img_blur_r[i]
            result_g = result_g + weights_g[i] * img_blur_g[i]
            result_b = result_b + weights_b[i] * img_blur_b[i]

        results[:, :, :, 0] = result_r
        results[:, :, :, 1] = result_g
        results[:, :, :, 2] = result_b
        return results.clamp(0., 1.)
    
    def gaussianBlur_sum(self, imgs_input, gauss_kernel_size, sigma_magnification):
        '''
        img_input:[b,h,w,c], gauss_kernel_size:[1,none], sigma_magnification:[1,none]
        return:[b,h,w,c]
        '''
        b, h, w, c = imgs_input.shape
        # print('imgs_input.shape',imgs_input.shape)
        results = torch.zeros_like(imgs_input)
        imgs_r = imgs_input[:, :, :, 0].unsqueeze(-1).permute(0, 3, 1, 2)
        imgs_g = imgs_input[:, :, :, 1].unsqueeze(-1).permute(0, 3, 1, 2)
        imgs_b = imgs_input[:, :, :, 2].unsqueeze(-1).permute(0, 3, 1, 2)

        # 6个高斯拟合皮肤的三层dipole profile
        sigma = torch.FloatTensor([0.0064, 0.0484, 0.1870, 0.5670, 1.9900, 7.4100]) * sigma_magnification
        weights_r = torch.FloatTensor([0.233, 0.100, 0.118, 0.113, 0.358, 0.078])
        weights_g = torch.FloatTensor([0.455, 0.336, 0.198, 0.007, 0.004, 0.000])
        weights_b = torch.FloatTensor([0.649, 0.344, 0.000, 0.007, 0.000, 0.000])

        Len = len(sigma)
        img_blur = torch.zeros([Len, b, c, h, w])  # [Len,c,h,w]
        for i in range(Len):
            img_blur_r = GaussianBlur(gauss_kernel_size, sigma[i], self.device).forward(imgs_r).to(self.device)
            img_blur_g = GaussianBlur(gauss_kernel_size, sigma[i], self.device).forward(imgs_g).to(self.device)
            img_blur_b = GaussianBlur(gauss_kernel_size, sigma[i], self.device).forward(imgs_b).to(self.device)
            img_blur[i] = torch.cat([img_blur_r,img_blur_g,img_blur_b],dim=1)

        result_r = torch.zeros([b, h, w])
        result_g = torch.zeros([b, h, w])
        result_b = torch.zeros([b, h, w])
        img_blur_r = img_blur[:, :, 0, :, :]  # [Len,b,h,w]
        img_blur_g = img_blur[:, :, 1, :, :]  # [Len,b,h,w]
        img_blur_b = img_blur[:, :, 2, :, :]  # [Len,b,h,w]

        for i in range(Len):
            result_r = result_r + weights_r[i] * img_blur_r[i]
            result_g = result_g + weights_g[i] * img_blur_g[i]
            result_b = result_b + weights_b[i] * img_blur_b[i]

        results[:, :, :, 0] = result_r
        results[:, :, :, 1] = result_g
        results[:, :, :, 2] = result_b
        return results.clamp(0., 1.).to(self.device)

    def shadow(self, lightDir, map_n):
        '''
        "render shadow"
        lightDir:[b,3], map_n:[b,h,w,c]
        return:[b,h,w]
        '''
        b, h, w, c = map_n.shape
        lightDir_bhwc = lightDir.reshape(b, 1, 1, c).repeat(1, h, w, 1)
        shade = (map_n * lightDir_bhwc).sum(-1).clamp(0., 1.)
        return shade
    
    def sscatter(self, shadow, map_ss, map_n, rho_dt_L, rho_dt_V,k1,k2):
        '''
        "render single scatter"
        shadow:[b,h,w], map_ss:[b,h,w], map_n:[b,h,w,c], rho_dt_L:[b,h,w], rho_dt_V:[b,h,w],k1:[b,c],k2:[b,c]
        return:[b,1,h,w]
        '''
        b, h, w, c = map_n.shape
        
        T_dt = (1-rho_dt_L) * (1-rho_dt_V)
        cos_angle = (k1*k2).sum(-1).clamp(0.,1.)
        cos_angle_re = cos_angle.reshape(b,1)
        phase1=self.HG_SScatterPhase(cos_angle_re,0.1)
        phase2=self.HG_SScatterPhase(cos_angle_re,0.8)
        phase = (phase1+phase2).clamp(0.,1.)
        phase_re = phase.reshape(b,1,1).repeat(1,h,w)
        term = (self.HG_SScatterTerm(map_n,k1,k2)).clamp(0.,1.)
        component_ss1 = map_ss * T_dt * phase_re * term
        
        # 使用lambert漫反射模拟后向散射现象
        component_ss2 = map_ss * shadow
        
        component_ss = component_ss1 + component_ss2 * self.weight_lambert
        # component_ss = component_ss * 2. - 1.
        return component_ss.unsqueeze(1)  # (b,1,h,w)

    def specular(self, lightDir, viewDir, lobePara, normal, map_sp):
        '''
        "render specular reflection"
        lightDir:[b,3],viewDir:[b,3],lobePara:[b,4], normal:[b,h,w,c], map_sp:[b,h,w]
        return:[b,1,h,w]
        '''
        # specular coefficients of the double lobes
        alpha = lobePara[:, 0]
        intensity = lobePara[:, 1]

        rho_sp = self.specularIntegralFunction(lightDir, viewDir, normal, self.r0, alpha, intensity)

        component_sp = rho_sp * map_sp * self.k_specularIntensity
        # component_sp = component_sp * 2. - 1.
        return component_sp.unsqueeze(1)  # (b,1,h,w)

    def subsurface(self, lightDir, wbPara, map_n, map_d, map_sp, rho_dt_L, rho_dt_V):
        '''
        "render subsurface scatter"
        lightDir:[b,c], wbPara:[b,c], map_n:[b,h,w,c], map_d:[b,h,w,c], map_sp:[b,h,w], rho_dt_L:[b,h,w], rho_dt_V:[b,h,w]
        return:[b,c,h,w]
        '''
        b, h, w, c = map_n.shape
        wbPara = wbPara.reshape(b,1,1,c).repeat(1,h,w,1)
        rho_dt_L = rho_dt_L * self.k_rho_dt
        rho_dt_V = rho_dt_V * self.k_rho_dt
        map_d_front = torch.pow(map_d, self.mix)
        map_d_back = torch.pow(map_d, 1 - self.mix)

        # [b,h,w]
        lightDir_re = lightDir.flatten().reshape(b, 1, 1, -1).repeat(1, h, w, 1)
        ndotL = (map_n * lightDir_re).sum(-1).clamp(0., 1.) *self.k_diffuseIntensity
        irradiance = ndotL * (1 - map_sp * rho_dt_L)
        irradiance_front = (irradiance.unsqueeze(-1).repeat(1, 1, 1, c) * map_d_front)
        # irradiance_mix = self.gaussianBlur_sum(irradiance_front, self.gauss_kernel_size, self.sigma_magnification)
        # irradiance_mix = self.gaussianBlur_sum1(irradiance_front, self.gauss_kernel_size, self.sigma_magnification)
        irradiance_mix = self.gaussianBlur_sum2(irradiance_front, self.gauss_kernel_size, self.sigma_magnification)
        component_sub_back = irradiance_mix * map_d_back

        component_sub = (1 - map_sp * rho_dt_V).unsqueeze(-1).repeat(1, 1, 1, c) * component_sub_back * wbPara
        # component_sub = component_sub * 2. -1.
        return component_sub.permute(0, 3, 1, 2)

    def diffuse(self, wbPara, shadow, map_d):
        '''
        "render diffuse"
        wbPara:[b,c],shadow:[b,h,w], map_d:[b,h,w,c]
        return:[b,c,h,w]
        '''
        b, h, w, c = map_d.shape
        wbPara = wbPara.reshape(b,1,1,c).repeat(1,h,w,1)
        shadow = shadow.unsqueeze(-1).repeat(1, 1, 1, c)
        component_d = map_d * shadow * wbPara *self.k_diffuseIntensity
        # component_d = component_d * 2. - 1.
        return component_d.permute(0, 3, 1, 2)

    def ambient(self, map_d):
        '''
        "render ambient"
        map_d:[b,h,w,c]
        return:[b,c,h,w]
        '''
        component_am = map_d
        # component_am = component_am * 2. - 1.
        return component_am.permute(0, 3, 1, 2)

    def ourRender(self, lightinfo, map_n, map_d, map_dtL, map_ss_sp_dtV):
        '''
        "render all component, including subsurface scatter, single scatter, specular"
        lightinfo:[b,10], map_n:[b,c,h,w], map_d:[b,c,h,w], map_dtL:[b,1,h,w], map_ss_sp_dtV:[b,c,h,w]
        return:[b,c,h,w]
        '''
        lightinfo = lightinfo.to(self.device)
        map_n = map_n.to(self.device)
        map_d = map_d.to(self.device) * 0.5 + 0.5
        map_dtL = map_dtL.to(self.device) * 0.5 + 0.5
        map_ss_sp_dtV = map_ss_sp_dtV.to(self.device) * 0.5 + 0.5
        b,c,h,w = map_n.shape
        
        lightDir = F.normalize(lightinfo[:, :3], p=2, dim=1)
        viewDir = F.normalize(lightinfo[:, 3:6], p=2, dim=1)
        # lobePara = (lightinfo[:, 6:8].to(self.device) * 0.5 + 0.5) * self.lobe_max
        # wbPara = (lightinfo[:, 8:11].to(self.device) * 0.5 + 0.5) * self.wb_max
        
        lobePara = torch.full_like(lightinfo[:, 6:8],0.0).to(self.device)
        wbPara = torch.full_like(lightinfo[:, 8:11],0.0).to(self.device)
        lobePara[:,0] = self.lobeParam0
        lobePara[:,1] = self.lobeParam1
        wbPara[:,0] = self.wbParam0
        wbPara[:,1] = self.wbParam1
        wbPara[:,2] = self.wbParam2

        map_n_bhwc = map_n.permute(0, 2, 3, 1)
        map_d_bhwc = map_d.permute(0, 2, 3, 1)
        map_ss_bhw = map_ss_sp_dtV[:, 0, :, :]
        map_sp_bhw = map_ss_sp_dtV[:, 1, :, :]
        rho_dt_V_bhw = map_ss_sp_dtV[:, 2, :, :] * self.k_rho_dt
        rho_dt_L_bhw = map_dtL[:, 0, :, :] * self.k_rho_dt

        shadow = self.shadow(lightDir, map_n_bhwc)
        component_ss = self.sscatter(shadow, map_ss_bhw,map_n_bhwc,rho_dt_L_bhw,rho_dt_V_bhw,lightDir,viewDir) *self.k_ss
        component_sp = self.specular(lightDir, viewDir, lobePara, map_n_bhwc, map_sp_bhw)
        
        # sp_tem = component_sp.reshape(b,-1).mean(1)
        # ss_tem = component_ss.reshape(b,-1).mean(1)
        # # k_temp = torch.div(1, sp_tem)*ss_tem
        # k_temp = torch.div(1, sp_tem)*np.where(ss_tem>sp_tem,ss_tem,sp_tem)
        # k_temp_re = k_temp.reshape(b,1,1,1).repeat(1,1,h,w)
        # component_spss = (component_sp * k_temp_re)
        component_spss = component_sp
        
        component_sub = self.subsurface(lightDir, wbPara, map_n_bhwc, map_d_bhwc, map_sp_bhw, rho_dt_L_bhw, rho_dt_V_bhw)
        # component_am = self.ambient(map_d_bhwc)
        
        # component_spss = component_spss *2.-1.
        # component_ss = component_ss *2.-1.
        # component_sub=component_sub *2.-1.
        # component_am = component_am *2.-1.
        # return component_spss, component_ss, component_sub, component_am
        return component_spss, component_ss, component_sub

    def otherRender(self, lightinfo, map_n, map_d, map_ss_sp_dtV):
        '''
        "render all component, including subsurface scatter, single scatter, specular"
        lightinfo:[b,10], map_n:[b,c,h,w], map_d:[b,c,h,w], map_ss_sp_dtV:[b,c,h,w]
        return:[b,c,h,w]
        '''
        lightinfo = lightinfo.to(self.device)
        map_n = map_n.to(self.device)
        map_d = map_d.to(self.device) * 0.5 + 0.5
        map_ss_sp_dtV = map_ss_sp_dtV.to(self.device) * 0.5 + 0.5
        
        lightDir = F.normalize(lightinfo[:, :3], p=2, dim=1)
        viewDir = F.normalize(lightinfo[:, 3:6], p=2, dim=1)
        # lobePara = (lightinfo[:, 6:8].to(self.device) * 0.5 + 0.5) * self.lobe_max
        # wbPara = (lightinfo[:, 8:11].to(self.device) * 0.5 + 0.5) * self.wb_max
        
        lobePara = torch.full_like(lightinfo[:, 6:8],0.0).to(self.device)
        wbPara = torch.full_like(lightinfo[:, 8:11],0.0).to(self.device)
        lobePara[:,0] = self.lobeParam0
        lobePara[:,1] = self.lobeParam1
        wbPara[:,0] = self.wbParam0
        wbPara[:,1] = self.wbParam1
        wbPara[:,2] = self.wbParam2

        map_n_bhwc = map_n.permute(0, 2, 3, 1)
        map_d_bhwc = map_d.permute(0, 2, 3, 1)
        map_sp_bhw = map_ss_sp_dtV[:, 1, :, :]

        shadow = self.shadow(lightDir, map_n_bhwc)
        component_sp = self.specular(lightDir, viewDir, lobePara, map_n_bhwc, map_sp_bhw)
        component_d = self.diffuse(wbPara, shadow, map_d_bhwc)
        # component_am = self.ambient(map_d_bhwc)

        # component_sp = component_sp *2.-1.
        # component_d = component_d *2.-1.
        # component_am = component_am *2.-1.
        # return component_sp, component_d, component_am
        return component_sp, component_d

    def allRender(self, lightinfo, map_n, map_d, map_dtL, map_ss_sp_dtV):
        '''
        "render all component, including subsurface scatter, single scatter, specular"
        lightinfo:[b,10], map_n:[b,c,h,w], map_d:[b,c,h,w], map_dtL:[b,1,h,w], map_ss_sp_dtV:[b,c,h,w]
        return:[b,c,h,w]
        '''
        lightinfo = lightinfo.to(self.device)
        map_n = map_n.to(self.device)
        map_d = map_d.to(self.device) * 0.5 + 0.5
        map_dtL = map_dtL.to(self.device) * 0.5 + 0.5
        map_ss_sp_dtV = map_ss_sp_dtV.to(self.device) * 0.5 + 0.5

        lightDir = F.normalize(lightinfo[:, :3], p=2, dim=1)
        viewDir = F.normalize(lightinfo[:, 3:6], p=2, dim=1)
        # lobePara = (lightinfo[:, 6:8].to(self.device) * 0.5 + 0.5) * self.lobe_max
        # wbPara = (lightinfo[:, 8:11].to(self.device) * 0.5 + 0.5) * self.wb_max
        
        lobePara = torch.full_like(lightinfo[:, 6:8],0.0).to(self.device)
        wbPara = torch.full_like(lightinfo[:, 8:11],0.0).to(self.device)
        lobePara[:,0] = self.lobeParam0
        lobePara[:,1] = self.lobeParam1
        wbPara[:,0] = self.wbParam0
        wbPara[:,1] = self.wbParam1
        wbPara[:,2] = self.wbParam2
        
        b,c,h,w = map_n.shape
        
        map_n_bhwc = map_n.permute(0, 2, 3, 1)
        map_d_bhwc = map_d.permute(0, 2, 3, 1)
        
        map_ss_bhw = map_ss_sp_dtV[:, 0, :, :]
        map_sp_bhw = map_ss_sp_dtV[:, 1, :, :]
        rho_dt_V_bhw = map_ss_sp_dtV[:, 2, :, :] * self.k_rho_dt
        rho_dt_L_bhw = map_dtL[:, 0, :, :] * self.k_rho_dt
        

        shadow = self.shadow(lightDir, map_n_bhwc)
        component_ss = self.sscatter(shadow, map_ss_bhw,map_n_bhwc,rho_dt_L_bhw,rho_dt_V_bhw,lightDir,viewDir) *self.k_ss
        component_sp = self.specular(lightDir, viewDir, lobePara, map_n_bhwc, map_sp_bhw)
        
        # sp_tem = component_sp.reshape(b,-1).mean(1)
        # ss_tem = component_ss.reshape(b,-1).mean(1)
        # # k_temp = torch.div(1, sp_tem)*ss_tem
        # k_temp = torch.div(1, sp_tem)*torch.where(ss_tem.gt(sp_tem),ss_tem,sp_tem)
        # k_temp_re = k_temp.reshape(b,1,1,1).repeat(1,1,h,w)
        # component_spss = (component_sp * k_temp_re) * 0.8 
        component_spss = component_sp
        
        component_sub = self.subsurface(lightDir, wbPara, map_n_bhwc, map_d_bhwc, map_sp_bhw, rho_dt_L_bhw, rho_dt_V_bhw)
        component_d = self.diffuse(wbPara, shadow, map_d_bhwc)
        # component_am = self.ambient(map_d_bhwc)
        
        # component_spss = component_spss.repeat(1,c,1,1)
        # component_ss = component_ss.repeat(1,c,1,1)
        # component_sp = component_sp.repeat(1,c,1,1)
        
        # component_spss = component_spss *2.-1.
        # component_ss = component_ss *2.-1.
        # component_sub=component_sub *2.-1.
        # component_sp = component_sp *2.-1.
        # component_d = component_d *2.-1.
        # component_am = component_am *2.-1.
        # return component_spss, component_ss, component_sub, component_sp,component_d, component_am
        return component_spss, component_ss, component_sub, component_sp,component_d


class Render_s():
    # def __init__(self, cfgs, normal, lightinfo, map_ss, map_sp):
    def __init__(self, cfgs):
        self.device = cfgs.get('device', 'cpu')
        self.PI = torch.from_numpy(np.asarray(math.pi)).to(self.device)
        self.E = torch.from_numpy(np.asarray(math.e)).to(self.device)

        self.batch_size = cfgs.get('batch_size', 64)
        self.gauss_kernel_size = cfgs.get('gauss_kernel_size', 13)
        self.sigma_magnification = cfgs.get('sigma_magnification', 10)
        self.r0 = cfgs.get('r0', 0.02549)
        self.k_rho_dt = cfgs.get('k_rho_dt', 0.027)
        self.k_diffuseIntensity = cfgs.get('k_diffuseIntensity', 0.7)
        self.k_specularIntensity = cfgs.get('k_specularIntensity', 180)
        self.k_ss = cfgs.get('k_ss', 0.1)
        self.weight_lambert = cfgs.get('weight_lambert', 0.7)
        self.mix = cfgs.get('mix', 0.5)
        self.lobe_max = cfgs.get('lobe_max', 1.7)
        self.wb_max = cfgs.get('wb_max', 1.5)
        self.cfgs = cfgs

    def func_F_s(self, r0, k2, h):
        '''
        r0:[1], k2:[c], h:[c]
        return:[1]
        '''
        val = r0 + (1 - r0) * torch.pow(1 - torch.dot(k2, h), 5)
        # print('val',val)
        return val

    def blinnPhong_s(self, cos_angle, shininess1):
        '''
        cos_angle:[h,w], shininess1:[1,]
        return:[h,w]
        '''
        a2 = shininess1 **2
        n = 2 / a2 -2
        val = (self.E+2) / (2*self.PI) * torch.pow(cos_angle,n)
        return val
    
    def blinnPhong_2_s(self, cos_angle, alpha, intensity):
        '''
        cos_angle:[h,w], alpha:[1,], intensity:[1,]
        return:[h,w]
        '''
        val = (1-alpha) * self.blinnPhong_s(cos_angle, 0.378) + alpha * self.blinnPhong_s(cos_angle, 0.20)
        val = val * intensity
        return val
    
    def HG_SScatterPhase_s(self,cos_angle,g):
        '''
        cos_angle:[1,],g:[1,]
        return:[1,]
        '''
        numerator = 1-g**2
        denominator = torch.pow(1+g**2-2*g*cos_angle, 1.5) *self.PI *4
        val = numerator / denominator
        return val
    
    def HG_SScatterTerm_s(self, n, k1, k2):
        '''
        n:[h,w,c], k1:[c], k2:[c]
        return:[h,w]
        '''        
        term_third=(torch.matmul(n,k1)+torch.matmul(n,k2)).clamp(0.,1.)
        if term_third.eq(0.0):
            term = 0.0
        else:
            term = torch.div(1, term_third)
        return term

    def func_G_s(self, n, half, k1, k2):
        '''
        n:[h,w,c],half:[c],k1:[c],k2:[c,],
        '''

        molecule1 = (2 * torch.matmul(n, half) * torch.matmul(n, k2))
        molecule2 = (2 * torch.matmul(n, half) * torch.matmul(n, k1))
        denominator = torch.dot(k2, half)
        # print(denominator)
        if denominator.eq(0.0):
            matrix = torch.full_like(denominator, 1.)
        else:
            temp1 = torch.div(molecule1, denominator)
            temp2 = torch.div(molecule2, denominator)
            # https://blog.csdn.net/gyt15663668337/article/details/95882646
            matrix = torch.where(torch.gt(temp1, temp2),
                                 temp1, temp2).clamp(0., 1.)
        return matrix

    def Norm_vec_s(self, normal, vector):
        '''
        normal: (h,w,c)
        vector:(c)
        return:(h,w)
        '''
        val = torch.matmul(normal, vector)
        return val

    def specularIntegralFunction_s(self, k1, k2, n, r0, alpha, intensity):
        '''
        k1 表示光源，k2表示视点，n表示法线(单点)，r0表示皮肤特性，s，ks表示bline-phong高光参数
        k1:[c], k2:[c], n:[h,w,c], r0:[1], s1:[1], ks1:[1], s2:[1], ks2:[1]
        return:[h,w]
        '''
        h, w, c = n.shape
        NdotL = torch.matmul(n, k1).clamp(0., 1.)  # (h,w)

        half = k1 + k2
        # print('h.shape', h.shape)
        half = F.normalize(half, p=2, dim=0)  # (3)

        Fresnel = self.func_F_s(r0, k2, half)
        # print('Fresnel', Fresnel)

        NdotH = torch.matmul(n, half).clamp(0., 1.)  # (h,w)
        Phong = self.blinnPhong_2_s(NdotH, alpha, intensity)  # (h,w)
        Geo = self.func_G_s(n, half, k1, k2)  # (h,w)
        sp_melocule = Phong * Fresnel * Geo  # (h,w)
        sp_denominator = self.Norm_vec_s(
            n, k1) + self.Norm_vec_s(n, k2) - self.Norm_vec_s(n, k1) * self.Norm_vec_s(n, k2)

        rho_sp = torch.where(sp_denominator.eq(0.0), torch.full_like(
            sp_denominator, 0.), torch.div(sp_melocule, sp_denominator))
        tem_sp = torch.where(rho_sp.gt(0.0), rho_sp, torch.full_like(rho_sp, 0.))
        val = NdotL * tem_sp
        return val

    def gaussianBlur_sum_s(self, imgs_input, gauss_kernel_size, sigma_magnification):
        '''
        img_input:[h,w,c], gauss_kernel_size:[1], sigma_magnification:[1]
        return:[h,w,c]
        '''
        h, w, c = imgs_input.shape
        img_input = self.tensor2numpy(imgs_input)

        # 6个高斯拟合皮肤的三层dipole profile
        sigma = np.array([0.0064, 0.0484, 0.1870, 0.5670,
                          1.9900, 7.4100]) * sigma_magnification
        weights_red = np.array([0.233, 0.100, 0.118, 0.113, 0.358, 0.078])
        weights_green = np.array(
            [0.455, 0.336, 0.198, 0.007, 0.004, 0.000])
        weights_blue = np.array([0.649, 0.344, 0.000, 0.007, 0.000, 0.000])
        h, w, c = img_input.shape
        Len = len(sigma)
        img_blur = np.zeros([Len, h, w, c], np.float32)  # [Len,h,w,c]
        for i in range(Len):
            img_blur[i, :, :, 0] = cv2.GaussianBlur(
                img_input[:, :, 0], (gauss_kernel_size, gauss_kernel_size), sigmaX=sigma[i])
            img_blur[i, :, :, 1] = cv2.GaussianBlur(
                img_input[:, :, 1], (gauss_kernel_size, gauss_kernel_size), sigmaX=sigma[i])
            img_blur[i, :, :, 2] = cv2.GaussianBlur(
                img_input[:, :, 2], (gauss_kernel_size, gauss_kernel_size), sigmaX=sigma[i])

        result_r = np.zeros([h, w], np.float32)
        result_g = np.zeros([h, w], np.float32)
        result_b = np.zeros([h, w], np.float32)
        img_blur_r = img_blur[:, :, :, 0]  # [Len,h,w]
        img_blur_g = img_blur[:, :, :, 1]  # [Len,h,w]
        img_blur_b = img_blur[:, :, :, 2]  # [Len,h,w]

        for i in range(Len):
            # result_r = addWeighted(result_r, img_blur_r[i], weights_red[i])
            # result_g = addWeighted(result_g, img_blur_g[i], weights_green[i])
            # result_b = addWeighted(result_b, img_blur_b[i], weights_blue[i])

            result_r = cv2.addWeighted(result_r, 1, img_blur_r[i], weights_red[i], 0)
            result_g = cv2.addWeighted(result_g, 1, img_blur_g[i], weights_green[i], 0)
            result_b = cv2.addWeighted(result_b, 1, img_blur_b[i], weights_blue[i], 0)

        result = cv2.merge([result_r, result_g, result_b])

        result = torch.from_numpy(result).to(self.device)
        return result

    def shadow_s(self, lightDir, map_n):
        '''
        "render shadow"
        lightDir:[3,], map_n:[h,w,c]
        return:[h,w]
        '''
        shade = torch.matmul(map_n, lightDir).clamp(0., 1.)

        return shade *self.k_diffuseIntensity

    def sscatter_s(self, shadow, map_ss, map_n, rho_dt_L, rho_dt_V,k1,k2):
        '''
        "render single scatter"
        shadow:[h,w], map_ss:[h,w],map_n:[h,w,c],rho_dt_L:[h,w], rho_dt_V:[h,w],k1:[c,],k2:[c,]
        return:[1,h,w]
        '''
        T_dt = (1-rho_dt_L) * (1-rho_dt_V)
        cos_angle = torch.matmul(k1,k2).clamp(0.,1.)
        phase1=self.HG_SScatterPhase_s(cos_angle,0.1)
        phase2=self.HG_SScatterPhase_s(cos_angle,0.8)
        phase = (phase1+phase2).clamp(0.,1.)
        term = (self.HG_SScatterTerm_s(map_n,k1,k2)).clamp(0.,1.)
        component_ss1 = map_ss * T_dt * phase * term
        
        # 使用lambert漫反射模拟后向散射现象
        component_ss2 = map_ss * shadow
        
        component_ss = component_ss1 + component_ss2 * self.weight_lambert
        # component_ss = component_ss * 2. - 1.
        return component_ss.unsqueeze(0)
    
    def specular_s(self, lightDir, viewDir, lobePara, map_n, map_sp):
        '''
        "render specular reflection"
        lightDir:[3,],viewDir:[3,],lobePara:[4,], map_n:[h,w,c], map_sp:[h,w]
        return:[1,h,w]
        '''
        # specular coefficients of the double lobes
        alpha = self.lobe_max * lobePara[0]
        intensity = self.lobe_max * lobePara[1]

        rho_sp = self.specularIntegralFunction_s(
            lightDir, viewDir, map_n, self.r0, alpha, intensity)

        component_sp = rho_sp * map_sp * self.k_specularIntensity
        # component_sp = component_sp * 2. - 1.
        return component_sp.unsqueeze(0)

    def subsurface_s(self, lightDir, wbPara, map_n, map_d, map_sp, rho_dt_L, rho_dt_V):
        '''
        "render subsurface scatter"
        lightDir:[c], wbPara:[c], map_n:[h,w,c], map_d:[h,w,c], map_sp:[h,w], rho_dt_L:[h,w], rho_dt_V:[h,w]
        return:[c,h,w]
        '''
        h, w, c = map_n.shape
        wbPara = wbPara * self.wb_max
        wbPara = wbPara.reshape(1,1,c).repeat(h,w,1)
        rho_dt_L = rho_dt_L * self.k_rho_dt
        rho_dt_V = rho_dt_V * self.k_rho_dt
        map_d_front = torch.pow(map_d, self.mix)
        map_d_back = torch.pow(map_d, 1 - self.mix)

        # [h,w]
        ndotL = torch.matmul(map_n, lightDir).clamp(0., 1.)
        irradiance = ndotL * (1 - map_sp * rho_dt_L)

        irradiance_front = (irradiance.unsqueeze(-1).repeat(1, 1, c) * map_d_front)
        irradiance_mix = self.gaussianBlur_sum_s(
            irradiance_front, self.gauss_kernel_size, self.sigma_magnification)
        component_sub = irradiance_mix * map_d_back

        component_sub = (1 - map_sp * rho_dt_V).unsqueeze(-1).repeat(1, 1, c) * component_sub * wbPara
        # component_sub = component_sub * 2. -1.
        return component_sub.permute(2, 0, 1)

    def diffuse_s(self, wbPara, shadow, map_d):
        '''
        "render diffuse"
        shadow:[h,w], map_d:[h,w,c]
        return:[c,h,w]
        '''
        h, w, c = map_d.shape
        wbPara = wbPara.reshape(1,1,c).repeat(h,w,1)
        shadow = shadow.unsqueeze(-1).repeat(1, 1, c)
        component_d = map_d * shadow * wbPara
        component_d = component_d * 2. - 1.
        return component_d.permute(2, 0, 1)

    def ambient_s(self, map_d):
        '''
        "render ambient"
        map_d:[h,w,c]
        return:[c,h,w]
        '''
        component_am = map_d
        # component_am = component_am * 2. - 1.
        return component_am.permute(2, 0, 1)

    def ourRender_s(self, lightinfo, normal, map_d, map_sp, map_ss):
        '''
        "render all component, including subsurface scatter, single scatter, specular"
        lightinfo:[11], normal:[c,h,w], map_d:[c,h,w], map_sp:[1,h,w], map_ss:[1,h,w]
        return:[c,h,w]
        '''
        lightinfo = lightinfo.to(self.device)
        normal = normal.to(self.device)
        map_d = map_d.to(self.device) * 0.5 + 0.5
        map_sp = map_sp.to(self.device) * 0.5 + 0.5
        map_ss = map_ss.to(self.device) * 0.5 + 0.5

        lightDir = F.normalize(lightinfo[:3], p=2, dim=0)
        viewDir = F.normalize(lightinfo[3:6], p=2, dim=0)
        lobePara = lightinfo[6:8].to(self.device) * 0.5 + 0.5
        wbPara = lightinfo[8:11].to(self.device) * 0.5 + 0.5

        normal_bhwc = normal.permute(1, 2, 0)
        map_d_bhwc = map_d.permute(1, 2, 0)
        map_sp_bhw = map_sp[0, :, :]
        map_ss_bhw = map_ss[0, :, :]
        rho_dt_L_bhw = map_ss[1, :, :] * self.k_rho_dt
        rho_dt_V_bhw = map_ss[2, :, :] * self.k_rho_dt

        shadow = self.shadow_s(lightDir, normal_bhwc)
        component_ss = self.sscatter_s(shadow, map_ss_bhw,normal_bhwc,rho_dt_L_bhw,rho_dt_V_bhw,lightDir,viewDir)
        component_sp = self.specular_s(lightDir, viewDir, lobePara, normal_bhwc, map_sp_bhw)
        k_temp = torch.div(1, component_sp.mean()*component_ss.mean())
        component_sp = component_sp * k_temp
        
        component_sub = self.subsurface_s(
            lightDir, wbPara, normal_bhwc, map_d_bhwc, map_sp_bhw, rho_dt_L_bhw, rho_dt_V_bhw)
        component_am = self.ambient_s(map_d)

        return component_sp, component_ss, component_sub, component_am

    def otherRender_s(self, lightinfo, normal, map_d, map_sp):
        '''
        "render all component, including subsurface scatter, single scatter, specular"
        lightinfo:[10], normal:[c,h,w], map_d:[c,h,w], map_sp:[c,h,w]
        return:[c,h,w]
        '''
        lightinfo = lightinfo.to(self.device)
        normal = normal.to(self.device)
        map_d = map_d.to(self.device) * 0.5 + 0.5
        map_sp = map_sp.to(self.device) * 0.5 + 0.5

        lightDir = F.normalize(lightinfo[:3], p=2, dim=0)
        viewDir = F.normalize(lightinfo[3:6], p=2, dim=0)
        lobePara = lightinfo[6:8].to(self.device) * 0.5 + 0.5
        wbPara = lightinfo[8:11].to(self.device) * 0.5 + 0.5

        normal_bhwc = normal.permute(1, 2, 0)
        map_d_bhwc = map_d.permute(1, 2, 0)
        map_sp_bhw = map_sp[0, :, :]

        shadow = self.shadow_s(lightDir, normal_bhwc)
        component_sp = self.specular_s(lightDir, viewDir, lobePara, normal_bhwc, map_sp_bhw)
        component_d = self.diffuse_s(wbPara,shadow, map_d_bhwc)
        component_am = self.ambient_s(map_d)

        return component_sp, component_d, component_am

    def allRender_s(self, lightinfo, normal, map_d, map_sp, map_ss):
        '''
        "render all component, including subsurface scatter, single scatter, specular"
        lightinfo:[10], normal:[c,h,w], map_d:[c,h,w], map_sp:[c,h,w], map_ss:[c,h,w]
        return:[c,h,w]
        '''
        lightinfo = lightinfo.to(self.device)
        normal = normal.to(self.device)
        map_d = map_d.to(self.device) * 0.5 + 0.5
        map_sp = map_sp.to(self.device) * 0.5 + 0.5
        map_ss = map_ss.to(self.device) * 0.5 + 0.5

        lightDir = F.normalize(lightinfo[:3], p=2, dim=0)
        viewDir = F.normalize(lightinfo[3:6], p=2, dim=0)
        lobePara = lightinfo[6:8].to(self.device) * 0.5 + 0.5
        wbPara = lightinfo[8:11].to(self.device) * 0.5 + 0.5

        normal_bhwc = normal.permute(1, 2, 0)
        map_d_bhwc = map_d.permute(1, 2, 0)
        map_sp_bhw = map_sp[0, :, :]
        map_ss_bhw = map_ss[0, :, :]
        rho_dt_L_bhw = map_ss[1, :, :] * self.k_rho_dt
        rho_dt_V_bhw = map_ss[2, :, :] * self.k_rho_dt

        shadow = self.shadow_s(lightDir, normal_bhwc)
        component_ss = self.sscatter_s(shadow, map_ss_bhw,normal_bhwc,rho_dt_L_bhw,rho_dt_V_bhw,lightDir,viewDir)
        component_sp = self.specular_s(lightDir, viewDir, lobePara, normal_bhwc, map_sp_bhw)
        k_temp = torch.div(1, component_sp.mean()*component_ss.mean())
        component_sp = component_sp * k_temp
        component_sub = self.subsurface_s(
            lightDir, wbPara, normal_bhwc, map_d_bhwc, map_sp_bhw, rho_dt_L_bhw, rho_dt_V_bhw)
        component_d = self.diffuse_s(wbPara,shadow, map_d_bhwc)
        component_am = self.ambient_s(map_d)
        
        return component_sp, component_ss, component_sub, component_d, component_am
    
