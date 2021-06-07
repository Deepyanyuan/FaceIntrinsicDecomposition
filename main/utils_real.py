import os
import glob
import yaml
import random
import numpy as np
import shutil
import torch
import zipfile
import cv2
import vg
from lmfit import Model  # 最小二乘拟合函数库
import math
import moviepy.editor as mpe
import plotly as plotly
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import tifffile as tiff
import imageio
from PIL import Image,ImageDraw,ImageFont
# from yaml.events import NodeEvent
from scipy import linalg
from tqdm import tqdm


def setup_runtime(args):
    '''Load configs.'''

    # Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    return cfgs


def load_yaml(path):
    '''载入配置文件参数'''
    print(f'Loading configs from {path}')
    with open(path, 'rb') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfgs):
    '''保存配置文件参数'''
    print(f'Saving configs to {path}')
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def xmkdir(path):
    '''Create directory PATH recursively if it does not exist.'''
    os.makedirs(path, exist_ok=True)


# ----------------------------------------Beny Start------------------------------------
def resize(img, size):
    img = np.float32(img)
    img_resize = cv2.resize(img, (size, size))
    return img_resize

def rgb2bgr(img):
    img = np.float32(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def rgb2gray3(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_rgb = np.float32(cv2.merge([gray, gray, gray]))
    return gray_rgb


def normalClipAndMax(imgs, flag=0):
    if len(imgs.shape) == 4:
        num_lights, height, width, channel = imgs.shape
        if flag == 0:
            imgs = np.clip(imgs, 0, 1)
            imgs_max = imgs.reshape(
                num_lights, -1).max(-1).reshape(num_lights, 1, 1, 1)
            imgs_max_num = np.tile(imgs_max, reps=(1, height, width, channel))
            val = np.divide(imgs, imgs_max_num, out=np.zeros_like(
                imgs), where=imgs_max_num != 0)
        elif flag == 1:
            val = np.clip(imgs, 0, 1)
            
        elif flag == 2:
            val = np.divide(imgs, imgs.max(), out=np.zeros_like(
                imgs), where=imgs.max() != 0)
        else:
            val = imgs
    if len(imgs.shape) == 3:
        if flag == 0:
            imgs = np.clip(imgs, 0, 1)
            val = np.divide(imgs, imgs.max(), out=np.zeros_like(
                imgs), where=imgs.max() != 0)
        elif flag == 1:
            val = np.clip(imgs, 0, 1)
        elif flag == 2:
            val = np.divide(imgs, imgs.max(), out=np.zeros_like(
                imgs), where=imgs.max() != 0)
        else:
            val = imgs
    return val


def createSaveFolders(root_path_object, path_views):
    for v in range(len(path_views)):
        path_view = 'Material' + str(path_views[v])
        path_png = root_path_object + '/Show/image_' + path_view
        path_npy = root_path_object + '/Show/npy_' + path_view
        path_video = root_path_object + '/Show/video_' + path_view
        xmkdir(path_npy)
        xmkdir(path_video)
        folders_list = ['img_diffuse','img_residue','img_single','img_mask',
                        'map', 
                        'render_shading', 'render_specular',
                        'render_subscatter','render_diffuse','render_subscatter_t','render_diffuse_t',
                        'render_diffimage','render_diffimage_other',
                        'render_singleimage','render_singleimage_other','render_singleimage_t','render_singleimage_other_t']
        for i in range (len(folders_list)):
            xmkdir(path_png + '/'+ folders_list[i])


def splitSingleDiffuseSpecular(imgCross, imgParallel):
    #单个图像分离不做归一化
    img_d = imgCross
    img_s = np.clip(imgParallel - imgCross, 0, 1)
    img_d_s = imgParallel

    # # 归一化
    # img_d = imgCross/imgParallel.max()
    # img_s = np.clip((imgParallel-imgCross)/imgParallel.max(),0,1)
    # img_d_s = imgParallel
    
    # # 对高光图像做灰度化处理
    # # img_s_gray = img_s[:,:,2]
    # # img_s_gray = (img_s[:,:,0] + img_s[:,:,1] + img_s[:,:,2]) /3
    # img_s_gray = (img_s[:,:,0]*0.30 + img_s[:,:,1]*0.59 + img_s[:,:,2]*0.11)
    # # img_s_temp = np.where(img_s[:,:,0]>img_s[:,:,1],img_s[:,:,0],img_s[:,:,1])
    # # img_s_gray = np.where(img_s_temp>img_s[:,:,2],img_s_temp,img_s[:,:,2])
    # img_s = cv2.merge([img_s_gray,img_s_gray,img_s_gray])
    
    return img_d, img_s, img_d_s


def splitDiffuseSpecularGamma(img_diffuse, img_residue, img_single, gamma):
    # 分离高光和法线
    # 公式参考 搭建-04 光度法线的计算  P10

    ## 没有归一化
    # img_d_s = img_single
    # # img_d_s = (img_single + img_diffuse)/2
    # img_d = img_diffuse
    # img_s = np.clip(img_residue,0,1)

    ## 梯度图像归一化后
    img_d_s = img_single
    # img_d_s = (img_single + img_diffuse)/2
    img_d = img_diffuse / img_d_s.max()
    img_s = np.clip(img_residue / img_d_s.max(), 0, 1)
    
    if gamma != 1:
        ## gamma
        # img_d_s = np.power(img_d_s, 1/gamma)
        # img_d = np.power(img_d, 1/gamma)
        # img_s = np.power(img_s, 1/gamma)
        img_d_s = linearToSrgb(img_d_s)
        img_d = linearToSrgb(img_d)
        img_s = linearToSrgb(img_s)
    
    return img_d, img_s, img_d_s


def threshold_mask(img, threshold, flag=None):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, threshold, 1.0, cv2.THRESH_BINARY)
    if flag is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # binary = np.where(binary>0, binary, gray)
    return binary


def normalImage(img):
    temp = np.linalg.norm(img, ord=2, axis=-1, keepdims=False)
    temp = cv2.merge([temp,temp,temp])
    val = np.divide(img, temp, out=np.zeros_like(img), where=temp!=0)
    return val


def getViewDirImage(img, viewDir):
    img_vd = np.ones_like(img)
    viewDir = viewDir / np.linalg.norm(viewDir, ord=2)
    img_vd[:, :, 0] = viewDir[0]
    img_vd[:, :, 1] = viewDir[1]
    img_vd[:, :, 2] = viewDir[2]
    return img_vd


def func_r0(n1, n2):
    term = (n1 - n2) / (n1 + n2)
    return term * term


def func_F(r0, k2, h):
    return r0 + (1 - r0) * np.power(1 - np.dot(k2, h), 5)


def func_G(n, h, k1, k2):
    if np.dot(k2, h) != 0:
        term1 = (2 * np.dot(n, h) * np.dot(n, k2)) / np.dot(k2, h)
        term2 = (2 * np.dot(n, h) * np.dot(n, k1)) / np.dot(k2, h)

        val = np.where(np.greater(term1, term2), term1, term2)
        return np.clip(val, 0, 1)
    else:
        return 1.0


def func_specular(frontlit, angle, c=1, r0=1):
    return frontlit * (2 * np.cos(angle) - np.cos(angle) * np.cos(angle)) / (r0 * c)
    # return frontlit * (2 * angle - angle * angle) / (r0 * c)


#! by beny refer in https://www.gameres.com/847808.html
def ggx(x, shininess1):
    x = np.cos(x)
    a2 = shininess1 **2
    tmp = (x * a2 - x) * x + 1
    val = a2 / (math.pi * tmp**2)
    return val


def beckmann(x, shininess1):
    x = np.cos(x)
    x_2 = x * x
    a2 = shininess1 **2
    val = np.exp((x_2 - 1)/ (a2 * x_2)) / (math.pi * a2 * x_2 **2)
    return val


def blinnPhong(x, shininess1):
    # 用以拟合的函数 
    cos_x = np.cos(x)
    a2 = shininess1 **2
    n = 2 / a2 -2

    # val = (math.e + 2)/(math.pi * 2) * np.power(cos_x, n) # 标准
    val = (n+2) / (2*math.pi) * np.power(cos_x, n)  # 常用
    val = val / 5
    return val


# def blinnPhong2(x, shininess1, shininess2, alpha):
# def blinnPhong2(x, shininess1, shininess2):
def blinnPhong2(x, alpha, intensity):
# def blinnPhong2(x, alpha):
    # 用以拟合的函数 # 30=0.25, 12=0.378, 48=0.20
    # alpha=0.4
    val = (1-alpha) * blinnPhong(x, 0.378) + alpha * blinnPhong(x, 0.20)
    # val = (1-alpha) * blinnPhong(x, shininess1) + alpha * blinnPhong(x, shininess2)
    return val * intensity


def blinnPhong_1(x, shininess1):
    # 用以拟合的函数 
    a2 = shininess1 **2
    n = 2 / a2 -2

    val = (math.e + 2)/(math.pi * 2) * np.power(x, n) # 标准
    # val = (n+2) / (2*math.pi) * np.power(x, n)  # 常用
    # val = val / 5
    return val


# def blinnPhong_2(x, shininess1, shininess2, alpha):
# def blinnPhong_2(x, shininess1, shininess2):
# def blinnPhong_2(x, alpha):
def blinnPhong_2(x, alpha, intensity):
    # 用以拟合的函数 # 30=0.25, 12=0.378, 48=0.20
    # alpha=0.4
    val = (1-alpha) * blinnPhong_1(x, 0.378) + alpha * blinnPhong_1(x, 0.20)
    # val = (1-alpha) * blinnPhong(x, shininess1) + alpha * blinnPhong(x, shininess2)
    return val * intensity


def HG_SScatterPhase(cos_angle, g):
    numerator = 1 - g ** 2
    denominator = np.power(1+g**2-2*g*cos_angle, 1.5) * math.pi * 4
    val = numerator / denominator
    # val = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    return val


def HG_SScatterTerm(n, k1, k2):
    term_third = np.dot(n,k1) + np.dot(n,k2)
    term_third = np.where(term_third>0,term_third,0)
    term = np.divide(1, term_third, out=np.zeros_like(term_third),where=term_third!=0.)
    return cv2.merge([term,term,term])


def gaussian(x, x0, xalpha, A):
    return A * np.exp(-((x - x0) / xalpha) ** 2)


# def gaussian(x, x0, xalpha, A):
#     return A * np.exp(-((x - x0) / xalpha) ** 2)


def get_anglemask(root_path_object, path_view, viewDir, max_angle, dark_thr, map_normal, img_front, img_mask, isSave=False):
    '''002:计算掩模, 根据角度以及亮度, 获得参与第一次拟合的像素区域'''

    ## 以观察方向为中心，pi/4范围内的圆锥，也即与viewDir夹角为pi/8内
    # angle = np.arccos(np.dot(map_normal, viewDir).clip(0,1)) *img_mask[:,:,0]
    # img_angle = (cv2.merge([angle,angle,angle])).astype('float32')
    
    # front = img_front[:, :, 0]
    # mask_angle = np.where((angle < max_angle), 1, 0)  # 根据条件计算模板
    # front_shif = front - dark_thr * (1 - angle / max_angle)
    # mask_brightness = np.where((front_shif > 0), 1, 0)  # 根据条件计算模板
    # img_mask_final = mask_angle * mask_brightness * img_mask[:,:,0]
    # img_mask_final = (cv2.merge([img_mask_final,img_mask_final,img_mask_final])).astype('float32')
    
    # 整理法线
    height, width, channel = map_normal.shape
    normal = map_normal.flatten().reshape(height * width, 3)
    front = img_front[:, :, 0].flatten().reshape(height * width)
    mask_regional = img_mask[:, :, 0].flatten().reshape(height * width)

    # 计算夹角
    angle = vg.angle(normal, viewDir, assume_normalized=True, units='rad')  # 输入向量全部做了单位化,输出弧度角
    
    mask_angle = np.where((angle < max_angle), 1, 0)  # 根据条件计算模板

    print('front.max()',front.max())
    front_shif = front - dark_thr * (1 - angle / max_angle)
    # front_shif = front - dark_thr * (1 - angle / angle.max())
    mask_brightness = np.where((front_shif > 0), 1, 0)  # 根据条件计算模板

    mask_angle_tmp = mask_angle
    mask_angle_tmp = np.tile(mask_angle_tmp.reshape(height, width, 1), reps=(1, 1, 3))   # beny
    
    img_angle = np.tile(angle.reshape(height, width, 1), reps=(1, 1, 3))*img_mask  # beny
    img_mask_final = np.tile((mask_angle * mask_brightness * mask_regional).reshape(height, width, 1),
                             reps=(1, 1, 3))
    
    path_npy = root_path_object + '/Show/npy_' + path_view
    np.save(path_npy + '/img_angle.npy', img_angle)
    np.save(path_npy + '/img_mask_angle.npy', img_mask_final)
    
    # 保存图像和数据
    if isSave:
        path_png = root_path_object + '/Show/image_' + path_view
        # img_angle = np.power(img_angle, 1/2.2)
        img_mask_final = np.power(img_mask_final, 1/2.2)
        imageio.imsave(path_png + '/img_mask_angle_tmp.png', (mask_angle_tmp * 255).astype(np.uint8))
        imageio.imsave(path_png + '/img_angle.png', (img_angle/1.6 * 255).astype(np.uint8))
        imageio.imsave(path_png + '/img_mask_angle.png',(img_mask_final / img_mask_final.max() * 255).astype(np.uint8))
    
    return img_angle, img_mask_final


def fit_region(r0, img_mask_final, img_angle, img_front, img_specular, Fit_Counter):
    '''003:数据拟合, 计算区域高光反射系数'''
    # 准备数据
    indexs = np.where(img_mask_final[:, :, 0].flatten() == 1)  # beny
    c = (img_specular[:, :, 0]).flatten()[indexs]
    c = np.where((c > 0.2), c, 0.2)  # 根据条件计算模板
    # data_hk=img_angle.flatten()[indexs]
    data_hk = img_angle[:, :, 0].flatten()[indexs]  # beny
    data_i = img_front[:, :, 0].flatten()[indexs]  # 源代码，三通道偏振差图像
    print('data_i.shape',data_i.shape)
    data_i_temp = func_specular(data_i, data_hk, c, r0)
    print('data_i.max()',data_i_temp.max())
    print('data_i.min()',data_i_temp.min())
    
    data_i = data_i_temp
    data_i = data_i / data_i_temp.max()
    
    # 开始拟合函数
    xx = data_hk  # 主要序号， angle是x轴
    yy = data_i

    # 曲线拟合
    # gmodel = Model(blinnPhong)
    # result = gmodel.fit(yy, x=xx, shininess1=0.35, method='least_squares',
    #                     fit_kws={'loss': 'huber'})
    
    gmodel = Model(blinnPhong2)
    result = gmodel.fit(yy, x=xx, alpha=0.4, intensity=1, method='least_squares',
                        fit_kws={'loss': 'huber'})
    
    # result = gmodel.fit(yy, x=xx, shininess1=0.35, shininess2=0.2, alpha=0.4, method='least_squares',
    #                     fit_kws={'loss': 'huber'})
    
    # gmodel = Model(blinnPhong2)
    

    # 绘制并展示拟合曲线
    show = False
    # show = True
    if show:
        c1 = np.ones_like(c)
        data_ic1 = img_front[:, :, 0].flatten()[indexs]
        data_ic1_temp = func_specular(data_ic1, data_hk, c1, r0)
        data_ic1 = data_ic1_temp
        data_ic1 = data_ic1 / data_ic1_temp.max()
        # print('data_ic1.max()',data_ic1.max())
        # print('data_ic1.min()',data_ic1.min())
        
        min = np.min(data_hk)
        max = np.max(data_hk)
        step1 = 0.001
        step2 = 0.005

        xx_fit = np.arange(min, max, step1)
        xx_mean = np.arange(min, max, step2)

        # 计算分段平均值
        yy_mean = np.zeros_like(xx_mean)
        yy_count = np.zeros_like(xx_mean)

        for hk, i in zip(data_hk, data_i):
            index = int((hk - min) / step2)
            yy_mean[index] = yy_mean[index] + i
            yy_count[index] = yy_count[index] + 1

        yy_mean = np.divide(yy_mean, yy_count, out=np.zeros_like(
            yy_mean), where=yy_count != 0)

        # 获得计算结果
        # shininess1 = result.best_values["shininess1"]
        # shininess2 = result.best_values["shininess2"]
        alpha = result.best_values["alpha"]
        intensity = result.best_values["intensity"]

        # 显示和保存计算结果
        print(">> 开始【拟合p(h)曲线 求ks和shininess】...")
        print(result.fit_report(), "\n\n")

        fig = fig = go.Figure()

        fig.add_trace(go.Scatter(x=data_hk, y=data_i, name="C 修正后数据",
                                mode='markers', marker=dict(color="#001400", size=7)))
        fig.add_trace(
            go.Scatter(x=data_hk, y=data_ic1, name="C=1 数据", mode='markers', marker=dict(color="#AAAAAA", size=5)))
        fig.add_trace(
            go.Scatter(x=xx_mean, y=yy_mean, name="平均数据", mode='lines+markers', marker=dict(color="#FF00FF", size=10)))

        # fig.add_trace(go.Scatter(x=xx_fit, y=blinnPhong(xx_fit, shininess1), name="拟合数据", mode='markers',
        #                          marker=dict(color="#FF0000", size=5)))
                                
        fig.add_trace(go.Scatter(x=xx_fit, y=blinnPhong2(xx_fit, alpha, intensity), name="拟合数据", mode='markers',
                                marker=dict(color="#FF0000", size=5)))
        
        # fig.add_trace(go.Scatter(x=xx_fit, y=blinnPhong2(xx_fit, shininess1, shininess2), name="拟合数据", mode='markers',
        #                          marker=dict(color="#FF0000", size=5)))
        
        # fig.add_trace(go.Scatter(x=xx_fit, y=blinnPhong2(xx_fit, shininess1, shininess2, alpha), name="拟合数据", mode='markers',
        #                          marker=dict(color="#FF0000", size=5)))

        # p.line(y=yy_mean, x=xx_mean, color="blue",line_width=3)

        plotly.offline.plot(fig, filename="Html/SP_hk_i_" +
                            "{0}".format(Fit_Counter) + ".html", auto_open=show)

        with open("Html/SP_fit_" + "{0}".format(Fit_Counter) + ".txt", "w") as f:
            f.write(result.fit_report())
    
        plt.show()

    return result.best_values


def calc_speclar(root_path_object, path_view, img_full, img_angle, img_mask, result, isSave=False):
    '''004:计算高光反照率 C Specular Albedo'''
    # s1 = result['Fit_First']['shininess1']
    # s2 = result['Fit_First']['shininess2']
    # alpha = result['Fit_First']['alpha']
    # alpha = 0.47
    
    s1 = 12
    s2 = 48
    alpha = result['Fit_First']['alpha']
    intensity = result['Fit_First']['intensity']
    
    dictIntegrate = np.load('Dict_Integration.npy', allow_pickle=True).item()
    # n = img_normal
    c_hs = img_full[:, :, 0]  # 原始代码，三通道偏振差图像

    theta = img_angle[:, :, 0]
    c1 = np.zeros_like(c_hs)
    for i in range(img_mask.shape[0]):
        for j in range(img_mask.shape[1]):
            if (img_mask[i, j][0] == 1):

                if (theta[i, j] > 0 and int(theta[i, j] / math.pi * 180) < 89):
                    key1 = '{:.1f}'.format(s1) + '_' + '{:.0f}'.format(theta[i, j] / math.pi * 180)
                    key2 = '{:.1f}'.format(s2) + '_' + '{:.0f}'.format(theta[i, j] / math.pi * 180)
                    pre_integrate1 = dictIntegrate[key1]
                    pre_integrate2 = dictIntegrate[key2]
                    # print('dictIntegrate[key1]',pre_integrate1)
                    # print('dictIntegrate[key2]',pre_integrate2)
                    
                    # c1[i, j] = pre_integrate1
                    c1[i, j] = (1-alpha) * pre_integrate1 + alpha * pre_integrate2
                    c1[i, j] = c1[i, j] * intensity *30

    c = np.divide(c_hs, c1, out=np.zeros_like(c_hs), where=c1 != 0)
    # c = np.clip(c, 0, 1)
    c_rgb = np.float32(cv2.merge([c, c, c]))
    # c1_rgb = np.float32(cv2.merge([c1, c1, c1]))
    # # 保存数据
    # if isSave:
    #     path_png = root_path_object + '/Show/image_' + path_view
    #     cv2.imwrite(path_png + '/albedo_speclar.png', (c_rgb * 255).astype(np.uint8))
    #     cv2.imwrite(path_png + '/albedo_speclar_max.png', (c_rgb / c_rgb.max() * 255).astype(np.uint8))
    #     cv2.imwrite(path_png + '/albedo_speclar_clip.png', (np.clip(c_rgb, 0, 1) * 255).astype(np.uint8))
    #     cv2.imwrite(path_png + '/albedo_speclar_C1.png', (c1_rgb * 255).astype(np.uint8))
    #     cv2.imwrite(path_png + '/albedo_speclar_C1_max.png', (c1_rgb / c1_rgb.max() * 255).astype(np.uint8))
    return c_rgb


def computeRhodtMatrix(k1, normal, rho_dt):
    # GPU Germs 3 p255,size [height,width,channel]
    assert len(
        normal.shape) == 3, 'the size of normal must be [height,width,channel]'
    # k1 = np.float32(k1.reshape(3,1))
    n = normal
    ndotL = np.dot(n, k1).clip(0, 1)  # [h,w]
    lengh = np.shape(rho_dt)[0]
    
    rho_dt_matric = np.trunc(ndotL * lengh)  # 对所有元素取整
    rho_dt_matric_new = np.zeros_like(ndotL)
    for i in range(rho_dt_matric.shape[0]):
        for j in range(rho_dt_matric.shape[1]):
            if rho_dt_matric[i, j] != 0:
                val = int(rho_dt_matric[i, j])
                rho_dt_matric_new[i, j] = rho_dt[val - 1]
            else:
                rho_dt_matric_new[i, j] = rho_dt.min()
                # print('print')
    return cv2.merge([rho_dt_matric_new, rho_dt_matric_new, rho_dt_matric_new])


def computeIrradiance(k1, normal, albedo_specular, rho_dt_matric, map_mask,weight_shading):
    assert len(
        normal.shape) == 3, 'the size of normal must be [height,width,channel]'
    n = normal
    ndotL = np.clip(np.dot(n, k1), 0, 1)  # [h,w]
    ndotL = np.where(ndotL>weight_shading,ndotL,0)
    ndotL_re = cv2.merge([ndotL, ndotL, ndotL])  # [h,w,c]
    rho_dt_L = (1 - albedo_specular * rho_dt_matric)
    rho_dt_L = rho_dt_L * ndotL_re
    irradiance = rho_dt_L
    return irradiance


def computeRhodt(costheta, r0, alpha, intensity, numterms=80):
    # gpu germs 3,p254
    pi = math.pi
    sum = 0.0
    N = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    V = np.zeros_like(N)
    V[1] = np.sqrt(1 - costheta * costheta)
    V[2] = costheta
    for i in range(numterms):
        phip = i / (numterms - 1) * 2.0 * pi
        localsum = 0.0
        cosp = np.cos(phip)
        sinp = np.sin(phip)
        for k in range(numterms):
            thetap = k / (numterms - 1) * pi / 2.0
            sint = np.sin(thetap)
            cost = np.cos(thetap)
            L = np.zeros_like(N)
            L[0] = sinp * sint
            L[1] = cosp * sint
            L[2] = cost
            srval = specularIntegralFunction_s(L, V, N, r0, alpha, intensity)
            temp = srval * sint
            localsum = localsum + temp
        sum = sum + localsum * (pi / 2.0) / numterms
    rho_dt = sum * (2.0 * pi) / numterms
    return rho_dt


def specularIntegralFunction_s(k1, k2, normal, r0, alpha, intensity):
    '''  '''
    assert len(normal.shape) == 1, 'the shape of normal is must [channel,None]'
    n = normal
    ndotL = np.dot(normal, k1)
    
    if ndotL > 0:
        h = np.divide(np.add(k1, k2), np.linalg.norm(np.add(k1, k2)), out=np.zeros_like(np.add(k1, k2)),
                      where=np.linalg.norm(np.add(k1, k2)) != 0)
        F = func_F(r0, k2, h)

        # 整理法线,计算余弦夹角
        NdotH = np.dot(normal, h).clip(0, 1)
        p = blinnPhong_2(NdotH, alpha, intensity).clip(0,1)

        G = func_G(n, h, k1, k2)
        
        rho_molecular = p * F * G
        rho_denominator = np.dot(n, k1) + np.dot(n, k2) - np.dot(n, k1) * np.dot(n, k2)
        rho = np.divide(rho_molecular, rho_denominator, out=np.zeros_like(
            rho_molecular), where=rho_denominator != 0)
        
        val = np.where(rho > 0, rho, 0)
        val = ndotL * val
        # print('val',val)
    else:
        val = 0.
    return val


def specularIntegralFunction(k1, k2, n, r0, alpha, intensity, weight_shading):
    '''  '''
    ndotL = np.dot(n, k1)
    ndotL = np.where(ndotL > weight_shading, ndotL, 0)

    h = np.divide(np.add(k1, k2), np.linalg.norm(np.add(k1, k2)), out=np.zeros_like(np.add(k1, k2)),
                  where=np.linalg.norm(np.add(k1, k2)) != 0)
    # print('h',h)
    F = func_F(r0, k2, h)
    # print('F:', F)
    assert len(n.shape) == 3, 'the size of normal must be [height,width,channel]'
    height, width, channel = n.shape

    # 整理法线 计算余弦夹角
    n_re = n.reshape(height * width, 3)
    NdotH = np.dot(n_re, h).clip(0, 1).reshape(height, width)
    p = blinnPhong_2(NdotH, alpha, intensity).clip(0,1)
    G = func_G(n, h, k1, k2)
    # print('np.max(G):', np.max(G))

    # rho = (p * F * G) / (np.dot(n, k1) + np.dot(n, k2) - np.dot(n, k1) * np.dot(n, k2))
    rho_molecular = p * F * G
    rho_denominator = np.dot(n, k1) + np.dot(n, k2) - np.dot(n, k1) * np.dot(n, k2)
    rho = np.divide(rho_molecular, rho_denominator, out=np.zeros_like(
        rho_molecular), where=rho_denominator != 0)
    # print('np.max(rho)',np.max(rho))

    val = np.where(rho > 0, rho, 0)
    val = ndotL * val
    return val


def addWeighted(input_1, input_2, weights):
    return cv2.addWeighted(input_1, 1, input_2, weights, 0)


def gaussianBlur_sum(img_input, gauss_kernel, sigma_magnification=1):
    # 6个高斯拟合皮肤的三层dipole profile
    sigma = np.array([0.0064, 0.0484, 0.1870, 0.5670,
                      1.9900, 7.4100]) * sigma_magnification
    weights_red = np.array([0.233, 0.100, 0.118, 0.113, 0.358, 0.078])
    weights_green = np.array([0.455, 0.336, 0.198, 0.007, 0.004, 0.000])
    weights_blue = np.array([0.649, 0.344, 0.000, 0.007, 0.000, 0.000])
    h, w, c = img_input.shape
    Len = len(sigma)
    img_blur = np.zeros([Len, h, w, c], np.float32)  # [Len,h,w,c]
    for i in range(Len):
        img_blur[i, :, :, 0] = cv2.GaussianBlur(
            img_input[:, :, 0], (gauss_kernel, gauss_kernel), sigmaX=sigma[i])
        img_blur[i, :, :, 1] = cv2.GaussianBlur(
            img_input[:, :, 1], (gauss_kernel, gauss_kernel), sigmaX=sigma[i])
        img_blur[i, :, :, 2] = cv2.GaussianBlur(
            img_input[:, :, 2], (gauss_kernel, gauss_kernel), sigmaX=sigma[i])

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

    result = cv2.merge([result_r, result_g, result_b])
    return result


def unlitShadingRender(map_normal, map_mask, lightDirs,diffuseIntensity,weight_shading):
    lightDirs = lightDirs / np.linalg.norm(lightDirs, ord=2, axis=1, keepdims=True)
    height, width, channel = map_normal.shape
    num_lights = lightDirs.shape[0]
    
    lightDirs_T = lightDirs[:, :].T
    render_unlit_shading_T = np.dot(map_normal, lightDirs_T).clip(0,1)  # [h,w,num_lights]
    
    # # 矩阵计算，内存可能受不了
    render_unlit_shading = np.float32(np.tile(np.expand_dims(render_unlit_shading_T.transpose(2, 0, 1),-1), reps=(1, 1, 1, channel)))
    render_unlit_shading = np.where(render_unlit_shading>weight_shading,render_unlit_shading,0)
    unlit_mask = np.float32(np.tile(np.expand_dims(map_mask,0), reps=(num_lights, 1, 1, 1)))
    
    
    # ## 循环计算
    # render_unlit_shading = []
    # for k in range(num_lights):
    #     lightDir = lightDirs[k]
    #     render_shading = singleShadingRender(map_normal, map_mask, lightDir,diffuseIntensity,weight_shading)
    #     render_unlit_shading.append(render_shading)
    # render_unlit_shading = np.array(render_unlit_shading).astype('floate32')
    
    return render_unlit_shading

def singleShadingRender(map_normal, map_mask, lightDir,diffuseIntensity,weight_shading):
    shading = (np.dot(map_normal,lightDir)).clip(0,1)
    shading = np.where(shading>weight_shading,shading,0)
    render_shading=cv2.merge([shading,shading,shading])
    return render_shading
    

def unlitSpecularRender(map_normal, map_specular, result, lightDirs, viewDir, r0, lightIntensity, weight_shading):
    lightDirs = lightDirs[:, :]
    lightDirs = lightDirs / np.linalg.norm(lightDirs, ord=2, axis=1, keepdims=True)
    height, width, channel = map_normal.shape
    num_lights = lightDirs.shape[0]
    render_unlit_specular = np.zeros([num_lights, height, width, channel], dtype=np.float32)
    # s1 = result['Fit_First']['shininess1']
    # s2 = result['Fit_First']['shininess1']
    # ks1 = result['Fit_First']['ks1']
    # ks2 = result['Fit_First']['ks2']
    alpha = result['Fit_First']['alpha']
    intensity = result['Fit_First']['intensity']
    n = map_normal
    c = map_specular[:, :, 0]

    k2 = viewDir
    k2 = k2 / np.linalg.norm(k2, ord=2)
    for k in range(num_lights):
        k1 = lightDirs[k, :]
        k1 = k1 / np.linalg.norm(k1, ord=2)
        render_unlit_specular[k] = singleSpecularRender(k1, k2, n, r0, alpha, intensity, c, lightIntensity, weight_shading)

    return render_unlit_specular


def singleSpecularRender(k1, k2, n, r0, alpha, intensity, c, lightIntensity, weight_shading):
    rho = specularIntegralFunction(k1, k2, n, r0, alpha, intensity, weight_shading)
    # rho = rho.clip(0,1) * c
    rho = rho * c
    render_sp = rho * lightIntensity  # 光照强度
    render_specular = np.float32(cv2.merge([render_sp, render_sp, render_sp]))
    
    return render_specular


def unlitSScatterRender(lightDirs,viewDir,map_mask,map_normal,map_sscatter,render_unlit_rho_dt_L,render_rho_dt_V,render_unlit_shading,weight_lambert, weight_shading):
    render_unlit_sscatter = np.zeros_like(render_unlit_rho_dt_L)
    k2 = viewDir
    k2 = k2 / np.linalg.norm(k2, ord=2)
    
    ## 矩阵计算
    # None
    
    ## 循环计算
    for k in range(lightDirs.shape[0]):
        k1 = lightDirs[k,0:]
        k1 = k1 / np.linalg.norm(k1, ord=2)
        render_rho_dt_L = render_unlit_rho_dt_L[k]
        render_shading = render_unlit_shading[k]

        render_sscatter = singleSScatterRender(map_mask,map_normal,map_sscatter,k1,k2,render_rho_dt_L,render_rho_dt_V,render_shading,weight_lambert, weight_shading)
        render_unlit_sscatter[k] = render_sscatter
    
    return render_unlit_sscatter
        

def singleSScatterRender(map_mask,map_normal,map_sscatter,k1,k2,render_rho_dt_L,render_rho_dt_V,render_shading,weight_lambert, weight_shading):
    T_dt = (1 - render_rho_dt_L) * (1 - render_rho_dt_V) 
    cos_angle = np.dot(k2,k1)
    cos_angle = np.where(cos_angle>weight_shading,cos_angle,0)
    phase1 = HG_SScatterPhase(cos_angle, 0.1)
    phase2 = HG_SScatterPhase(cos_angle, 0.8)
    phase = (phase1+phase2).clip(0,1)
    # print('phase：',phase)
    term = HG_SScatterTerm(map_normal, k1, k2).clip(0,1)
    # print('term.shape',term.shape)
    render_sscatter = map_sscatter * T_dt * phase * term
    
    # 使用lambert漫反射模拟后向散射现象
    render_sscatter2 = map_sscatter * render_shading
    render_sscatter = render_sscatter + render_sscatter2 * weight_lambert
    
    return render_sscatter


def unlitSubScatterRender(k2,diffuse_back,map_diffuse,map_specular,map_normal,map_mask,rho_dt,render_unlit_mix_irradiance,ambientIntensity):
    
    ## 矩阵计算
    num_lights, height, width, channel = render_unlit_mix_irradiance.shape
    # 计算依赖于ndotV的rho_dt矩阵
    render_rho_dt_V = computeRhodtMatrix(k2, map_normal, rho_dt)
    render_rho_dt_V = np.float32(render_rho_dt_V)
    render_unlit_rho_dt_V = np.tile(np.expand_dims(render_rho_dt_V,0), reps=(num_lights, 1, 1, 1))
    diffuse_back_num = np.tile(np.expand_dims(diffuse_back,0), reps=(num_lights, 1, 1, 1))
    
    render_unlit_subscatter_back = render_unlit_mix_irradiance * diffuse_back_num    # 不乘以 pi
    
    # 考虑从内往外散射
    render_unlit_subscatter_out = render_unlit_subscatter_back * (1 - map_specular * render_unlit_rho_dt_V)
    
    if ambientIntensity !=0:
        # 考虑全局间接光照
        diffuse_num = np.tile(np.expand_dims(map_diffuse,0), reps=(num_lights, 1, 1, 1))
        render_unlit_subscatter = render_unlit_subscatter_out * (1-ambientIntensity)
    
        render_unlit_global = diffuse_num * ambientIntensity
        render_unlit_subscatter = render_unlit_subscatter + render_unlit_global
    else:
        render_unlit_subscatter = render_unlit_subscatter_out
    
    # ## 循环计算
    # render_unlit_subscatter = np.zeros_like(render_unlit_mix_irradiance)
    # for k in range(render_unlit_mix_irradiance.shape[0]):
    #     render_mix_irradiance = render_unlit_mix_irradiance[k]
    #     render_subscatter = singleSubScatterRender(k2,diffuse_back,map_diffuse,map_specular,map_normal,map_mask,rho_dt,render_mix_irradiance,ambientIntensity)
    #     render_unlit_subscatter[k] = render_subscatter
    
    return render_unlit_subscatter


def singleSubScatterRender(k2,diffuse_back,map_diffuse,map_specular,map_normal,map_mask,rho_dt,render_mix_irradiance,ambientIntensity):
    
    ## 计算后散射分量
    render_subscatter_back = render_mix_irradiance * diffuse_back
    ## 计算依赖于ndotV的rho_dt矩阵
    render_rho_dt_V = computeRhodtMatrix(k2, map_normal, rho_dt)
    ## 考虑从内往外散射
    render_subscatter_out = render_subscatter_back * (1 - map_specular * render_rho_dt_V)
    ## 考虑全局间接光照
    render_global = map_diffuse * ambientIntensity
    ## 最终的次表面散射分量
    render_subscatter = render_subscatter_out * (1 - ambientIntensity) + render_global
    
    return render_subscatter


def unlitDiffuseRender(map_diffuse,render_unlit_shading,ambientIntensity):
    num_lights = render_unlit_shading.shape[0]

    ## 矩阵计算
    map_diffuse_num = np.tile(np.expand_dims(map_diffuse,0),
                                    reps=(num_lights, 1, 1, 1)).astype(np.float32)
    render_unlit_diffuse = map_diffuse_num * render_unlit_shading * (1-ambientIntensity)
    render_unlit_global = map_diffuse_num * ambientIntensity
    render_unlit_diffuse = render_unlit_diffuse + render_unlit_global
    
    # ## 循环计算
    # render_unlit_diffuse = []
    # for k in range(num_lights):
    #     render_shading = render_unlit_shading[k]
    #     render_unlit_diffuse.append(singleDiffuseRender(map_diffuse,render_shading,ambientIntensity))
    # render_unlit_diffuse = np.array(render_unlit_diffuse).astype('float32')
    
    return render_unlit_diffuse


def singleDiffuseRender(map_diffuse,render_shading,ambientIntensity):
    
    render_diffuse = map_diffuse * render_shading * (1-ambientIntensity)
    render_global = map_diffuse * ambientIntensity
    render_out = render_diffuse + render_global
    return render_out
    

def addImages(renderImages_1,renderImages_2,flag=1):
    ## 矩阵计算
    renderImages = renderImages_1 + renderImages_2
    renderImages = normalClipAndMax(renderImages, flag)
    
    # ## 循环计算
    # renderImages = []
    # for k in range(renderImages_1.shape[0]):
    #     renderImages.append(addImage(renderImages_1[k],renderImages_2[k],flag))
    # renderImages = np.array(renderImages).astype('float32')
    
    return renderImages


def addImage(renderImage_1,renderImage_2,flag=1):
    renderImage = renderImage_1 + renderImage_2
    renderImage = normalClipAndMax(renderImage, flag)
    return renderImage


def out_video(root_path_object, path_view, file_name, fps, size, video_suffix):
    # print('file_name',file_name)
    # fourcc = cv2.VideoWriter_fourcc('I','4','2','0')    # 未压缩
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # 压缩为1/10
    out_path = root_path_object + '/Show/video_' + \
        path_view + '/' + file_name + '_001' + video_suffix
    video = cv2.VideoWriter(out_path, fourcc, fps, size)
    files = sorted(glob.glob(root_path_object + '/Show/image_' +
                             path_view + '/' + file_name + '/*.png'))

    for file in files:
        # print('file:', file)
        img = cv2.imread(file)
        video.write(img)
    video.release()


def outComposeVideos(lists, outname):
    files = lists
    clipArrays = []
    tmpClip = []
    for file in files:
        # print('file', files)
        clip = mpe.VideoFileClip(file)
        tmpClip.append(clip)
    clipArrays.append(tmpClip)

    # 视频拼接
    # destClip = mpe.concatenate_videoclips(tmpClip)
    # destClip.write_videofile(object_path+outname)

    # 视频堆叠
    destClip_2 = mpe.clips_array(clipArrays)
    destClip_2.write_videofile(outname)


## -----------------------------------Traning Data--------------
def createTrainFolders(root_path_object, path_object, path_views):
    path_npy = os.path.join(root_path_object, 'npy', path_object)
    xmkdir(path_npy)
    # list_type = ['ori', 'crop', 'resize']
    list_type = ['crop', 'resize']
    # list_type = ['crop']
    list_files = ['all','train', 'val', 'test']
    for i in range(len(list_type)):
        path_image = os.path.join(root_path_object, 'image', path_object) + '/' + list_type[i]
        for k in range (len(list_files)):
            path_all = path_image + '/' + list_files[k]
            xmkdir(path_all + '/image_mix')
            xmkdir(path_all + '/image_diffuse')
            xmkdir(path_all + '/image_mask')
            xmkdir(path_all + '/image_light')
            
            xmkdir(path_all + '/coord3')
            xmkdir(path_all + '/map_albedo_capture')
            xmkdir(path_all + '/map_albedo_mix28')
            xmkdir(path_all + '/map_albedo_diffuse28')
            xmkdir(path_all + '/map_normal_mix7')
            xmkdir(path_all + '/map_normal_mix28')
            xmkdir(path_all + '/map_normal_diffuse7')
            xmkdir(path_all + '/map_normal_diffuse28')
            xmkdir(path_all + '/map_mask')
            
            xmkdir(path_all + '/lightinfo')


def saveTrainDataCrop(path_gene_object, path_view, input, diffuse, dtL, lightpng, dtV, a_d, a_ss, a_sp, norm, mask, lightinfo, index, isCrop, size):
    '''
    input:[h,w,c], diffuse:[h,w,c], dtL:[h,w,c], lightpng:[h,w,c], dtV:[h,w,c], a_d:[h,w,c], a_ss:[h,w,c],a_sp:[h,w,c], norm:[h,w,c], mask:[h,w,c], lightinfo:[11]
    return:[]
    '''
    path_png = os.path.join(path_gene_object, 'image')
    personName = os.path.split(path_gene_object)[1]
    image_input = input
    image_diff = diffuse
    diffuse = a_d
    sscatter_g = a_ss
    specular_g = a_sp
    ss_sp_dtV = np.float32(cv2.merge([a_ss[:,:,0], a_sp[:,:,0], dtV[:,:,0]]))
    normal = norm
    mask = mask
    lightpng = lightpng
    lightinfo = lightinfo.reshape(1, -1)
    
    
    if isCrop:
        crop_step = 512
        crop_size = 1024
        
        height, width, channel = image_input.shape
        assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        num_row = (height - crop_size) / crop_step + 1
        num_col = (width - crop_size) / crop_step + 1
        count = 0
        mask_threshold = crop_size * crop_size *0.6
        # print('mask_threshold',mask_threshold)
        for row in range(int(num_row)):
            for col in range(int(num_col)):
                crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                if count_nonzero > mask_threshold:
                    # print('count_nonzero',count_nonzero)
                    crop_image_input = cropSavefiles(image_input, crop_size, crop_step, row, col)
                    crop_image_diff = cropSavefiles(image_diff, crop_size, crop_step, row, col)
                    crop_diffuse = cropSavefiles(diffuse, crop_size, crop_step, row, col)
                    crop_sscatter_g = cropSavefiles(sscatter_g, crop_size, crop_step, row, col)
                    crop_specular_g = cropSavefiles(specular_g, crop_size, crop_step, row, col)
                    crop_ss_sp_dtV = cropSavefiles(ss_sp_dtV, crop_size, crop_step, row, col)
                    crop_dtL = cropSavefiles(dtL, crop_size, crop_step, row, col)
                    crop_normal = cropSavefiles(normal, crop_size, crop_step, row, col)
                    crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                    crop_lightpng = cropSavefiles(lightpng, crop_size, crop_step, row, col)
                    
                    savePNG(path_png+'/image_input/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_image_input,resize=size)
                    savePNG(path_png+'/image_diff/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_image_diff,resize=size)
                    savePNG(path_png+'/diffuse/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_diffuse,resize=size)

                    savePNG(path_png+'/sscatter_g/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_sscatter_g,resize=size)
                    savePNG(path_png+'/specular_g/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_specular_g,resize=size)
                    savePNG(path_png+'/ss_sp_dtV/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_ss_sp_dtV,resize=size)
                    savePNG(path_png+'/dtL/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_dtL,resize=size)

                    savePNG(path_png+'/normal/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_normal,resize=size)
                    savePNG(path_png+'/mask/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mask,resize=size)
                    savePNG(path_png+'/lightpng/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_lightpng,resize=size)
                    
                    np.savetxt(path_png+'/lightinfo/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.txt', lightinfo)
                    count = count + 1
    

def saveTrainData(path_gene_object, type, path_view, mix, diffuse, mask, lightpng, coord,
                     map_albedo_capture,map_albedo_mix28,map_albedo_diffuse28,map_normal_mix7,map_normal_mix28,map_normal_diffuse7,map_normal_diffuse28,
                     lightinfo, map_mask, index, size):
    '''
    input:[h,w,c], diffuse:[h,w,c], mask:[h,w,c], lightpng:[h,w,c], dtV:[h,w,c], a_d:[h,w,c], a_sp:[h,w,c], norm:[h,w,c], mask:[h,w,c], lightinfo:[6]
    return:[]
    '''
    
    personName = os.path.split(path_gene_object)[1]
    lightinfo = lightinfo.reshape(1, -1)
    path_png = os.path.join(path_gene_object, type, 'all')
    
    if type == 'ori':
        savePNG(path_png+'/image_mix/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', mix)
        savePNG(path_png+'/image_diffuse/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', diffuse)
        savePNG(path_png+'/image_mask/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', mask)
        savePNG(path_png+'/image_light/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', lightpng)
        
        savePNG(path_png+'/coord3/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', coord)
        savePNG(path_png+'/map_albedo_capture/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_capture)
        savePNG(path_png+'/map_albedo_mix28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_mix28)
        savePNG(path_png+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_diffuse28)
        
        savePNG(path_png+'/map_normal_mix7/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_mix7)
        savePNG(path_png+'/map_normal_mix28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_mix28)
        savePNG(path_png+'/map_normal_diffuse7/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_diffuse7)
        savePNG(path_png+'/map_normal_diffuse28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_diffuse28)
        savePNG(path_png+'/map_mask/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_mask)
        
        np.savetxt(path_png+'/lightinfo/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.txt', lightinfo)
        
    elif type == 'crop':
        # crop_step = 512
        # crop_size = 1024
        
        crop_step = int(size/2)
        crop_size = size
        
        height, width, channel = mix.shape
        assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        num_row = (height - crop_size) / crop_step + 1
        num_col = (width - crop_size) / crop_step + 1
        count = 0
        mask_threshold = crop_size * crop_size *0.6
        # print('mask_threshold',mask_threshold)
        for row in range(int(num_row)):
            for col in range(int(num_col)):
                crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                if count_nonzero > mask_threshold:
                    # print('count_nonzero',count_nonzero)
                    crop_mix = cropSavefiles(mix, crop_size, crop_step, row, col)
                    crop_diffuse = cropSavefiles(diffuse, crop_size, crop_step, row, col)
                    crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                    crop_lightpng = cropSavefiles(lightpng, crop_size, crop_step, row, col)
                    
                    crop_coord = cropSavefiles(coord, crop_size, crop_step, row, col)
                    crop_map_albedo_capture = cropSavefiles(map_albedo_capture, crop_size, crop_step, row, col)
                    crop_map_albedo_mix28 = cropSavefiles(map_albedo_mix28, crop_size, crop_step, row, col)
                    crop_map_albedo_diffuse28 = cropSavefiles(map_albedo_diffuse28, crop_size, crop_step, row, col)
                    
                    crop_map_normal_mix7 = cropSavefiles(map_normal_mix7, crop_size, crop_step, row, col)
                    crop_map_normal_mix28 = cropSavefiles(map_normal_mix28, crop_size, crop_step, row, col)
                    crop_map_normal_diffuse7 = cropSavefiles(map_normal_diffuse7, crop_size, crop_step, row, col)
                    crop_map_normal_diffuse28 = cropSavefiles(map_normal_diffuse28, crop_size, crop_step, row, col)
                    crop_map_mask = cropSavefiles(map_mask, crop_size, crop_step, row, col)
                    
                    savePNG(path_png+'/image_mix/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mix,resize=size)
                    savePNG(path_png+'/image_diffuse/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_diffuse,resize=size)
                    savePNG(path_png+'/image_mask/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mask,resize=size)
                    savePNG(path_png+'/image_light/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_lightpng,resize=size)
                    
                    
                    savePNG(path_png+'/coord3/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_coord,resize=size)
                    savePNG(path_png+'/map_albedo_capture/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_capture,resize=size)
                    savePNG(path_png+'/map_albedo_mix28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_mix28,resize=size)
                    savePNG(path_png+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_diffuse28,resize=size)

                    savePNG(path_png+'/map_normal_mix7/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix7,resize=size)
                    savePNG(path_png+'/map_normal_mix28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix28,resize=size)
                    savePNG(path_png+'/map_normal_diffuse7/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse7,resize=size)
                    savePNG(path_png+'/map_normal_diffuse28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse28,resize=size)
                    savePNG(path_png+'/map_mask/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_mask,resize=size)
                    
                    np.savetxt(path_png+'/lightinfo/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.txt', lightinfo)
                    count = count + 1
        
        
    elif type == 'resize':
        savePNG(path_png+'/image_mix/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', mix,resize=size)
        savePNG(path_png+'/image_diffuse/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', diffuse,resize=size)
        savePNG(path_png+'/image_mask/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', mask,resize=size)
        savePNG(path_png+'/image_light/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', lightpng,resize=size)
        
        
        savePNG(path_png+'/coord3/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', coord,resize=size)
        savePNG(path_png+'/map_albedo_capture/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_capture,resize=size)
        savePNG(path_png+'/map_albedo_mix28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_mix28,resize=size)

        savePNG(path_png+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_diffuse28,resize=size)
        savePNG(path_png+'/map_normal_mix7/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_mix7,resize=size)
        savePNG(path_png+'/map_normal_mix28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_mix28,resize=size)
        savePNG(path_png+'/map_normal_diffuse7/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_diffuse7,resize=size)
        savePNG(path_png+'/map_normal_diffuse28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_diffuse28,resize=size)
        savePNG(path_png+'/map_mask/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_mask,resize=size)
        
        np.savetxt(path_png+'/lightinfo/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.txt', lightinfo)
        
        
    elif type == 'all':
        ##------------‘ori’
        path_png_ori = os.path.join(path_gene_object, 'ori', 'all')
        savePNG(path_png_ori+'/image_mix/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', mix)
        savePNG(path_png_ori+'/image_diffuse/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', diffuse)
        savePNG(path_png_ori+'/image_mask/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', mask)
        savePNG(path_png_ori+'/image_light/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', lightpng)
        
        savePNG(path_png_ori+'/coord3/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', coord)
        savePNG(path_png_ori+'/map_albedo_capture/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_capture)
        savePNG(path_png_ori+'/map_albedo_mix28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_mix28)
        savePNG(path_png_ori+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_diffuse28)
        
        savePNG(path_png_ori+'/map_normal_mix7/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_mix7)
        savePNG(path_png_ori+'/map_normal_mix28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_mix28)
        savePNG(path_png_ori+'/map_normal_diffuse7/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_diffuse7)
        savePNG(path_png_ori+'/map_normal_diffuse28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_diffuse28)
        savePNG(path_png_ori+'/map_mask/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_mask)
        
        np.savetxt(path_png_ori+'/lightinfo/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.txt', lightinfo)
        
        
        #----------------------'resize'
        path_png_resize = os.path.join(path_gene_object, 'resize', 'all')
        savePNG(path_png_resize+'/image_mix/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', mix,resize=size)
        savePNG(path_png_resize+'/image_diffuse/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', diffuse,resize=size)
        savePNG(path_png_resize+'/image_mask/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', mask,resize=size)
        savePNG(path_png_resize+'/image_light/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', lightpng,resize=size)
        
        
        savePNG(path_png_resize+'/coord3/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', coord,resize=size)
        savePNG(path_png_resize+'/map_albedo_capture/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_capture,resize=size)
        savePNG(path_png_resize+'/map_albedo_mix28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_mix28,resize=size)

        savePNG(path_png_resize+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_albedo_diffuse28,resize=size)
        savePNG(path_png_resize+'/map_normal_mix7/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_mix7,resize=size)
        savePNG(path_png_resize+'/map_normal_mix28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_mix28,resize=size)
        savePNG(path_png_resize+'/map_normal_diffuse7/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_diffuse7,resize=size)
        savePNG(path_png_resize+'/map_normal_diffuse28/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_normal_diffuse28,resize=size)
        savePNG(path_png_resize+'/map_mask/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', map_mask,resize=size)
        
        np.savetxt(path_png_resize+'/lightinfo/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.txt', lightinfo)
        
        
        ##--------------------'crop'
        path_png_crop = os.path.join(path_gene_object, 'crop', 'all')
        crop_step = int(size/2)
        crop_size = size
        
        height, width, channel = mix.shape
        assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        num_row = (height - crop_size) / crop_step + 1
        num_col = (width - crop_size) / crop_step + 1
        count = 0
        mask_threshold = crop_size * crop_size *0.6
        # print('mask_threshold',mask_threshold)
        for row in range(int(num_row)):
            for col in range(int(num_col)):
                crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                if count_nonzero > mask_threshold:
                    # print('count_nonzero',count_nonzero)
                    crop_mix = cropSavefiles(mix, crop_size, crop_step, row, col)
                    crop_diffuse = cropSavefiles(diffuse, crop_size, crop_step, row, col)
                    crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                    crop_lightpng = cropSavefiles(lightpng, crop_size, crop_step, row, col)
                    
                    crop_coord = cropSavefiles(coord, crop_size, crop_step, row, col)
                    crop_map_albedo_capture = cropSavefiles(map_albedo_capture, crop_size, crop_step, row, col)
                    crop_map_albedo_mix28 = cropSavefiles(map_albedo_mix28, crop_size, crop_step, row, col)
                    crop_map_albedo_diffuse28 = cropSavefiles(map_albedo_diffuse28, crop_size, crop_step, row, col)
                    
                    crop_map_normal_mix7 = cropSavefiles(map_normal_mix7, crop_size, crop_step, row, col)
                    crop_map_normal_mix28 = cropSavefiles(map_normal_mix28, crop_size, crop_step, row, col)
                    crop_map_normal_diffuse7 = cropSavefiles(map_normal_diffuse7, crop_size, crop_step, row, col)
                    crop_map_normal_diffuse28 = cropSavefiles(map_normal_diffuse28, crop_size, crop_step, row, col)
                    crop_map_mask = cropSavefiles(map_mask, crop_size, crop_step, row, col)
                    
                    savePNG(path_png_crop+'/image_mix/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mix,resize=size)
                    savePNG(path_png_crop+'/image_diffuse/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_diffuse,resize=size)
                    savePNG(path_png_crop+'/image_mask/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mask,resize=size)
                    savePNG(path_png_crop+'/image_light/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_lightpng,resize=size)
                    
                    
                    savePNG(path_png_crop+'/coord3/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_coord,resize=size)
                    savePNG(path_png_crop+'/map_albedo_capture/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_capture,resize=size)
                    savePNG(path_png_crop+'/map_albedo_mix28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_mix28,resize=size)
                    savePNG(path_png_crop+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_diffuse28,resize=size)

                    savePNG(path_png_crop+'/map_normal_mix7/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix7,resize=size)
                    savePNG(path_png_crop+'/map_normal_mix28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix28,resize=size)
                    savePNG(path_png_crop+'/map_normal_diffuse7/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse7,resize=size)
                    savePNG(path_png_crop+'/map_normal_diffuse28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse28,resize=size)
                    savePNG(path_png_crop+'/map_mask/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_mask,resize=size)
                    
                    np.savetxt(path_png_crop+'/lightinfo/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.txt', lightinfo)
                    count = count + 1
        

def saveTrainDataPart(path_gene_object, type, path_view, mix, diffuse, mask, lightpng, coord,
                     map_albedo_capture,map_albedo_mix28,map_albedo_diffuse28,map_normal_mix7,map_normal_mix28,map_normal_diffuse7,map_normal_diffuse28,
                     lightinfo, index, size):
    '''
    input:[h,w,c], diffuse:[h,w,c], mask:[h,w,c], lightpng:[h,w,c], dtV:[h,w,c], a_d:[h,w,c], a_sp:[h,w,c], norm:[h,w,c], mask:[h,w,c], lightinfo:[6]
    return:[]
    '''
    
    personName = os.path.split(path_gene_object)[1]
    lightinfo = lightinfo.reshape(1, -1)
    path_png = os.path.join(path_gene_object, type, 'all')
    
    if type == 'ori':
        # savePNG(path_png+'/image_mix/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', mix)
        # savePNG(path_png+'/image_diffuse/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', diffuse)
        # savePNG(path_png+'/image_mask/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', mask)
        # savePNG(path_png+'/image_light/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', lightpng)
        
        savePNG(path_png+'/coord3/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', coord)
        # savePNG(path_png+'/map_albedo_capture/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_capture)
        # savePNG(path_png+'/map_albedo_mix28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_mix28)
        # savePNG(path_png+'/map_albedo_diffuse28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_diffuse28)
        
        # savePNG(path_png+'/map_normal_mix7/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_mix7)
        # savePNG(path_png+'/map_normal_mix28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_mix28)
        # savePNG(path_png+'/map_normal_diffuse7/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_diffuse7)
        # savePNG(path_png+'/map_normal_diffuse28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_diffuse28)
        
        # np.savetxt(path_png+'/lightinfo/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.txt', lightinfo)
        
    elif type == 'crop':
        crop_step = 512
        crop_size = 1024
        
        height, width, channel = mix.shape
        assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        num_row = (height - crop_size) / crop_step + 1
        num_col = (width - crop_size) / crop_step + 1
        count = 0
        mask_threshold = crop_size * crop_size *0.6
        # print('mask_threshold',mask_threshold)
        for row in range(int(num_row)):
            for col in range(int(num_col)):
                crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                if count_nonzero > mask_threshold:
                    # print('count_nonzero',count_nonzero)
                    crop_mix = cropSavefiles(mix, crop_size, crop_step, row, col)
                    crop_diffuse = cropSavefiles(diffuse, crop_size, crop_step, row, col)
                    crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                    crop_lightpng = cropSavefiles(lightpng, crop_size, crop_step, row, col)
                    
                    crop_coord = cropSavefiles(coord, crop_size, crop_step, row, col)
                    crop_map_albedo_capture = cropSavefiles(map_albedo_capture, crop_size, crop_step, row, col)
                    crop_map_albedo_mix28 = cropSavefiles(map_albedo_mix28, crop_size, crop_step, row, col)
                    crop_map_albedo_diffuse28 = cropSavefiles(map_albedo_diffuse28, crop_size, crop_step, row, col)
                    
                    crop_map_normal_mix7 = cropSavefiles(map_normal_mix7, crop_size, crop_step, row, col)
                    crop_map_normal_mix28 = cropSavefiles(map_normal_mix28, crop_size, crop_step, row, col)
                    crop_map_normal_diffuse7 = cropSavefiles(map_normal_diffuse7, crop_size, crop_step, row, col)
                    crop_map_normal_diffuse28 = cropSavefiles(map_normal_diffuse28, crop_size, crop_step, row, col)
                    
                    savePNG(path_png+'/image_mix/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mix,resize=size)
                    savePNG(path_png+'/image_diffuse/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_diffuse,resize=size)
                    savePNG(path_png+'/image_mask/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mask,resize=size)
                    savePNG(path_png+'/image_light/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_lightpng,resize=size)
                    
                    
                    savePNG(path_png+'/coord3/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_coord,resize=size)
                    savePNG(path_png+'/map_albedo_capture/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_capture,resize=size)
                    savePNG(path_png+'/map_albedo_mix28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_mix28,resize=size)
                    savePNG(path_png+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_diffuse28,resize=size)

                    savePNG(path_png+'/map_normal_mix7/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix7,resize=size)
                    savePNG(path_png+'/map_normal_mix28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix28,resize=size)
                    savePNG(path_png+'/map_normal_diffuse7/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse7,resize=size)
                    savePNG(path_png+'/map_normal_diffuse28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse28,resize=size)
                    
                    np.savetxt(path_png+'/lightinfo/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.txt', lightinfo)
                    count = count + 1
        
        
    elif type == 'resize':
        # savePNG(path_png+'/image_mix/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', mix,resize=size)
        # savePNG(path_png+'/image_diffuse/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', diffuse,resize=size)
        # savePNG(path_png+'/image_mask/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', mask,resize=size)
        # savePNG(path_png+'/image_light/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', lightpng,resize=size)
        
        
        savePNG(path_png+'/coord3/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', coord,resize=size)
        # savePNG(path_png+'/map_albedo_capture/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_capture,resize=size)
        # savePNG(path_png+'/map_albedo_mix28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_mix28,resize=size)

        # savePNG(path_png+'/map_albedo_diffuse28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_diffuse28,resize=size)
        # savePNG(path_png+'/map_normal_mix7/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_mix7,resize=size)
        # savePNG(path_png+'/map_normal_mix28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_mix28,resize=size)
        # savePNG(path_png+'/map_normal_diffuse7/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_diffuse7,resize=size)
        # savePNG(path_png+'/map_normal_diffuse28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_diffuse28,resize=size)
        
        # np.savetxt(path_png+'/lightinfo/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.txt', lightinfo)
        
        
    elif type == 'all':
        ##------------‘ori’
        path_png_ori = os.path.join(path_gene_object, 'ori', 'all')
        # savePNG(path_png_ori+'/image_mix/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', mix)
        # savePNG(path_png_ori+'/image_diffuse/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', diffuse)
        # savePNG(path_png_ori+'/image_mask/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', mask)
        # savePNG(path_png_ori+'/image_light/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', lightpng)
        
        savePNG(path_png_ori+'/coord3/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', coord)
        # savePNG(path_png_ori+'/map_albedo_capture/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_capture)
        # savePNG(path_png_ori+'/map_albedo_mix28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_mix28)
        # savePNG(path_png_ori+'/map_albedo_diffuse28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_diffuse28)
        
        # savePNG(path_png_ori+'/map_normal_mix7/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_mix7)
        # savePNG(path_png_ori+'/map_normal_mix28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_mix28)
        # savePNG(path_png_ori+'/map_normal_diffuse7/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_diffuse7)
        # savePNG(path_png_ori+'/map_normal_diffuse28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_diffuse28)
        
        # np.savetxt(path_png_ori+'/lightinfo/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.txt', lightinfo)
        
        
        #----------------------'resize'
        path_png_resize = os.path.join(path_gene_object, 'resize', 'all')
        # savePNG(path_png_resize+'/image_mix/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', mix,resize=size)
        # savePNG(path_png_resize+'/image_diffuse/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', diffuse,resize=size)
        # savePNG(path_png_resize+'/image_mask/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', mask,resize=size)
        # savePNG(path_png_resize+'/image_light/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', lightpng,resize=size)
        
        
        savePNG(path_png_resize+'/coord3/'+personName+'_'+path_view +
                    f'_{str(index).zfill(4)}.png', coord,resize=size)
        # savePNG(path_png_resize+'/map_albedo_capture/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_capture,resize=size)
        # savePNG(path_png_resize+'/map_albedo_mix28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_mix28,resize=size)

        # savePNG(path_png_resize+'/map_albedo_diffuse28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_albedo_diffuse28,resize=size)
        # savePNG(path_png_resize+'/map_normal_mix7/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_mix7,resize=size)
        # savePNG(path_png_resize+'/map_normal_mix28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_mix28,resize=size)
        # savePNG(path_png_resize+'/map_normal_diffuse7/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_diffuse7,resize=size)
        # savePNG(path_png_resize+'/map_normal_diffuse28/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.png', map_normal_diffuse28,resize=size)
        
        # np.savetxt(path_png_resize+'/lightinfo/'+personName+'_'+path_view +
        #             f'_{str(index).zfill(4)}.txt', lightinfo)
        
        
        ##--------------------'crop'
        path_png_crop = os.path.join(path_gene_object, 'crop', 'all')
        crop_step = 512
        crop_size = 1024
        
        height, width, channel = mix.shape
        assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
        num_row = (height - crop_size) / crop_step + 1
        num_col = (width - crop_size) / crop_step + 1
        count = 0
        mask_threshold = crop_size * crop_size *0.6
        # print('mask_threshold',mask_threshold)
        for row in range(int(num_row)):
            for col in range(int(num_col)):
                crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                if count_nonzero > mask_threshold:
                    # print('count_nonzero',count_nonzero)
                    crop_mix = cropSavefiles(mix, crop_size, crop_step, row, col)
                    crop_diffuse = cropSavefiles(diffuse, crop_size, crop_step, row, col)
                    crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                    crop_lightpng = cropSavefiles(lightpng, crop_size, crop_step, row, col)
                    
                    crop_coord = cropSavefiles(coord, crop_size, crop_step, row, col)
                    crop_map_albedo_capture = cropSavefiles(map_albedo_capture, crop_size, crop_step, row, col)
                    crop_map_albedo_mix28 = cropSavefiles(map_albedo_mix28, crop_size, crop_step, row, col)
                    crop_map_albedo_diffuse28 = cropSavefiles(map_albedo_diffuse28, crop_size, crop_step, row, col)
                    
                    crop_map_normal_mix7 = cropSavefiles(map_normal_mix7, crop_size, crop_step, row, col)
                    crop_map_normal_mix28 = cropSavefiles(map_normal_mix28, crop_size, crop_step, row, col)
                    crop_map_normal_diffuse7 = cropSavefiles(map_normal_diffuse7, crop_size, crop_step, row, col)
                    crop_map_normal_diffuse28 = cropSavefiles(map_normal_diffuse28, crop_size, crop_step, row, col)
                    
                    savePNG(path_png_crop+'/image_mix/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mix,resize=size)
                    savePNG(path_png_crop+'/image_diffuse/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_diffuse,resize=size)
                    savePNG(path_png_crop+'/image_mask/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mask,resize=size)
                    savePNG(path_png_crop+'/image_light/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_lightpng,resize=size)
                    
                    
                    savePNG(path_png_crop+'/coord3/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_coord,resize=size)
                    savePNG(path_png_crop+'/map_albedo_capture/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_capture,resize=size)
                    savePNG(path_png_crop+'/map_albedo_mix28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_mix28,resize=size)
                    savePNG(path_png_crop+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_diffuse28,resize=size)

                    savePNG(path_png_crop+'/map_normal_mix7/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix7,resize=size)
                    savePNG(path_png_crop+'/map_normal_mix28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix28,resize=size)
                    savePNG(path_png_crop+'/map_normal_diffuse7/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse7,resize=size)
                    savePNG(path_png_crop+'/map_normal_diffuse28/'+personName+'_'+path_view +
                                f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse28,resize=size)
                    
                    np.savetxt(path_png_crop+'/lightinfo/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.txt', lightinfo)
                    count = count + 1
 
def saveTrainDataOriCrop(path_gene_object, type, path_view, mix, diffuse, mask, lightpng, coord,
                     map_albedo_capture,map_albedo_mix28,map_albedo_diffuse28,map_normal_mix7,map_normal_mix28,map_normal_diffuse7,map_normal_diffuse28,
                     lightinfo, map_mask, index, size):
    '''
    input:[h,w,c], diffuse:[h,w,c], mask:[h,w,c], lightpng:[h,w,c], dtV:[h,w,c], a_d:[h,w,c], a_sp:[h,w,c], norm:[h,w,c], mask:[h,w,c], lightinfo:[6]
    return:[]
    '''
    
    personName = os.path.split(path_gene_object)[1]
    lightinfo = lightinfo.reshape(1, -1)
    path_png = os.path.join(path_gene_object, type, 'all')
    assert type == 'crop', 'type must be crop'

    crop_step = int(size/2)
    crop_size = size
    
    height, width, channel = mix.shape
    assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
    assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
    num_row = (height - crop_size) / crop_step + 1
    num_col = (width - crop_size) / crop_step + 1
    count = 0
    mask_threshold = crop_size * crop_size *0.75
    # print('mask_threshold',mask_threshold)
    for row in range(int(num_row)):
        for col in range(int(num_col)):
            # crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
            crop_mask = cropSavefiles(map_mask, crop_size, crop_step, row, col)
            count_nonzero = np.count_nonzero(crop_mask[:,:,0])
            if count_nonzero > mask_threshold:
                # print('count_nonzero',count_nonzero)
                crop_mix = cropSavefiles(mix, crop_size, crop_step, row, col)
                crop_diffuse = cropSavefiles(diffuse, crop_size, crop_step, row, col)
                crop_mask = cropSavefiles(mask, crop_size, crop_step, row, col)
                crop_lightpng = cropSavefiles(lightpng, crop_size, crop_step, row, col)
                
                crop_coord = cropSavefiles(coord, crop_size, crop_step, row, col)
                crop_map_albedo_capture = cropSavefiles(map_albedo_capture, crop_size, crop_step, row, col)
                crop_map_albedo_mix28 = cropSavefiles(map_albedo_mix28, crop_size, crop_step, row, col)
                crop_map_albedo_diffuse28 = cropSavefiles(map_albedo_diffuse28, crop_size, crop_step, row, col)
                
                crop_map_normal_mix7 = cropSavefiles(map_normal_mix7, crop_size, crop_step, row, col)
                crop_map_normal_mix28 = cropSavefiles(map_normal_mix28, crop_size, crop_step, row, col)
                crop_map_normal_diffuse7 = cropSavefiles(map_normal_diffuse7, crop_size, crop_step, row, col)
                crop_map_normal_diffuse28 = cropSavefiles(map_normal_diffuse28, crop_size, crop_step, row, col)
                crop_map_mask = cropSavefiles(map_mask, crop_size, crop_step, row, col)
                
                savePNG(path_png+'/image_mix/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mix,resize=size)
                savePNG(path_png+'/image_diffuse/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_diffuse,resize=size)
                savePNG(path_png+'/image_mask/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mask,resize=size)
                savePNG(path_png+'/image_light/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_lightpng,resize=size)
                
                
                savePNG(path_png+'/coord3/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_coord,resize=size)
                savePNG(path_png+'/map_albedo_capture/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_capture,resize=size)
                savePNG(path_png+'/map_albedo_mix28/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_mix28,resize=size)
                savePNG(path_png+'/map_albedo_diffuse28/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_albedo_diffuse28,resize=size)

                savePNG(path_png+'/map_normal_mix7/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix7,resize=size)
                savePNG(path_png+'/map_normal_mix28/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_mix28,resize=size)
                savePNG(path_png+'/map_normal_diffuse7/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse7,resize=size)
                savePNG(path_png+'/map_normal_diffuse28/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_normal_diffuse28,resize=size)
                savePNG(path_png+'/map_mask/'+personName+'_'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_map_mask,resize=size)
                
                np.savetxt(path_png+'/lightinfo/'+personName+'_'+path_view +
                        f'_{str(index).zfill(4)}_{str(count).zfill(4)}.txt', lightinfo)
                count = count + 1
        

def regroupByType(path_gene_object, path_all, path_files):
    files=['image_mix','image_diffuse','image_mask','image_light','coord3',
               'map_albedo_capture','map_albedo_mix28','map_albedo_diffuse28',
               'map_normal_mix7','map_normal_mix28','map_normal_diffuse7','map_normal_diffuse28','map_mask','lightinfo'
               ]
    list_files = os.listdir(path_all)
    num_val = int(len(list_files)*0.1)
    index_permu = np.random.permutation(len(list_files))
    index_test = index_permu[0:num_val]
    index_val = index_permu[num_val:2*num_val]
    index_train = index_permu[2*num_val:]
    print('len(index_permu)',len(index_permu))
    print('len(index_test)',len(index_test))
    print('len(index_val)',len(index_val))
    print('len(index_train)',len(index_train))

    for i in tqdm(range(len(files))):
        list_files = sorted(os.listdir(os.path.join(path_files, files[i])))

        for k_val in range(len(index_test)):
            index = index_test[k_val]
            path_file_in = os.path.join(path_files, files[i], list_files[index])
            path_out = os.path.join(path_files.replace('all', 'test'), files[i])
            groupfiles(path_file_in, path_out)

        for k_val in range(len(index_val)):
            index = index_val[k_val]
            path_file_in = os.path.join(path_files, files[i], list_files[index])
            path_out = os.path.join(path_files.replace('all', 'val'), files[i])
            groupfiles(path_file_in, path_out)

        for k_train in range(len(index_train)):
            index = index_train[k_train]
            path_file_in = os.path.join(path_files, files[i], list_files[index])
            path_out = os.path.join(path_files.replace('all', 'train'), files[i])
            groupfiles(path_file_in, path_out)
    
        print('file %s done!'%files[i])
        
def cropSavefiles(image, size, step, row, col):
    crop_img = image[row*step:row*step+size, col*step:col*step+size]
    return crop_img


def copyfiles(path_source, path_target):
    if not os.path.exists(path_target):
        os.makedirs(path_target)
    if os.path.exists(path_source):
        for root, dirs, files in os.walk(path_source):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, path_target)


def groupfiles(path_source_file, path_target):
    if not os.path.exists(path_target):
        os.makedirs(path_target)
    shutil.copy(path_source_file, path_target)


def renamefiles(path_source):
    filelist = os.listdir(path_source)  # 获取文件路径
    total_num = len(filelist)  # 获取文件长度（个数）
    i = 0  # 表示文件的命名是从1开始的
    suffix = '.png'
    for item in filelist:
        if item.endswith(suffix):  # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
            src = os.path.join(os.path.abspath(path_source), item)
            dst = os.path.join(os.path.abspath(path_source),
                               str(i).zfill(5) + suffix)
            try:
                os.rename(src, dst)
                # print ('converting %s to %s ...' % (src, dst))
                i = i + 1
            except:
                continue

        if item.endswith('.txt'):
            src = os.path.join(os.path.abspath(path_source), item)
            dst = os.path.join(os.path.abspath(path_source),
                               str(i).zfill(5) + '.txt')
            try:
                os.rename(src, dst)
                # print ('converting %s to %s ...' % (src, dst))
                i = i + 1
            except:
                continue
    # print ('total %d to rename & converted %d jpgs' % (total_num, i))


def photometric_stereo_normal(images_array, mask_array, light_matrix, threshold=0.25):
    gray_ = [0.299, 0.587, 0.114]
    mask_array = np.asarray(mask_array)
    imgs_diffuse_gray = np.dot(images_array, gray_)
    map_mask_gray = np.dot(mask_array, gray_)
    
    shap = mask_array.shape
    shaper = (shap[0], shap[1], 3)

    normal_map = np.zeros(shaper)
    ivec = np.zeros(len(images_array))

    for (xT, value) in np.ndenumerate(map_mask_gray):
        if(value > threshold):
            for (pos, image) in enumerate(imgs_diffuse_gray):
                ivec[pos] = image[xT[0], xT[1]]

            (normal, res, rank, s) = linalg.lstsq(light_matrix, ivec)

            normal = normal/linalg.norm(normal)

            if not np.isnan(np.sum(normal)):
                normal_map[xT] = normal
    
    return normal_map

def photometric_stereo_diffuse(images_array, mask_array, normal_map, light_matrix, threshold=0.25):
    # light_matrix = np.array(light_matrix)
    gray_ = [0.299, 0.587, 0.114]
    map_mask_gray = np.dot(mask_array, gray_)
    shap = map_mask_gray.shape
    shaper = (shap[0], shap[1], 3)

    albedo_map = np.zeros(shaper)
    
    new_images_array = images_array
    
    new_images_array = []
    for k1 in range(len(images_array)):
        new_image = images_array[k1]
        new_images_array.append(new_image)
        
    
    ivec = np.zeros((len(new_images_array), 3))

    for (xT, value) in np.ndenumerate(map_mask_gray):
        if(value > threshold):
            for (pos, image) in enumerate(new_images_array):
                ivec[pos] = image[xT[0], xT[1]]# shape(n*3),n为张数
                
            # print(xT)
            # print('light_matrix.shape',light_matrix.shape)
            # print('normal_map.shape',(normal_map).shape)
            # print('normal_map[xT].shape',(normal_map[xT]).shape)
            i_t = np.dot(light_matrix, normal_map[xT])#指定点的点乘 shape（n*1)
            # print('i_t.shape',i_t.shape)
            
            k = np.dot(np.transpose(ivec), i_t)/(np.dot(i_t, i_t))#输出（3,1）

            if not np.isnan(np.sum(k)):
                albedo_map[xT] = k
    
    return albedo_map


def readTiff(srcPath, resize=None, gamma=1):
    img = tiff.imread(srcPath)
    if resize is not None and img.shape[0] > resize:
        img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_CUBIC)
    img = np.array(img /255, dtype='float32')
    if gamma !=1:
        # img = np.power(img, gamma)  # gamma 矫正 
        img = srgbToLinear(img)     # 线性矫正
    return img


def readPNG(srcPath, resize=None, gamma=1):
    # img = imageio.imread(srcPath)
    img = Image.open(srcPath)
    if resize is not None and img.size[0] > resize:
        img = img.resize((resize, resize), Image.LANCZOS)
        # print('resize')
    img = np.array(img)
    img = (img / 255).astype(np.float32)
    if gamma !=1:
        # img = np.power(img, gamma)  # gamma 矫正
        img = srgbToLinear(img)     # 线性矫正
    return img

def savePNG(srcPath, img, fileName=None, isName=False, resize=None):
    width = img.shape[1]
    img=Image.fromarray((img*255).astype('uint8')).convert('RGB')
    if resize is not None and width != resize:
        img=img.resize((resize,resize),Image.LANCZOS)
    
    if isName:
        strs = fileName + ' by Beny'
        label_x = int(width/3)
        label_y = 20
        label_size = 60
        font = ImageFont.truetype("consola.ttf", label_size, encoding="unic")#设置字体 
        draw=ImageDraw.Draw(img)
        draw.text((label_x,label_y),strs,(255,255,0),font=font)
        img.save(srcPath)
        del draw
    else:
        img.save(srcPath)


def srgbToLinear(img):
    ## [0,1]
    res = np.empty_like(img)
    c = img<0.04045
    res[c] = img[c]/12.92
    res[~c] = ((img[~c]+0.055)/1.055)**2.4
    return res


def linearToSrgb(img):
    ## [0,1]
    res = np.empty_like(img)
    c = img<0.0031308
    res[c] = img[c]*12.92
    res[~c] = (1.055*(img[~c]**(1/2.4)))-0.055
    return res


# def coordinateImage(img):
#     height, width = img.shape[0], img.shape[1]
    
#     x = int(height)  # should be 1080, but we crop the center part of it
#     y = int(width)  # ratio because we might crop later on
#     i = (2*torch.arange(y, dtype=torch.float)/y - 1)[None, :, None].expand((1, -1, x))
#     j = (2*torch.arange(x, dtype=torch.float)/x - 1)[None, None, :].expand((1, y, -1))
#     coords = torch.cat((i, j), dim=0)
    
#     return coords
    
def coordinateImage(img):
    height, width = img.shape[0], img.shape[1]
    
    y = int(height)  # ratio because we might crop later on
    x = int(width)  # should be 1080, but we crop the center part of it
    i = (2*torch.arange(y, dtype=torch.float)/y - 1)[None, :, None].expand((1, -1, x))
    j = (2*torch.arange(x, dtype=torch.float)/x - 1)[None, None, :].expand((1, y, -1))
    print('c,h,w')
    coords = torch.cat((i, j), dim=0).permute(1,2,0)
    return coords.numpy()

def coordinateImage_tensor(img):
    height, width = img.shape[0], img.shape[1]
    
    y = int(height)  # ratio because we might crop later on
    x = int(width)  # should be 1080, but we crop the center part of it
    i = (2*torch.arange(y, dtype=torch.float)/y - 1)[None, :, None].expand((1, -1, x))
    j = (2*torch.arange(x, dtype=torch.float)/x - 1)[None, None, :].expand((1, y, -1))
    print('')
    coords = torch.cat((i, j), dim=0)
    return coords

if __name__ == '__main__':
    # img = np.arange(12).reshape(3,4,1)
    img = np.random.random((4,4,3))
    coord = coordinateImage(img)
    print('coord', coord)
    print('coord', coord.shape)