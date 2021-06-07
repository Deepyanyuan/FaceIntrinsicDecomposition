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


def setup_runtime(args):
    '''Load configs, initialize CUDA, CuDNN and the random seeds.'''

    # Setup CUDA
    cuda_device_id = args.gpu
    if cuda_device_id is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Setup random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    cfgs['config'] = args.config
    cfgs['seed'] = args.seed
    cfgs['num_workers'] = args.num_workers
    cfgs['device'] = 'cuda:0' if torch.cuda.is_available(
    ) and cuda_device_id is not None else 'cpu'

    print(
        f'Environment: GPU {cuda_device_id} seed {args.seed} number of workers {args.num_workers}')
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


def clean_checkpoint(checkpoint_dir, keep_num=2):
    '''清理多余的cp'''
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f'Deleting obslete checkpoint file {name}')
                os.remove(name)


def archive_code(arc_path, filetypes=['.py', '.yml']):
    ''' 将代码建立文档'''
    print(f'Archiving code to {arc_path}')
    xmkdir(os.path.dirname(arc_path))
    zipf = zipfile.ZipFile(arc_path, 'w', zipfile.ZIP_DEFLATED)
    cur_dir = os.getcwd()
    flist = []
    for ftype in filetypes:
        flist.extend(glob.glob(os.path.join(
            cur_dir, '**', '*' + ftype), recursive=True))
    [zipf.write(f, arcname=f.replace(cur_dir, 'archived_code', 1))
     for f in flist]
    zipf.close()


def get_model_device(model):
    return next(model.parameters()).device


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# 暂时无用
def draw_bbox(im, size):
    b, c, h, w = im.shape
    h2, w2 = (h - size) // 2, (w - size) // 2
    marker = np.tile(np.array([[1.], [0.], [0.]]), (1, size))
    marker = torch.FloatTensor(marker)
    im[:, :, h2, w2:w2 + size] = marker
    im[:, :, h2 + size, w2:w2 + size] = marker
    im[:, :, h2:h2 + size, w2] = marker
    im[:, :, h2:h2 + size, w2 + size] = marker
    return im


def save_videos(out_fold, imgs, prefix='', suffix='', sep_folder=True, ext='.mp4', cycle=False):
    '''将结果保存成视频'''
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(
        out_fold, prefix + '*' + suffix + ext))) + 1

    imgs = imgs.transpose(0, 1, 3, 4, 2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')
        vid = cv2.VideoWriter(os.path.join(out_fold, prefix + '%05d' % (i + offset) + suffix + ext), fourcc, 5,
                              (fs.shape[2], fs.shape[1]))
        [vid.write(np.uint8(f[..., ::-1] * 255.)) for f in fs]
        vid.release()


def save_images(out_fold, imgs, resize, prefix='', suffix='', sep_folder=True, ext='.png'):
    '''保存结果成图像'''
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(
        out_fold, prefix + '*' + suffix + ext))) + 1

    if len(imgs.shape) == 4:
        imgs = imgs.transpose(0, 2, 3, 1)
    elif len(imgs.shape) == 3:
        imgs = imgs[:, :, :, np.newaxis]

    # imgs = imgs.transpose(0, 2, 3, 1)
    for i, img in enumerate(imgs):
        if 'depth' in suffix:
            im_out = np.uint16(img[..., ::-1] * 65535.)
        else:
            im_out = np.uint8(img[..., ::-1] * 255.)
        im_out = cv2.resize(im_out, (resize, resize))
        cv2.imwrite(os.path.join(out_fold, prefix + '%05d' %
                                 (i + offset) + suffix + ext), im_out)


def save_txt(out_fold, data, prefix='', suffix='', sep_folder=True, ext='.txt'):
    '''保存TXT文件'''
    if sep_folder:
        out_fold = os.path.join(out_fold, suffix)
    xmkdir(out_fold)
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''
    offset = len(glob.glob(os.path.join(
        out_fold, prefix + '*' + suffix + ext))) + 1

    [np.savetxt(os.path.join(out_fold, prefix + '%05d' % (i + offset) + suffix + ext), d, fmt='%.6f', delimiter=', ')
     for i, d in enumerate(data)]


# 源代码,针对(b,h,w)的深度图像
# def compute_sc_inv_err(d_pred, d_gt, mask=None):
#     b = d_pred.size(0)
#     diff = d_pred - d_gt
#     if mask is not None:
#         diff = diff * mask
#         avg = diff.view(b, -1).sum(1) / (mask.view(b, -1).sum(1))
#         score = (diff - avg.view(b,1,1))**2 * mask
#     else:
#         avg = diff.view(b, -1).mean(1)
#         score = (diff - avg.view(b,1,1))**2
#     return score  # masked error maps


def compute_sc_inv_err(color_pred, color_gt, mask=None):
    '''修改代码,针对(b,c,h,w)的三通道图像'''
    b, c, h, w = color_pred.shape
    diff = color_pred - color_gt
    if mask is not None:
        diff = diff * mask
        avg = diff.view(b, c, -1).sum(2) / (mask.view(b, c, -1).sum(2))
        score = (diff - avg.view(b, c, 1, 1)) ** 2 * mask
    else:
        avg = diff.view(b, c, -1).mean(2)
        score = (diff - avg.view(b, c, 1, 1)) ** 2
    return score  # masked error maps


def compute_angular_distance(n1, n2, mask=None):
    dist = (n1 * n2).sum(3).clamp(-1, 1).acos() / np.pi * 180
    return dist * mask if mask is not None else dist


def save_scores(out_path, scores, header=''):
    print('Saving scores to %s' % out_path)
    np.savetxt(out_path, scores, fmt='%.8f', delimiter=',\t', header=header)


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
        path_view = 'view_' + str(path_views[v])
        path_png = root_path_object + '/Show/image_' + path_view
        path_npy = root_path_object + '/Show/npy_' + path_view
        path_video = root_path_object + '/Show/video_' + path_view
        xmkdir(path_npy)
        xmkdir(path_video)
        folders_list = ['img_single','img_diffuse','img_residue',
        'render_shade','render_rhodt_L','render_rhodt_V','render_irradiance','render_mix_irradiance',
        'render_specular','render_sscatter','render_subscatter',
        'render_diffuse','render_diffimage_direct','render_diffimage_direct_1','render_diffimage','render_diffimage_1',
        'render_singleimage_direct','render_singleimage_direct_1','render_singleimage','render_singleimage_1']
        for i in range (len(folders_list)):
            xmkdir(path_png + '/'+ folders_list[i])


def mergeSubPath(root_path_object, path_view, path_object_sub, resize=None, isSave=False):
    '''合并指定目录下的所有图片'''
    files = glob.glob(root_path_object + '/Source/' +
                      path_view + '/' + path_object_sub + '/*.png')
    imgs = []
    for file in files:
        img = cv2.imread(file)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img.shape[0] == img.shape[1], 'The image size must be [m,m,c]'
        if resize is not None and resize > img.shape[0]:
            img = cv2.resize(img, (resize, resize),
                             interpolation=cv2.INTER_CUBIC)      # 测试使用
        img = np.float32(img / 255.)
        imgs.append(img)
    allImages = np.float32(np.array(imgs))

    if isSave:
        cv2.imwrite(root_path_object + '/Show/image_' + path_view + '/' + path_object_sub + '.png',
                    rgb2bgr(allImages.mean(0) *5* 255))

    np.save(root_path_object + '/Show/npy_' + path_view + '/' +
            path_object_sub + '.npy', allImages.sum(0))  # [h,w,c]
    # np.save(root_path_object + '/npy_' + path_view + '/' + path_object_sub + '_all.npy', allImages)  # [num_lights,h,w,c]


def moveSubPath(root_path_object, path_object_sub, resize=None, isSave=False):
    '''转移指定目录下的所有图片'''
    files = glob.glob(root_path_object + '/Source/' +
                      path_object_sub + '/*.png')
    # print('files',files)

    for file in files:
        # 分成路径和文件的二元元组
        front_path_object_sub = os.path.split(file)
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img.shape[0] == img.shape[1], 'The image size must be [m,m,c]'
        if resize is not None and resize > img.shape[0]:
            img = cv2.resize(img, (resize, resize),
                             interpolation=cv2.INTER_CUBIC)      # 测试使用
        img = np.float32(img / 255.)

        if isSave:
            cv2.imwrite(root_path_object + '/Show/image/' + front_path_object_sub[1],
                        rgb2bgr(img/np.max(img) * 255))
        np.save(root_path_object + '/Show/npy/' + front_path_object_sub[1].replace('.png', '.npy'),
                img)  # 保存法线图,遮罩图,前点交叉偏振图和前点平行偏振图


def splitSingleDiffuseSpecular_png(file_cross, file_parallel, resize=None):
    # 分离高光和法线
    img_1 = cv2.imread(file_cross)
    try:
        img_1.shape
        # print('img_1.shape',img_1.shape)
    except:
        print('Can not read the image')

    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    assert img_1.shape[0] == img_1.shape[1], 'image size must be [m,m,c]'
    if resize is not None and resize > img_1.shape[0]:
        img_1 = cv2.resize(img_1, (resize, resize),
                           interpolation=cv2.INTER_CUBIC)      # 测试使用
        print('resize done!')

    img_1 = np.float32(img_1 / 255.)

    img_2 = cv2.imread(file_parallel)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    assert img_1.shape[0] == img_1.shape[1], 'image size must be [m,m,c]'
    if resize is not None and resize > img_2.shape[0]:
        img_2 = cv2.resize(img_2, (resize, resize),
                           interpolation=cv2.INTER_CUBIC)      # 测试使用
    img_2 = np.float32(img_2 / 255.)

    # 单个图像分离不做归一化
    img_d_s = img_2 + img_1
    img_d = img_1 *2
    img_s = np.clip(img_2 - img_1, 0, 1)
    
    

    # 归一化
    # img_d_s = img_2
    # img_d = img_1/np.max(img_2)
    # img_s = np.clip((img_2-img_1)/np.max(img_d_s),0,1)

    return img_d, img_s, img_d_s


def splitDiffuseSpecular(path_obtain_npy, file_cross, file_parallel):
    # 分离高光和法线
    # 公式参考 搭建-04 光度法线的计算  P10
    img_1 = np.load(path_obtain_npy + '/' + file_cross)
    img_2 = np.load(path_obtain_npy + '/' + file_parallel)

    # print('shape',img_1.shape,img_2.shape)
    # 没有归一化
    img_d_s = img_2 + img_1
    img_d = img_1 *2
    # img_s = np.clip((img_2-img_1),0,1)

    # 梯度图像归一化后,法线显示效果好
    # img_d_s = img_2
    # img_d = img_1 / np.max(img_2)
    # img_s = (img_2 - img_1) / np.max(img_2)
    img_s = np.clip((img_2 - img_1) / np.max(img_d_s), 0, 1)

    # img_d_s = img_2
    # img_d = img_1 / np.max(img_1)
    # img_s = (img_2 - img_1) / np.max(img_2 - img_1)

    return img_d, img_s, img_d_s


def threshold_mask(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, threshold, 1.0, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    dst = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return dst


def normalImage(img):
    temp = np.linalg.norm(img, ord=2, axis=-1, keepdims=False)
    temp = cv2.merge([temp,temp,temp])
    val = np.divide(img, temp, out=np.zeros_like(img), where=temp!=0)
    return val
    
    # r = img[:, :, 0]
    # g = img[:, :, 1]
    # b = img[:, :, 2]

    # normal = np.sqrt(b * b + g * g + r * r)
    # r = np.divide(r, normal, out=np.zeros_like(r), where=normal != 0)
    # g = np.divide(g, normal, out=np.zeros_like(g), where=normal != 0)
    # b = np.divide(b, normal, out=np.zeros_like(b), where=normal != 0)
    # return cv2.merge([r, g, b])


def getViewDirImage(img, viewDir):
    img_vd = np.ones_like(img)
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


# def phong2(x, shininess1, ks1, shininess2, ks2):
#     # 用以拟合的函数
#     # val = ks1 * np.float_power(np.cos(x), shininess1) + ks2 * np.float_power(np.cos(x), shininess2)
#     val = 0.47*ks1 * np.power(np.cos(x), shininess1) + 0.53*ks2 * \
#         np.power(np.cos(x), shininess2)
#     # val = val / 2
#     return val

#! by beny
def blinnPhong(x, shininess1):
    # 用以拟合的函数
    val = (math.e + 2)/(math.pi * 2) * np.power(np.cos(x), shininess1)
    return val

def blinnPhong2(x, shininess1, shininess2):
    # 用以拟合的函数
    alpha=0.4
    val = (1-alpha) * (math.e + 2)/(math.pi * 2) * np.power(np.cos(x), shininess1) + alpha * (math.e + 2)/(math.pi * 2) * np.power(np.cos(x), shininess2)
    # val = ((math.e + 2)/(math.pi * 2) * np.power(np.cos(x), shininess1) + (math.e + 2)/(math.pi * 2) * np.power(np.cos(x), shininess2)) /2
    return val

def blinnPhong_2(cos_x, shininess1, shininess2):
    # 用以拟合的函数
    alpha=0.4
    val = (1-alpha) * (math.e + 2)/(math.pi * 2) * np.power(cos_x, shininess1) + alpha * (math.e + 2)/(math.pi * 2) * np.power(cos_x, shininess2)
    # val = ((math.e + 2)/(math.pi * 2) * np.power(cos_x, shininess1) + (math.e + 2)/(math.pi * 2) * np.power(cos_x, shininess2)) /2
    return val


def HG_fit(x, g):
    val = x * (1 - g ** 2) / (np.power(1+g**2-2*g, 1.5) * math.pi * 4)
    return val


def HG_SscatterPhase(g, cos_angle=1):
    numerator = 1 - g ** 2
    denominator = np.power(1+g**2-2*g*cos_angle, 1.5) * math.pi * 4
    val = numerator / denominator
    # val = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    return val


def HG_SscatterTerm(normal, k1, k2):
    term_third = np.dot(normal,k1) + np.dot(normal,k2)
    term_third = np.where(term_third>0,term_third,0)
    term = np.divide(1, term_third, out=np.zeros_like(term_third),where=term_third!=0.)
    return term


def gaussian(x, x0, xalpha, A):
    return A * np.exp(-((x - x0) / xalpha) ** 2)


def phong(x, shininess, ks):
    return ks * np.float_power(np.cos(x), shininess)


def pre_processing(root_path_object, file_front_residue_path, file_full_residue_sum_path, file_full_diffuse_sum_path,
                   isSave=False):
    '''001:数据预处理, 计算 前照明-差值图 和 全照明-差值图'''
    img_front_difference = np.load(
        root_path_object + '/Show/npy/' + file_front_residue_path)  # [h,w,c]
    # img_full_difference = np.load(root_path_object+'/npy/'+file_full_residue_mean_path)    # [h,w,c]
    img_full_difference = np.load(
        root_path_object + '/Show/npy/' + file_full_residue_sum_path)  # [h,w,c]
    img_full_cross = np.load(
        root_path_object + '/Show/npy/' + file_full_diffuse_sum_path)  # [h,w,c]

    img_front_difference_gray = rgb2gray3(np.float32(img_front_difference))
    img_full_difference_gray = rgb2gray3(np.float32(img_full_difference))

    # 返回灰度化的三通道偏振差图像和三通道交叉偏振图像
    return img_front_difference_gray, img_full_difference_gray, img_full_cross


def get_anglemask(root_path_object, path_view, viewDir, max_angle, dark_thr, img_normal, img_front, img_mask, isSave=False):
    '''002:计算掩模, 根据角度以及亮度, 获得参与第一次拟合的像素区域'''
    # 观察方向
    viewDir = viewDir / np.linalg.norm(viewDir)
    minAngle = 0

    # 整理法线
    height, width, channel = img_normal.shape
    # img_normal = normalImage(img_normal)          # 已经做过归一化处理
    normal = img_normal.flatten().reshape(height * width, 3)
    front = img_front[:, :, 0].flatten().reshape(height * width)
    mask_regional = img_mask[:, :, 0].flatten().reshape(height * width)

    # 计算夹角
    angle = vg.angle(normal, viewDir, assume_normalized=True, units='rad')  # 输入向量全部做了单位化,输出弧度角
    mask_angle = np.where((angle < max_angle), 1, 0)  # 根据条件计算模板

    front_shif = front - dark_thr * (1 - angle / max_angle)
    mask_brightness = np.where((front_shif > 0), 1, 0)  # 根据条件计算模板

    img_angle = np.tile(angle.reshape(height, width, 1), reps=(1, 1, 3))  # beny
    img_mask_final = np.tile((mask_angle * mask_brightness * mask_regional).reshape(height, width, 1),
                             reps=(1, 1, 3))  # beny

    # 保存图像和数据
    if isSave:
        img_angle_bgr = rgb2bgr(np.float32(img_angle))
        img_mask_final_bgr = rgb2bgr(np.float32(img_mask_final))
        cv2.imwrite(root_path_object + '/Show/image_' + path_view + '/' + 'Cal_Angle.png',
                    img_angle_bgr / (math.pi / 2) / np.max(img_angle_bgr / (math.pi / 2)) * 255)
        cv2.imwrite(root_path_object + '/Show/image_' + path_view + '/' + 'Mask_Angle.png',
                    img_mask_final_bgr / np.max(img_mask_final_bgr) * 255)

    np.save(root_path_object + '/Show/npy_' + path_view + '/' + 'Cal_Angle.npy', img_angle)
    np.save(root_path_object + '/Show/npy_' + path_view +
            '/' + 'Mask_Angle.npy', img_mask_final)
    return img_angle, img_mask_final


def fit_region(r0, img_mask_final, img_angle, img_front, img_specular, Fit_Counter):
    '''003:数据拟合, 计算区域高光反射系数'''
    # 准备数据
    indexs = np.where(img_mask_final[:, :, 0].flatten() == 1)  # beny
    tem = img_specular[:, :, 0]
    c = tem.flatten()[indexs]
    c = np.where((c > 0.1), c, 0.1)  # 根据条件计算模板
    # data_hk=img_angle.flatten()[indexs]
    data_hk = img_angle[:, :, 0].flatten()[indexs]  # beny
    data_i = img_front[:, :, 0].flatten()[indexs]  # 源代码，三通道偏振差图像
    data_i = func_specular(data_i, data_hk, c, r0)

    # 开始拟合函数
    xx = data_hk  # 主要序号， angle是x轴
    yy = data_i

    # 曲线拟合
    gmodel = Model(blinnPhong)
    result = gmodel.fit(yy, x=xx, shininess1=10, method='least_squares',
                        fit_kws={'loss': 'huber'})
    
    # gmodel = Model(blinnPhong2)
    # result = gmodel.fit(yy, x=xx, shininess1=10, shininess2=11, method='least_squares',
    #                     fit_kws={'loss': 'huber'})

    # 绘制并展示拟合曲线
    show = True

    c1 = np.ones_like(c)
    data_ic1 = img_front[:, :, 0].flatten()[indexs]
    data_ic1 = func_specular(data_ic1, data_hk, c1, r0)
    min = np.min(xx)
    max = np.max(xx)
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
    shininess1 = result.best_values["shininess1"]
    # shininess2 = result.best_values["shininess2"]
    # alpha = result.best_values["alpha"]

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

    fig.add_trace(go.Scatter(x=xx_fit, y=blinnPhong(xx_fit, shininess1), name="拟合数据", mode='markers',
                             marker=dict(color="#FF0000", size=5)))
    
    # fig.add_trace(go.Scatter(x=xx_fit, y=blinnPhong2(xx_fit, shininess1, shininess2), name="拟合数据", mode='markers',
    #                          marker=dict(color="#FF0000", size=5)))

    # p.circle(y=phong2(xx_fit,shininess1,ks1,shininess2,ks2, alpha), x=xx_fit,color="red")

    # p.line(y=yy_mean, x=xx_mean, color="blue",line_width=3)

    plotly.offline.plot(fig, filename="Html/SP_hk_i_" +
                        "{0}".format(Fit_Counter) + ".html", auto_open=show)

    with open("Html/SP_fit_" + "{0}".format(Fit_Counter) + ".txt", "w") as f:
        f.write(result.fit_report())

    if show:
        plt.show()

    return result.best_values


def fit_region_ss(img_mask_final, img_front, img_sscatter, Fit_Counter):
    '''003:数据拟合, 计算区域单次散射相位函数系数g'''
    # 准备数据
    indexs = np.where(img_mask_final[:, :, 0].flatten() == 1)  # beny
    data_hk = img_sscatter[:, :, 0].flatten()[indexs]  # beny
    data_i = img_front[:, :, 0].flatten()[indexs]  # 源代码，三通道偏振差图像

    # 开始拟合函数
    xx = data_hk  # 主要序号， angle是x轴
    yy = data_i

    # 曲线拟合
    gmodel = Model(HG_fit)
    # result = gmodel.fit(yy, x=xx, g=0., method='least_squares', fit_kws={'loss': 'huber'})
    result = gmodel.fit(yy, x=xx, g=0.3)

    # 绘制并展示拟合曲线
    show = True

    min = np.min(xx)
    max = np.max(xx)
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
    g = result.best_values["g"]

    # 显示和保存计算结果
    print(">> 开始【拟合p(g)曲线 求g】...")
    print(result.fit_report(), "\n\n")

    fig = fig = go.Figure()

    fig.add_trace(go.Scatter(x=data_hk, y=data_i, name="拟合后数据",
                             mode='markers', marker=dict(color="#001400", size=7)))
    fig.add_trace(
        go.Scatter(x=xx_mean, y=yy_mean, name="平均数据", mode='lines+markers', marker=dict(color="#FF00FF", size=10)))

    fig.add_trace(go.Scatter(x=xx_fit, y=HG_fit(xx_fit, g), name="拟合数据", mode='markers', marker=dict(color="#FF0000", size=5)))

    # p.circle(y=phong2(xx_fit,shininess1,ks1,shininess2,ks2), x=xx_fit,color="red")

    # p.line(y=yy_mean, x=xx_mean, color="blue",line_width=3)

    plotly.offline.plot(fig, filename="Html/SS_hk_i_" +
                        "{0}".format(Fit_Counter) + ".html", auto_open=show)

    with open("Html/SS_fit_" + "{0}".format(Fit_Counter) + ".txt", "w") as f:
        f.write(result.fit_report())

    if show:
        plt.show()

    return result.best_values


def calc_speclar(root_path_object, path_view, img_full, img_angle, img_mask, result, isSave=False):
    '''004:计算高光反照率 C Specular Albedo'''
    s1 = result['Fit_First']['shininess1']
    # s2 = result['Fit_First']['shininess2']
    dictIntegrate = np.load('Dict_Integration.npy', allow_pickle=True).item()
    # n = img_normal
    c_hs = img_full[:, :, 0]  # 原始代码，三通道偏振差图像
    # print('np.max(c_hs)',np.max(c_hs))      # np.max(c_hs)=13
    # c_hs = c_hs/np.max(c_hs)
    # c_hs = c_hs / 40

    theta = img_angle[:, :, 0]
    c1 = np.zeros_like(c_hs)
    for i in range(img_mask.shape[0]):
        for j in range(img_mask.shape[1]):
            if (img_mask[i, j][0] == 1):

                if (theta[i, j] > 0 and int(theta[i, j] / math.pi * 180) < 89):
                    key1 = '{:.1f}'.format(s1) + '_' + '{:.0f}'.format(theta[i, j] / math.pi * 180)
                    # key2 = '{:.1f}'.format(s2) + '_' + '{:.0f}'.format(theta[i, j] / math.pi * 180)
                    pre_integrate1 = dictIntegrate[key1]
                    # pre_integrate2 = dictIntegrate[key2]
                    # c1[i, j] = (1-0.4) * pre_integrate1 + 0.4 * pre_integrate2
                    c1[i, j] = pre_integrate1
                    # c1[i, j] = c1[i, j] *2

    c = np.divide(c_hs, c1, out=np.zeros_like(c_hs), where=c1 != 0)
    # c = c / 2
    # c = np.clip(c, 0, 1)
    c_rgb = np.float32(cv2.merge([c, c, c]))
    c1_rgb = np.float32(cv2.merge([c1, c1, c1]))
    # # 保存数据
    # if isSave:
    #     c_bgr = rgb2bgr(c_rgb)
    #     c1_bgr = rgb2bgr(c1_rgb)
    #     cv2.imwrite(root_path_object + '/Show/image_' + path_view + '/' + 'albedo_speclar_C.png', c_bgr * 255)
    #     cv2.imwrite(root_path_object + '/Show/image_' + path_view + '/' + 'albedo_speclar_C_max.png', c_bgr / c_bgr.max() * 255)
    #     cv2.imwrite(root_path_object + '/Show/image_' + path_view + '/' + 'albedo_speclar_C_clip.png', np.clip(c_bgr, 0, 1) * 255)
    #     cv2.imwrite(root_path_object + '/Show/image_' + path_view + '/' + 'albedo_speclar_C_1.png', c1_bgr * 255)
    #     cv2.imwrite(root_path_object + '/Show/image_' + path_view + '/' + 'albedo_speclar_C_1_max.png', c1_bgr / c1_bgr.max() * 255)
    return c_rgb

# @jit(nopython=True)


def computeRhodtMatrix(k1, normal, rho_dt):
    # GPU Germs 3 p255,size [height,width,channel]
    assert len(
        normal.shape) == 3, 'the size of normal must be [height,width,channel]'
    # k1 = np.float32(k1.reshape(3,1))
    n = normal
    # ndotL = (np.dot(n, k1))  # [h,w]
    # print('n.shape',n.shape)
    # print('k1.shape', k1.shape)
    ndotL = np.dot(n, k1).clip(0, 1)  # [h,w]
    # ndotL = np.dot(n, k1).squeeze().clip(0, 1)  # [h,w]
    # ndotL = np.matmul(n, k1).squeeze().clip(0, 1)  # [h,w]
    # print('ndotL.shape',ndotL.shape)
    lengh = np.shape(rho_dt)[0]
    rho_dt_matric = np.trunc(ndotL * lengh)  # 对所有元素取整
    rho_dt_matric_new = np.zeros_like(ndotL)
    for i in range(rho_dt_matric.shape[0]):
        for j in range(rho_dt_matric.shape[1]):
            if rho_dt_matric[i, j] != 0:
                val = int(rho_dt_matric[i, j])
                rho_dt_matric_new[i, j] = rho_dt[val - 1]
            else:
                rho_dt_matric_new[i, j] = 0.
    return cv2.merge([rho_dt_matric_new, rho_dt_matric_new, rho_dt_matric_new])


def computeIrradiance(k1, normal, albedo_specular, rho_dt_matric):
    assert len(
        normal.shape) == 3, 'the size of normal must be [height,width,channel]'
    n = normal
    ndotL = np.clip(np.dot(n, k1), 0, 1)  # [h,w]
    ndotL_re = cv2.merge([ndotL, ndotL, ndotL])  # [h,w,c]
    rho_dt_L = 1.0 - albedo_specular * rho_dt_matric
    rho_dt_L = np.clip(rho_dt_L, 0, 1)
    irradiance = ndotL_re * rho_dt_L
    return irradiance


def computeRhodt(costheta, r0, s1, ks1, s2, ks2, numterms=80):
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
            srval = specularIntegralFunction_s(L, V, N, r0, s1, ks1, s2, ks2)
            temp = srval * sint
            localsum = localsum + temp
        sum = sum + localsum * (pi / 2.0) / numterms
    rho_dt = sum * (2.0 * pi) / numterms
    return rho_dt


def specularIntegralFunction_s(k1, k2, normal, r0, s1, ks1, s2, ks2):
    '''  '''
    assert len(normal.shape) == 1, 'the shape of normal is must [channel,None]'
    n = normal
    ndotL = np.dot(normal, k1)
    if ndotL > 0:
        h = np.divide(np.add(k1, k2), np.linalg.norm(np.add(k1, k2)), out=np.zeros_like(np.add(k1, k2)),
                      where=np.linalg.norm(np.add(k1, k2)) != 0)
        F = func_F(r0, k2, h)

        # # 整理法线 计算夹角
        # angle_N_H = vg.angle(n, h, assume_normalized=True, units='rad')  # 输入向量全部单位化
        # a = np.clip(angle_N_H, 0, math.pi / 2)
        #
        # p = phong2(a, s1, ks1, s2, ks2)

        # 整理法线,计算余弦夹角
        NdotH = np.dot(normal, h)
        p = phong_2(NdotH, s1, ks1, s2, ks2)

        p = p / (ks1 + ks2)
        G = func_G(n, h, k1, k2)

        rho = (p * F * G) / (np.dot(n, k1) +
                             np.dot(n, k2) - np.dot(n, k1) * np.dot(n, k2))
        rho = np.maximum(rho, 0)
        val = ndotL * rho
    else:
        val = 0.
    return val


def specularIntegralFunction(k1, k2, n, r0, s1, ks1, s2, ks2):
    '''  '''
    ndotL = np.dot(n, k1)
    ndotL = np.where(ndotL > 0, ndotL, 0)

    h = np.divide(np.add(k1, k2), np.linalg.norm(np.add(k1, k2)), out=np.zeros_like(np.add(k1, k2)),
                  where=np.linalg.norm(np.add(k1, k2)) != 0)
    # print('h',h)
    F = func_F(r0, k2, h)
    # print('F:', F)
    assert len(
        n.shape) == 3, 'the size of normal must be [height,width,channel]'
    height, width, channel = n.shape

    # 整理法线 计算余弦夹角
    n_re = n.reshape(height * width, 3)
    NdotH = np.dot(n_re, h).clip(0, 1).reshape(height, width)
    p = phong_2(NdotH, s1, ks1, s2, ks2)
    # print('np.max(p):',np.max(p))
    # p = p / np.max(p)
    # p = p / (ks1 + ks2)
    # print('np.max(p):', np.max(p))
    G = func_G(n, h, k1, k2)
    # print('np.max(G):', np.max(G))

    # rho = (p * F * G) / (np.dot(n, k1) + np.dot(n, k2) - np.dot(n, k1) * np.dot(n, k2))
    rho_molecular = p * F * G
    rho_denominator = np.dot(n, k1) + np.dot(n, k2) - \
        np.dot(n, k1) * np.dot(n, k2)
    rho = np.divide(rho_molecular, rho_denominator, out=np.zeros_like(
        rho_molecular), where=rho_denominator != 0)
    # print('np.max(rho)',np.max(rho))

    val = np.where(rho > 0, rho, 0)
    val = ndotL * val
    return val


def unlitShadingRender(img_normal, img_mask, lightDirs):
    lightDirs = lightDirs[:, 1:]
    lightDirs = lightDirs / \
        np.linalg.norm(lightDirs, ord=2, axis=1, keepdims=True)
    height, width, channel = img_normal.shape
    num_lights = lightDirs.shape[0]
    # 矩阵计算
    lightDirs_T = lightDirs[:, :].T
    render_unlit_shading_T = np.dot(
        img_normal, lightDirs_T)  # [h,w,num_lights]
    render_unlit_shading_T = np.clip(render_unlit_shading_T, 0, 1)  # 去除负值
    render_unlit_shading_rgb = np.tile(render_unlit_shading_T.transpose(2, 0, 1).reshape(num_lights, height, width, -1),
                                       reps=(1, 1, 1, channel))
    unlit_mask = np.tile(img_mask.reshape(
        1, height, width, channel), reps=(num_lights, 1, 1, 1))
    render_unlit_shading = render_unlit_shading_rgb * unlit_mask

    return render_unlit_shading


def unlitSpecularRender(img_normal, img_specular, result, lightDirs, viewDir, r0, lightIntensity):
    lightDirs = lightDirs[:, 1:]
    lightDirs = lightDirs / \
        np.linalg.norm(lightDirs, ord=2, axis=1, keepdims=True)
    height, width, channel = img_normal.shape
    num_lights = lightDirs.shape[0]
    render_unlit_specular = np.zeros(
        [num_lights, height, width, channel], dtype=np.float32)  # [num_lights,h,w,c]
    s1 = result['Fit_First']['shininess1']
    s2 = result['Fit_First']['shininess1']
    ks1 = result['Fit_First']['ks1']
    ks2 = result['Fit_First']['ks2']
    n = img_normal
    c = img_specular[:, :, 0]

    k2 = viewDir
    k2 = k2 / np.linalg.norm(k2, ord=2)
    for k in range(num_lights):
        k1 = lightDirs[k, :]
        k1 = k1 / np.linalg.norm(k1, ord=2)
        # print('k',k)
        # print('lightDir:',k1)
        rho = specularIntegralFunction(k1, k2, n, r0, s1, ks1, s2, ks2)
        # rho = c * rho / rho.max()
        rho = c * rho.clip(0, 1)
        # print('np.max(rho)', np.max(rho))
        # print('np.mean(rho)', np.mean(rho))
        # print('\n')
        img_render = rho * lightIntensity  # 光照强度
        img_render_rgb = np.float32(
            cv2.merge([img_render, img_render, img_render]))

        # img_render_rgb = np.clip(img_render_rgb, 0, 1)
        # img_render_rgb = img_render_rgb / img_render_rgb.max()

        render_unlit_specular[k, :, :, :] = img_render_rgb
        # print('np.max(img_render_rgb)',np.max(img_render_rgb))

    return render_unlit_specular


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
        print('file', files)
        clip = mpe.VideoFileClip(file)
        tmpClip.append(clip)
    clipArrays.append(tmpClip)

    # 视频拼接
    # destClip = mpe.concatenate_videoclips(tmpClip)
    # destClip.write_videofile(object_path+outname)

    # 视频堆叠
    destClip_2 = mpe.clips_array(clipArrays)
    destClip_2.write_videofile(outname)


def createTrainFolders(root_path_object, path_views):
    path_npy = root_path_object + '/Show/npy'
    xmkdir(path_npy)
    list_files = ['image', 'train', 'val', 'test', 'test_all']
    for i in range(len(list_files)):
        path_all = root_path_object + '/Show/' + list_files[i]
        xmkdir(path_all + '/image_input')
        xmkdir(path_all + '/image_diff')
        xmkdir(path_all + '/diffuse')
        xmkdir(path_all + '/sscatter_c')
        xmkdir(path_all + '/specular_c')

        xmkdir(path_all + '/sscatter_g')
        xmkdir(path_all + '/specular_g')
        xmkdir(path_all + '/normal')
        xmkdir(path_all + '/mask')
        xmkdir(path_all + '/lightinfo')

    # for v in range(len(path_views)):
    #     path_view = 'view_' + str(path_views[v])
    #     path_png = root_path_object + '/Show/image_' + path_view

    #     xmkdir(path_png + '/image_input')
    #     xmkdir(path_png + '/image_diff')
    #     xmkdir(path_png + '/diffuse')
    #     xmkdir(path_png + '/sscatter_c')
    #     xmkdir(path_png + '/specular_c')

    #     xmkdir(path_png + '/sscatter_g')
    #     xmkdir(path_png + '/specular_g')
    #     xmkdir(path_png + '/normal')
    #     xmkdir(path_png + '/mask')
    #     xmkdir(path_png + '/lightinfo')


def saveTrainData(path_png, path_view, input, rs, dtL, dtV, a_d, a_ss, a_sp, norm, mask, lightinfo, index, isCrop, size):
    '''
    input:[h,w,c], rs:[h,w,c], dtL:[h,w,c], dtV:[h,w,c], a_d:[h,w,c], a_sp:[h,w,c], norm:[h,w,c], mask:[h,w,c], lightinfo:[h,w,c]
    return:[]
    '''
    image_input = rgb2bgr(input)
    image_diff = cv2.cvtColor(rs, cv2.COLOR_RGB2GRAY)
    rho_dtL = cv2.cvtColor(dtL, cv2.COLOR_RGB2GRAY)
    rho_dtV = cv2.cvtColor(dtV, cv2.COLOR_RGB2GRAY)
    diffuse = rgb2bgr(a_d)
    sscatter_g = cv2.cvtColor(a_ss, cv2.COLOR_RGB2GRAY)
    specular_g = cv2.cvtColor(a_sp, cv2.COLOR_RGB2GRAY)
    sscatter_c = rgb2bgr(cv2.merge([sscatter_g, rho_dtL, rho_dtV]))
    specular_c = rgb2bgr(cv2.merge([specular_g, rho_dtL, rho_dtV]))
    normal = rgb2bgr(norm)
    mask = rgb2bgr(mask)
    lightinfo = lightinfo.reshape(1, -1)

    if isCrop:
        step = 128
        height, width, channel = image_input.shape
        assert (height - size) % step == 0, 'The size of image must be divide by size'
        assert (width - size) % step == 0, 'The size of image must be divide by size'
        num_row = (height - size) / step + 1
        num_col = (width - size) / step + 1
        count = 0
        for row in range(int(num_row)):
            for col in range(int(num_col)):
                crop_image_input = cropSavefiles(
                    image_input, size, step, row, col)
                crop_image_diff = cropSavefiles(
                    image_diff, size, step, row, col)
                crop_diffuse = cropSavefiles(diffuse, size, step, row, col)
                crop_sscatter_g = cropSavefiles(
                    sscatter_g, size, step, row, col)
                crop_specular_g = cropSavefiles(
                    specular_g, size, step, row, col)
                crop_sscatter_c = cropSavefiles(
                    sscatter_c, size, step, row, col)
                crop_specular_c = cropSavefiles(
                    specular_c, size, step, row, col)
                crop_normal = cropSavefiles(normal, size, step, row, col)
                crop_mask = cropSavefiles(mask, size, step, row, col)

                cv2.imwrite(path_png+'/image_input/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_image_input * 255)
                cv2.imwrite(path_png+'/image_diff/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_image_diff * 255)
                cv2.imwrite(path_png+'/diffuse/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_diffuse * 255)

                cv2.imwrite(path_png+'/sscatter_g/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_sscatter_g * 255)
                cv2.imwrite(path_png+'/specular_g/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_specular_g * 255)
                cv2.imwrite(path_png+'/sscatter_c/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_sscatter_c * 255)
                cv2.imwrite(path_png+'/specular_c/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_specular_c * 255)

                cv2.imwrite(path_png+'/normal/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_normal * 255)
                cv2.imwrite(path_png+'/mask/'+path_view +
                            f'_{str(index).zfill(4)}_{str(count).zfill(4)}.png', crop_mask * 255)
                np.savetxt(path_png+'/lightinfo/'+path_view +
                           f'_{str(index).zfill(4)}_{str(count).zfill(4)}.txt', lightinfo)
                count = count + 1
    else:
        cv2.imwrite(path_png+'/image_input/'+path_view +
                    f'_{str(index).zfill(4)}.png', (image_input) * 255)
        cv2.imwrite(path_png+'/image_diff/'+path_view +
                    f'_{str(index).zfill(4)}.png', image_diff * 255)
        cv2.imwrite(path_png+'/diffuse/'+path_view +
                    f'_{str(index).zfill(4)}.png', (diffuse) * 255)

        cv2.imwrite(path_png+'/sscatter_g/'+path_view +
                    f'_{str(index).zfill(4)}.png', sscatter_g * 255)
        cv2.imwrite(path_png+'/specular_g/'+path_view +
                    f'_{str(index).zfill(4)}.png', specular_g * 255)
        cv2.imwrite(path_png+'/sscatter_c/'+path_view +
                    f'_{str(index).zfill(4)}.png', (sscatter_c) * 255)
        cv2.imwrite(path_png+'/specular_c/'+path_view +
                    f'_{str(index).zfill(4)}.png', (specular_c) * 255)

        cv2.imwrite(path_png+'/normal/'+path_view +
                    f'_{str(index).zfill(4)}.png', (normal) * 255)
        cv2.imwrite(path_png+'/mask/'+path_view +
                    f'_{str(index).zfill(4)}.png', (mask) * 255)
        np.savetxt(path_png+'/lightinfo/'+path_view +
                   f'_{str(index).zfill(4)}.txt', lightinfo)


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


def movefiles(path_source_file, path_target):
    if not os.path.exists(path_target):
        os.makedirs(path_target)
    shutil.move(path_source_file, path_target)


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
