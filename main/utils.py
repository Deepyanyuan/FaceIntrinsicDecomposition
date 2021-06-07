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
from PIL import Image


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
        # print('img.shape',img.shape)
        if img.shape[-1] == 3:
            img=Image.fromarray((img*255).astype('uint8')).convert('RGB')
        elif img.shape[-1] == 1:
            img=Image.fromarray((img[:,:,0]*255).astype('uint8')).convert('L')
        if img.size[0] != resize:
            # img=img.resize((resize,resize),Image.LANCZOS)
            img=img.resize((resize,resize))
            # print('recover')
        # img = img.resize((resize, resize), Image.LANCZOS)
        img.save(os.path.join(out_fold, prefix + '%05d' %
                                 (i + offset) + suffix + ext))

    # for i, img in enumerate(imgs):
    #     if 'depth' in suffix:
    #         im_out = np.uint16(img[..., ::-1] * 65535.)
    #     else:
    #         im_out = np.uint8(img[..., ::-1] * 255.)
    #     im_out = cv2.resize(im_out, (resize, resize))
    #     cv2.imwrite(os.path.join(out_fold, prefix + '%05d' %
    #                              (i + offset) + suffix + ext), im_out)


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