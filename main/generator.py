# -*- encoding: utf-8 -*-
'''
@File    : generator.py
@Time    : 2020/8/16 18:04
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
'''
'''
代码描述

'''
# ----- Load libraries -------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfs

import numpy as np
import os, glob, time
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import imageio
import cv2
import shutil

from . import utils_real as utils


# ---- Custom functions -----------------------------------

class Generator():
    def __init__(self,cfgs):
        self.root_path = cfgs.get('root_path', 'Data')
        self.path_generate = cfgs.get('path_generate', 'Data')
        self.type = cfgs.get('type', 'resize')
        self.ori_size = cfgs.get('ori_size', 2048)
        self.image_size = cfgs.get('image_size', 512)
        
    def ALL_new(self):
        path_src_first = os.path.join(os.path.abspath(self.root_path), 'ori')

        path_src_first_files = os.listdir(path_src_first)
        for k1 in range(len(path_src_first_files)):
            path_src_first_file = path_src_first_files[k1]
            path_src_second = os.path.join(path_src_first, path_src_first_file)
            
            path_src_second_files = os.listdir(path_src_second)
            for k2 in range(len(path_src_second_files)):
                path_src_second_file = path_src_second_files[k2]
                path_src_third = os.path.join(path_src_second, path_src_second_file)
                print('path_src_third', path_src_third)
                
                path_src_third_files = os.listdir(path_src_third)
                for k3 in range(len(path_src_third_files)):
                    path_src_third_file = path_src_third_files[k3]
                    path_file = os.path.join(path_src_third, path_src_third_file)
                    path_dst = os.path.join(self.root_path, self.type, path_src_first_file, path_src_second_file)
                    utils.xmkdir(path_dst)
                    coef = 0.9
                    
                    if path_src_third_file.endswith('.png'):
                        if self.type == 'both':
                            # resize
                            img_resize = utils.readPNG(path_file, self.image_size)
                            path_save_resize = os.path.join(path_dst, path_src_third_file)
                            utils.savePNG(path_save_resize, img_resize)
                            
                            # crop
                            img = utils.readPNG(path_file)
                            # mask = utils.readPNG(path_file.replace(path_src_second_file, 'image_mask'))
                            mask = utils.readPNG(path_file.replace(path_src_second_file, 'map_mask'))
                            crop_step = int(self.image_size/2)
                            crop_size = self.image_size
                            
                            height, width, channel = img.shape
                            assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
                            assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
                            num_row = (height - crop_size) / crop_step + 1
                            num_col = (width - crop_size) / crop_step + 1
                            count = 0
                            mask_threshold = crop_size * crop_size * coef
                            for row in range(int(num_row)):
                                for col in range(int(num_col)):
                                    crop_mask = utils.cropSavefiles(mask, crop_size, crop_step, row, col)
                                    count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                                    if count_nonzero > mask_threshold:
                                        crop_img = utils.cropSavefiles(img, crop_size, crop_step, row, col)
                                        path_save = os.path.join(path_dst, path_src_third_file.replace('.png', f'_{str(count).zfill(4)}.png'))
                                        utils.savePNG(path_save, crop_img, resize=crop_size)
                                        count = count + 1
                        
                        elif self.type == 'crop':
                            img = utils.readPNG(path_file)
                            # mask = utils.readPNG(path_file.replace(path_src_second_file, 'image_mask'))
                            mask = utils.readPNG(path_file.replace(path_src_second_file, 'map_mask'))
                            crop_step = int(self.image_size/2)
                            crop_size = self.image_size
                            
                            height, width, channel = img.shape
                            assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
                            assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
                            num_row = (height - crop_size) / crop_step + 1
                            num_col = (width - crop_size) / crop_step + 1
                            count = 0
                            mask_threshold = crop_size * crop_size * coef
                            for row in range(int(num_row)):
                                for col in range(int(num_col)):
                                    crop_mask = utils.cropSavefiles(mask, crop_size, crop_step, row, col)
                                    count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                                    if count_nonzero > mask_threshold:
                                        crop_img = utils.cropSavefiles(img, crop_size, crop_step, row, col)
                                        
                                        path_save = os.path.join(path_dst, path_src_third_file.replace('.png', f'_{str(count).zfill(4)}.png'))
                                        utils.savePNG(path_save, crop_img, resize=crop_size)
                                        count = count + 1
                                        
                        elif self.type == 'resize':
                            img = utils.readPNG(path_file, self.image_size)
                            path_save = os.path.join(path_dst, path_src_third_file)
                            utils.savePNG(path_save, img)
                    
                    if path_src_third_file.endswith('.txt'):
                        lightinfo = np.loadtxt(path_file)
                        if self.type == 'both':
                            # resize
                            path_save_resize = os.path.join(path_dst, path_src_third_file)
                            np.savetxt(path_save_resize, lightinfo)
                            
                            #crop
                            # mask = utils.readPNG(path_file.replace(path_src_second_file, 'image_mask'))
                            mask = utils.readPNG(path_file.replace(path_src_second_file, 'map_mask').replace('.txt', '.png'))
                            crop_step = int(self.image_size/2)
                            crop_size = self.image_size
                            
                            height, width, channel = img.shape
                            assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
                            assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
                            num_row = (height - crop_size) / crop_step + 1
                            num_col = (width - crop_size) / crop_step + 1
                            count = 0
                            mask_threshold = crop_size * crop_size * coef
                            for row in range(int(num_row)):
                                for col in range(int(num_col)):
                                    crop_mask = utils.cropSavefiles(mask, crop_size, crop_step, row, col)
                                    count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                                    if count_nonzero > mask_threshold:
                                        path_save = os.path.join(path_dst, path_src_third_file.replace('.txt', f'_{str(count).zfill(4)}.txt'))
                                        np.savetxt(path_save, lightinfo)
                                        count = count + 1
                            
                        elif self.type == 'crop':
                            # mask = utils.readPNG(path_file.replace(path_src_second_file, 'image_mask'))
                            mask = utils.readPNG(path_file.replace(path_src_second_file, 'map_mask').replace('.txt', '.png'))
                            crop_step = int(self.image_size/2)
                            crop_size = self.image_size
                            
                            height, width, channel = img.shape
                            assert (height - crop_size) % crop_step == 0, 'The size of image must be divide by size'
                            assert (width - crop_size) % crop_step == 0, 'The size of image must be divide by size'
                            num_row = (height - crop_size) / crop_step + 1
                            num_col = (width - crop_size) / crop_step + 1
                            count = 0
                            mask_threshold = crop_size * crop_size * coef
                            for row in range(int(num_row)):
                                for col in range(int(num_col)):
                                    crop_mask = utils.cropSavefiles(mask, crop_size, crop_step, row, col)
                                    count_nonzero = np.count_nonzero(crop_mask[:,:,0])
                                    if count_nonzero > mask_threshold:
                                        path_save = os.path.join(path_dst, path_src_third_file.replace('.txt', f'_{str(count).zfill(4)}.txt'))
                                        np.savetxt(path_save, lightinfo)
                                        count = count + 1
                        
                        elif self.type == 'resize':
                            path_save = os.path.join(path_dst, path_src_third_file)
                            np.savetxt(path_save, lightinfo)
    
  