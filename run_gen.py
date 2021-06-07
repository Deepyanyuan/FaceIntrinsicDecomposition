# -*- encoding: utf-8 -*-
"""
@File    : run_create.py
@Time    : 2020/8/16 17:55
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
代码描述

'''
# ----- Load libraries -------------------------------------
import numpy as np
import os
import argparse
from tqdm import tqdm
import time
import yaml
from main import Generator

# ---- Custom functions -----------------------------------
def setup_runtime(args):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    return cfgs

# 载入配置文件参数
def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'rb') as f:
        return yaml.safe_load(f)


# ----- Main function --------------------------------------
if __name__ == '__main__':
    # Begining running
    ## runtime arguments
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', default='experiments/pre_realface.yml', type=str, help='Specify a config file path')
    args = parser.parse_args()

    ## set up
    # 获取配置文件参数
    # python run.py --config experiments/train_synface.yml --gpu 0 --num_workers 4
    cfgs = setup_runtime(args)
    root_path = cfgs.get('root_path', None)
    type = cfgs.get('type', None)
    image_size = cfgs.get('image_size', None)

    # 载入处理器
    generator = Generator(cfgs)
    Total_start = time.time()
    generator.ALL_new()
    Total_end = time.time()
    print('The total models need %.1f min' % ((Total_end - Total_start) / 60))

    # Program over
    print('success! u are a smart boy!')
