# -*- encoding: utf-8 -*-
'''
@File    : run.py.py
@Time    : 2020/7/25 18:31
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
'''
'''
代码描述

'''
# ----- Load libraries -------------------------------------
import argparse
import os
import yaml
from tqdm import tqdm
import time
from main import Processor


# ---- Custom functions -----------------------------------
def setup_runtime(args):
    '''Load configs, initialize CUDA, CuDNN and the random seeds.'''

    ## Load config
    cfgs = {}
    if args.config is not None and os.path.isfile(args.config):
        cfgs = load_yaml(args.config)

    return cfgs

# 载入配置文件参数
def load_yaml(path):
    print(f'Loading configs from {path}')
    with open(path, 'rb') as f:
        return yaml.safe_load(f)


# ----- Main function --------------------------------------
if __name__ == '__main__':
    # Begining running
    ## runtime arguments
    parser = argparse.ArgumentParser(description='Preprocess configurations.')
    parser.add_argument('--config', default='experiments/pre_realface.yml', type=str, help='Specify a config file path')
    args = parser.parse_args()

    ## set up
    cfgs = setup_runtime(args)
    root_path = cfgs.get('root_path', None)

    ## 载入处理器
    processor = Processor(cfgs)
    path_objects = processor.createPath()

    Total_start = time.time()
    # for i in tqdm(range(len(path_objects))):
    for i in tqdm(range(0,31)):         # 单个文件测试
    # for i in tqdm(range(5,17)):         # 单个文件测试
    # for i in tqdm(range(0,1)):         # 单个文件测试
    # for i in tqdm(range(1,2)):         # 单个文件测试
    
        Each_start = time.time()
        path_object = path_objects[i]
        print('获得不同的人物模型的文件：', path_object)
        processor.ALL(path_object)

        Each_end = time.time()
        print('each epoch need %.1f min' % ((Each_end - Each_start) / 60))

    Total_end = time.time()
    print('each epoch need %.1f min' % ((Total_end - Total_start) / 60))
    # Program over
    print('success! u are a smart boy!')
