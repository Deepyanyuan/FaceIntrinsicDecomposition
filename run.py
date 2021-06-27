# -*- encoding: utf-8 -*-
"""
@File    : run.py.py
@Time    : 2020/7/25 18:31
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
代码描述

'''
# ----- Load libraries -------------------------------------
import argparse
from main import setup_runtime, Trainer, FaceSep

# ----- Main function --------------------------------------
if __name__ == '__main__':
    # Begining runninga
    ## runtime arguments
    parser = argparse.ArgumentParser(description='Training configurations.')
    parser.add_argument('--config', default='experiments/train_realface.yml', type=str, help='Specify a config file path') # train step
    # parser.add_argument('--config', default='experiments/test_realface.yml', type=str, help='Specify a config file path') # test step
    parser.add_argument('--gpu', default=0, type=int, help='Specify a GPU device')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Specify the number of worker threads for data loaders')
    parser.add_argument('--seed', default=0, type=int, help='Specify a random seed')
    args = parser.parse_args()

    ## set up
    # 获取配置文件参数
    cfgs = setup_runtime(args)

    # 载入训练器，配置参数，网络模型
    run_train = cfgs.get('run_train', False)
    run_test = cfgs.get('run_test', False)
    trainer = Trainer(cfgs, FaceSep)

    ## run
    if run_train:
        trainer.train()
    if run_test:
        trainer.test()

    # Program over
    print('success! u are a smart boy!')
