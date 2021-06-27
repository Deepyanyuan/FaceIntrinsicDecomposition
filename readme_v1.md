# 基于偏振特性的人脸高光分离与本征分解研究

## 题目: Research on Face Specular Separation and Intrinsic Decomposition Based on Polarization Characteristics

## 拟投：optics express journal



## 预解决的问题：

在人脸图像中，高光突出或者分散是一个较为常见的现象，但这不利于提取该区域的真实细节信息来进行人脸本征图像分解任务。常规方法要么忽略其高光特性，又或者只能依赖各种假设进行高光分离，很不符合其物理原理。另外，目前尚没有一个针对人脸高分辨率图像的高光分离和本征分解的数据集得以公布。

## 结构：

![image-20210618144927638](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210618144927638.png)

## 贡献点：

1)构建了一个多功能组合式的偏振照明系统来采集不同光照下的人脸高光-漫反射图像，并构建了首个高分辨率的亚洲人脸图像-材质贴图数据库；

2)对比了不同偏振模式下的人脸重渲染图像的效果，发现diffuse模式（cross偏振）下的获取的贴图的效果要优于mix模式（parallel 偏振）下的；

3)提出了一个联合高光分离和本征分解的对抗神经网络，目的是从单张带高光的人脸图像获得人脸纯漫反射图像和一系列的材质贴图，如表面法线贴图，反照率贴图，可见性贴图和残余贴图，用于人脸图像的重编辑和后处理等领域。



## data and code

step1：从百度云中下载原始文件ori放入data文件夹中，ori文件夹中包含三个文件夹，分别是train、val和test，所有图像的分辨率都为2048*2048。

step2：运行文件 ‘run_gen.py’ 生成resize模式和both模式，通过 ‘./experiments/pre_realFace.yml’来控制参数‘type’来实现。

step3：运行文件 ‘run.py’ 进行网络训练，通过 ‘./experiments/train_realface.yml’ 。

step4: 运行文件 ‘run.py’ 进行网络测试，通过 ‘./experiments/test_realface.yml’ 。



## parameters configuration

以‘./experiments/train_realface.yml’为例，进行说明：

‘no_deSpecular’表示没有高光分离

‘no_source_illumination’表示没有光源信息

当‘no_deSpecular: False’ and 'no_source_illumination: False'，则'checkpoint_dir' and 'test_result_dir'应该为'*-nodeF_nosourceF',表示our模型;

当‘no_deSpecular: False’ and 'no_source_illumination: True'，则'checkpoint_dir' and 'test_result_dir'应该为'*-nodeF_nosourceT'，表示our/L模型;

当‘no_deSpecular: True’ and 'no_source_illumination: True'，则'checkpoint_dir' and 'test_result_dir'应该为'*-nodeT_nosourceT'，表示our/LS模型;









