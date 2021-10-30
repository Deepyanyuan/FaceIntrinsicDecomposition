# Research on Face Specular Removal and Intrinsic Decomposition Based on Polarization Characteristics

## title: Research on Face Specular Removal and Intrinsic Decomposition Based on Polarization Characteristics

## Journal：optics express journal



## Pre-solved problems：

In face images, highlighting or specular is a relatively common phenomenon, but this is not conducive to extracting the real detail information  for the task of facial image intrinsic decomposing. Conventional methods either ignore its specular characteristics, or can only rely on various assumptions for specular separation, which does not conform to its physical principles. In addition, there is currently no databases for specular separation and intrinsic decomposition of high-resolution images of human faces has been published.

## Architecture：

![image-20210618144927638](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210618144927638.png)

## Contributions：

(1)	We built a multi-functional practical polarized illumination system to collect face specular-diffuse images under different illuminations, and constructed the first high-resolution Asian Face Image-Material database.
(2)	After comparing the effects of face re-rendering images in different polarization modes, it is found that the effect of the materials obtained in diffuse mode (cross polarization) is better than that in mix mode (parallel polarization).
(3)	An adversarial neural network combining specular separation and intrinsic decomposition is proposed. It can obtain a pure diffuse image, normal map, albedo map, visibility map and residue map from a single face image with specular.



## data and code

step1：Download the original file ori from Baidu Cloud (links：https://pan.baidu.com/s/1EvRRAV1c5qJmY11CEYU-Xg 
password：ja1a) and put it into the data folder. The ori folder contains three folders, namely train, val, and test. The resolution of all images is 2048*2048.

step2：Run the file ‘run_gen.py’ to generate the resize mode and both mode, which can be achieved by controlling the parameter ‘type’ via ‘./experiments/pre_realFace.yml’.

step3：Run the file ‘run.py’ for network training through ‘./experiments/train_realface.yml’.

step4: Run the file ‘run.py’ for network testing through ‘./experiments/test_realface.yml’.



## parameters configuration

Take ‘./experiments/train_realface.yml’ as an example to illustrate:

‘no_deSpecular’ means no specular separation

‘no_source_illumination’ means there is no light source information

When 'no_deSpecular: False' and 'no_source_illumination: False', then 'checkpoint_dir' and 'test_result_dir' should be '-nodeF_nosourceF', which means Our model;

When 'no_deSpecular: False' and 'no_source_illumination: True', then 'checkpoint_dir' and 'test_result_dir' should be '-nodeF_nosourceT', which means Our/L model;











