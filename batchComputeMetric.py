import torch
import numpy as np
from main import metrics
import os
from PIL import Image
import scipy.io as scio

class batchProcession():
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = metrics.Metrics()
        self.gamma = 1
        self.srcPath = 'D:/04_paper/results for different states/'
        
        # self.fakeList = ['fake_image_mix','fake_image_diffuse','fake_image_residue','fake_image_lambertian','fake_image_mask','fake_map_normal','fake_map_albedo','fake_shading']
        # self.realList = ['real_image_mix','real_image_diffuse','real_image_residue','real_image_lambertian','real_image_mask','real_map_normal','real_map_albedo','real_shading']

        self.fakeList = ['fake_image_mix','fake_image_diffuse','fake_image_residue','fake_image_mask','fake_map_normal','fake_map_albedo']
        self.realList = ['real_image_mix','real_image_diffuse','real_image_residue','real_image_mask','real_map_normal','real_map_albedo']
        
    
    def computeSingleMetrics(self, img, img_gt):
        l1, sl1, l2, psnr, ssim, msssim, lpips = self.metrics.metrics_all(img, img_gt, self.device)
        return l1, sl1, l2, psnr, ssim, msssim, lpips
    
    def readImage(self, path, gamma, format='RGB'):
        img = Image.open(path).convert(format)
        img = np.array(img)
        img = (img / 255).astype(np.float32)
        img = np.power(img, gamma)
        return img
    
    def computeSingleList(self, Path, list, list_gt, gamma, format):
        mapList = []
        for k1 in range(len(list)):
            mapName = list[k1]
            mapName_gt = list_gt[k1]
            mapPath = os.path.join(os.path.abspath(Path), mapName)
            mapPath_gt = os.path.join(os.path.abspath(Path), mapName_gt)
            
            filesList = os.listdir(mapPath)
            filesList_gt = os.listdir(mapPath_gt)
            
            list_l1 = []
            list_l2 = []
            list_cos3 = []
            list_psnr = []
            list_ssim = []
            list_msssim = []
            list_lpips = []
            for k2 in range(len(filesList)):
                file = filesList[k2]
                file_gt = filesList_gt[k2]
                filePath = os.path.join(mapPath, file)
                filePath_gt = os.path.join(mapPath_gt, file_gt)
                
                img = self.readImage(filePath, gamma=gamma, format=format)
                img_gt = self.readImage(filePath_gt, gamma=gamma, format=format)
                
                # print('img.shape',img.shape)
                if format == 'RGB':
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    img_gt = torch.from_numpy(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
                elif format == 'L':
                    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(self.device)
                    img_gt = torch.from_numpy(img_gt).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(self.device)
                else:
                    print('image format is wrong!!!')
                # print('img.shape',img.shape)
                
                l1, l2, cos3, psnr, ssim, msssim, lpips = self.metrics.metrics_all(img, img_gt, self.device)
                list_l1.append(l1)
                list_l2.append(l2)
                list_cos3.append(cos3)
                list_psnr.append(psnr)
                list_ssim.append(ssim)
                list_msssim.append(msssim)
                list_lpips.append(lpips)
            mapList.append((list_l1,list_l2,list_cos3,list_psnr,list_ssim,list_msssim,list_lpips))
            
            #     l1, l2, cos3, psnr, ssim, msssim = self.metrics.metrics_lpips(img, img_gt, self.device)
            #     list_l1.append(l1)
            #     list_l2.append(l2)
            #     list_cos3.append(cos3)
            #     list_psnr.append(psnr)
            #     list_ssim.append(ssim)
            #     list_msssim.append(msssim)
            # mapList.append((list_l1,list_l2,list_cos3,list_psnr,list_ssim,list_msssim))
        
            #     ssim, msssim = self.metrics.metrics_ssim(img, img_gt, self.device)
            #     list_ssim.append(ssim)
            #     list_msssim.append(msssim)
            # mapList.append((list_ssim,list_msssim))
            
            #     psnr = self.metrics.metrics_psnr(img, img_gt, self.device)
            #     list_psnr.append(psnr)
            # mapList.append((list_psnr))
            
            #     lpips = self.metrics.metrics_lpips2(img, img_gt, self.device)
            #     list_lpips.append(lpips)
            # mapList.append((list_lpips))
            
        return mapList
        
    
    def computeAllLists(self):
        
        
        files_src_1 = os.listdir(self.srcPath)
        # for k1 in range(len(files_src_1)):
        for k1 in range(1):
            
            file_1 = files_src_1[k1]
            path_src_1 = os.path.join(self.srcPath, file_1, 'results')
            
            files_src_2 = os.listdir(path_src_1)
            for k2 in range(len(files_src_2)):
                file_src_2 = files_src_2[k2]
                path_src_2 = os.path.join(path_src_1, file_src_2)
                
                if os.path.isdir(path_src_2):
                    files_src_3 = os.listdir(path_src_2)
                    for k3 in range(len(files_src_3)):
                    # for k3 in range(1):
                        file_src_3 = files_src_3[k3]
                        path_src_3 = os.path.join(path_src_2, file_src_3)
                        
                        metricList = self.computeSingleList(path_src_3, self.fakeList, self.realList, gamma=1, format='RGB')
                        metricList = np.array(metricList).astype(np.float32)
                        metricsMean = metricList.mean(-1)
                        scio.savemat(path_src_1+'/metricsList_'+ file_src_2+'_'+ file_src_3+'.mat', {'metricsList':metricList})
                        np.savetxt(path_src_1+'/metricsList_'+ file_src_2+'_'+ file_src_3+'.txt', metricsMean)
        
        
                
if __name__ == '__main__':
    demo = batchProcession()
    demo.computeAllLists()
    