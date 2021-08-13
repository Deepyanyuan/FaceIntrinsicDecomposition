import torch
# import pytorch_msssim
import lpips
from pytorch_msssim import ssim, ms_ssim

class Metrics():

    def PSNR(self, img0, img1):
        mse = ((1.*img0-1.*img1)**2).mean()
        if mse < 1.0e-10:
            return 100
        peak = 1.
        return 10*torch.log10(peak**2/mse)
    

    def metrics_all(self, img1, img2, device):
        l1 = torch.nn.L1Loss(reduction='mean').to(device) # reduction='mean','sum','none',其中'none'输出同shape的loss
        # sl1 = torch.nn.SmoothL1Loss(reduction='mean').to(device)
        l2 = torch.nn.MSELoss(reduction='mean').to(device)
        cos3 = torch.nn.CosineEmbeddingLoss(0., reduction='mean').to(device)
        
        # ssim = pytorch_msssim.SSIM().to(device)
        # msssim = pytorch_msssim.MS_SSIM().to(device)
        loss_lpips = lpips.LPIPS(net='alex').to(device)

        metrics_l1 = l1(img1, img2)
        # metrics_sl1 = sl1(img1, img2)
        metrics_l2 = l2(img1, img2)
        target = torch.tensor([[1]], dtype=torch.float).to(device)
        metrics_cos3 = cos3(img1, img2, target)
        metrics_psnr = self.PSNR(img1, img2)
        metrics_ssim = ssim(img1, img2, data_range=1, size_average=False)
        metrics_msssim = ms_ssim(img1, img2, data_range=1, size_average=False)
        

        metrics_lpips = loss_lpips.forward(img1,img2).mean()

        # print('metrics_l1',metrics_l1)
        # print('metrics_sl1',metrics_sl1)
        # print('metrics_l2',metrics_l2)
        # print('metrics_cos3',metrics_cos3)
        # print('metrics_psnr',metrics_psnr)
        # print('metrics_ssim',metrics_ssim)
        # print('metrics_msssim',metrics_msssim)
        # print('metrics_lpips',metrics_lpips)

        # return metrics_l1,metrics_l2,metrics_sl1,metrics_cos3,metrics_psnr,metrics_ssim,metrics_msssim,metrics_lpips
        return metrics_l1,metrics_l2,metrics_cos3,metrics_psnr,metrics_ssim,metrics_msssim,metrics_lpips


    def metrics_ssim(self, img1, img2, device):
        
        metrics_ssim = ssim(img1, img2, data_range=1, size_average=False)
        metrics_msssim = ms_ssim(img1, img2, data_range=1, size_average=False)
        
        return metrics_ssim,metrics_msssim
    
    def metrics_psnr(self, img1, img2, device):
        metrics_psnr = self.PSNR(img1, img2)
        return metrics_psnr
    
    def metrics_lpips2(self, img1, img2, device):
        loss_lpips = lpips.LPIPS(net='alex').to(device)
        metrics_lpips = loss_lpips.forward(img1,img2).mean()
        
        return metrics_lpips
        
    def metrics_lpips(self, img1, img2, device):
        l1 = torch.nn.L1Loss(reduction='mean').to(device) # reduction='mean','sum','none',其中'none'输出同shape的loss
        # sl1 = torch.nn.SmoothL1Loss(reduction='mean').to(device)
        l2 = torch.nn.MSELoss(reduction='mean').to(device)
        cos3 = torch.nn.CosineEmbeddingLoss(0., reduction='mean').to(device)
        
        # ssim = pytorch_msssim.SSIM().to(device)
        # msssim = pytorch_msssim.MS_SSIM().to(device)
        # loss_lpips = lpips.LPIPS(net='alex').to(device)

        metrics_l1 = l1(img1, img2)
        # metrics_sl1 = sl1(img1, img2)
        metrics_l2 = l2(img1, img2)
        target = torch.tensor([[1]], dtype=torch.float).to(device)
        metrics_cos3 = cos3(img1, img2, target)
        metrics_psnr = self.PSNR(img1, img2)
        metrics_ssim = ssim(img1, img2, data_range=1, size_average=False)
        metrics_msssim = ms_ssim(img1, img2, data_range=1, size_average=False)
        
        # print('metrics_l1',metrics_l1)
        # print('metrics_sl1',metrics_sl1)
        # print('metrics_l2',metrics_l2)
        # print('metrics_cos3',metrics_cos3)
        # print('metrics_psnr',metrics_psnr)
        # print('metrics_ssim',metrics_ssim)
        # print('metrics_msssim',metrics_msssim)
        # print('metrics_lpips',metrics_lpips)

        # return metrics_l1,metrics_l2,metrics_sl1,metrics_cos3,metrics_psnr,metrics_ssim,metrics_msssim,metrics_lpips
        return metrics_l1,metrics_l2,metrics_cos3,metrics_psnr,metrics_ssim,metrics_msssim
    

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = torch.randn(1, 3, 256, 256).to(device)
    img2 = torch.randn(1, 3, 256, 256).to(device)
    # img2 = img1 *0.8
    metrics = Metrics()
    metrics_l1,metrics_l2,metrics_cos3,metrics_psnr,metrics_ssim,metrics_msssim,metrics_lpips = metrics.metrics_all(img1,img2,device)
    # metrics_l1,metrics_l2,metrics_sl1,metrics_cos3,metrics_psnr,metrics_ssim,metrics_msssim,metrics_lpips = metrics.metrics_all(img1,img1,device)
    print('metrics_l1',metrics_l1)
    
    print('metrics_l2',metrics_l2)
    print('metrics_cos3',metrics_cos3)
    print('metrics_psnr',metrics_psnr)
    print('metrics_ssim',metrics_ssim)
    print('metrics_msssim',metrics_msssim)
    print('metrics_lpips',metrics_lpips)

if __name__ == '__main__':
    main()


