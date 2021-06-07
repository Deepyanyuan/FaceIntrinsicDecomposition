import torch
import pytorch_msssim
import lpips

class Metrics():

    def PSNR(self, img0, img1, peak=255.):
        return 10*torch.log10(peak**2/((1.*img0-1.*img1)**2).mean())

    def metrics_all(self, img1, img2, device):
        l1 = torch.nn.L1Loss(reduction='sum').to(device) # reduction='mean','sum','none',其中'none'输出同shape的loss
        sl1 = torch.nn.SmoothL1Loss(reduction='sum').to(device)
        l2 = torch.nn.MSELoss(reduction='sum').to(device)
        
        ssim = pytorch_msssim.SSIM().to(device)
        msssim = pytorch_msssim.MS_SSIM().to(device)
        loss_lpips = lpips.LPIPS(net='alex').to(device)

        metrics_l1 = l1(img1, img2)
        metrics_sl1 = sl1(img1, img2)
        metrics_l2 = l2(img1, img2)
        metrics_psnr = self.PSNR(img1, img2).to(device)
        metrics_msssim = msssim(img1, img2)
        metrics_ssim = ssim(img1, img2)

        metrics_lpips = loss_lpips.forward(img1,img2).mean()

        # print('metrics_l1',metrics_l1)
        # print('metrics_sl1',metrics_sl1)
        # print('metrics_l2',metrics_l2)
        # print('metrics_psnr',metrics_psnr)
        # print('metrics_ssim',metrics_ssim)
        # print('metrics_msssim',metrics_msssim)
        # print('metrics_lpips',metrics_lpips)

        return metrics_l1,metrics_sl1,metrics_l2,metrics_psnr,metrics_ssim,metrics_msssim,metrics_lpips

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1 = torch.randn(12, 3, 256, 256).to(device)
    img2 = torch.randn(12, 3, 256, 256).to(device)
    metrics = Metrics()
    metrics_l1,metrics_sl1,metrics_l2,metrics_psnr,metrics_ssim,metrics_msssim,metrics_lpips = metrics.metrics_all(img1,img2,device)
    print('metrics_l1',metrics_l1)
    print('metrics_sl1',metrics_sl1)
    print('metrics_l2',metrics_l2)
    print('metrics_psnr',metrics_psnr)
    print('metrics_ssim',metrics_ssim)
    print('metrics_msssim',metrics_msssim)
    print('metrics_lpips',metrics_lpips)

if __name__ == '__main__':
    main()


