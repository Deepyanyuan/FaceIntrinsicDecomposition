import math
from numpy.core.fromnumeric import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from . import networks
# from . import networksWithoutInplace as networks
# from . import render
from . import utils
import numpy as np
# import lpips

EPS = 1e-7


class FaceSep():
    def __init__(self, cfgs):
        # -------------------------------------Beny Start-------------------------------------------
        self.model_name = cfgs.get('model_name', self.__class__.__name__)
        self.device = cfgs.get('device', 'cpu')
        self.image_size = cfgs.get('image_size', 256)
        self.batch_size = cfgs.get('batch_size', 2)
        self.resize = cfgs.get('resize', 512)
        self.lam_diffuse = cfgs.get('lam_diffuse', 1)
        self.lam_normal = cfgs.get('lam_normal', 1)
        self.lam_light = cfgs.get('lam_light', 1)
        self.lam_subscatter = cfgs.get('lam_subscatter', 1)
        self.lam_sscatter = cfgs.get('lam_sscatter', 1)
        self.lam_dtL = cfgs.get('lam_dtL', 1)
        self.lam_gradient = cfgs.get('lam_gradient', 1)
        self.lam_norm = cfgs.get('lam_norm', 1)
        self.lam_perc = cfgs.get('lam_perc', 1)
        self.lam_recon = cfgs.get('lam_recon', 1)
        self.lam_deSpecular = cfgs.get('lam_deSpecular', 1)
        self.lam_intrinsics = cfgs.get('lam_intrinsics', 1)
        self.lam_error = cfgs.get('lam_error', 3)

        self.gamma = cfgs.get('gamma', 2.2)

        
        # self.ourRender = cfgs.get('ourRender', False)
        # self.deepRender = cfgs.get('deepRender', False)
        
        self.ngf = cfgs.get('ngf', 64)
        self.ndf = cfgs.get('ndf', 64)
        self.archG = cfgs.get('archG', 'unet_128')
        self.archD = cfgs.get('archD', 'basic')
        self.norm = cfgs.get('norm', 'batch')
        self.no_dropout = cfgs.get('no_dropout', True)
        self.init_type = cfgs.get('init_type', 'normal')
        self.init_gain = cfgs.get('init_gain', 0.02)
        self.gpu_ids = cfgs.get('gpu_ids', [0])
        self.no_lsgan = cfgs.get('no_lsgan', True)
        self.n_layers_D = cfgs.get('n_layers_D', 3)
        
        
        self.beta1 = cfgs.get('beta1', 3)
        self.lr = cfgs.get('lr', 1e-4)
        
        self.no_source_illumination = cfgs.get('no_source_illumination', False)
        self.no_deSpecular = cfgs.get('no_deSpecular', False)
        
        # self.lpips = lpips.LPIPS(net='alex').to(self.device)
        
        
        ## TODO: 还没修改，先占位
        if self.no_source_illumination:
            print("Generator which will NOT use the source illumination.")
            if not self.no_deSpecular:
                self.net_C = networks.define_G(5, 3,  # opt.input_nc, opt.output_nc,
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G1 = networks.define_G(8, 3,  # opt.input_nc, opt.output_nc,
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G2 = networks.define_G(8, 3,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G3 = networks.define_G(11, 3,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G4 = networks.define_G(8, 1,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
            else:
                self.net_C = networks.define_G(5, 3,  # opt.input_nc, opt.output_nc,
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G1 = networks.define_G(5, 3,  # opt.input_nc, opt.output_nc,
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G2 = networks.define_G(5, 3,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G3 = networks.define_G(8, 3,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G4 = networks.define_G(5, 1,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
        
        else:
            print("Generator which WILL use the source illumination.")
            if not self.no_deSpecular:
                self.net_C = networks.define_G(8, 3,  # opt.input_nc, opt.output_nc,
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G1 = networks.define_G(11, 3,  # opt.input_nc, opt.output_nc,
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G2 = networks.define_G(11, 3,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G3 = networks.define_G(21, 3,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G4 = networks.define_G(12, 1,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
            else:
                self.net_C = networks.define_G(8, 3,  # opt.input_nc, opt.output_nc,
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G1 = networks.define_G(8, 3,  # opt.input_nc, opt.output_nc,
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G2 = networks.define_G(8, 3,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G3 = networks.define_G(18, 3,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
                self.net_G4 = networks.define_G(9, 1,  # opt.input_nc, opt.output_nc, 
                                            self.ngf, self.archG, self.norm,
                                            not self.no_dropout, self.init_type, self.init_gain, self.gpu_ids)
            
        use_sigmoid = self.no_lsgan
        self.net_D = networks.define_D(3 + 3,  # opt.input_nc + opt.output_nc, 
                                    self.ndf, self.archD,
                                    self.n_layers_D, self.norm, use_sigmoid, self.init_type, self.init_gain, self.gpu_ids)
        self.criterionGAN = networks.GANLoss(gan_mode='vanilla').to(self.device)
            
        self.network_names = [k for k in vars(self) if 'net' in k]
        
        self.make_optimizer = lambda model: torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, 
            betas=(self.beta1, 0.999), 
            weight_decay=5e-4)
        # print('self.make_optimizer', self.make_optimizer)
        
        # self.make_optimizer = []
        list_model_G = list(self.net_G1.parameters())+list(self.net_G2.parameters())+list(self.net_G3.parameters())+list(self.net_G4.parameters())
        self.optimizer_C = torch.optim.Adam(self.net_C.parameters(),
                                            lr=self.lr,  # pytorch standard is 1e-3, pix2pix uses 2e-4
                                            betas=(self.beta1, 0.999),  # pytorch standard is (0.9, 0.999), pix2pix uses (0.5, 0.999)
                                            eps=1e-8,  # standard is 1e-8
                                            weight_decay=1e-3,  # standard is 0
                                            amsgrad=False,  # standard is False
                                            )
        self.optimizer_G = torch.optim.Adam(list_model_G,
                                            lr=self.lr,  # pytorch standard is 1e-3, pix2pix uses 2e-4
                                            betas=(self.beta1, 0.999),  # pytorch standard is (0.9, 0.999), pix2pix uses (0.5, 0.999)
                                            eps=1e-8,  # standard is 1e-8
                                            weight_decay=1e-3,  # standard is 0
                                            amsgrad=False,  # standard is False
                                            )
        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(),
                                            lr=self.lr,                 # pytorch standard is 1e-3, pix2pix uses 2e-4
                                            betas=(self.beta1, 0.999),  # pytorch standard is (0.9, 0.999), pix2pix uses (0.5, 0.999)
                                            eps=1e-8,                   # standard is 1e-8
                                            weight_decay=1e-3,          # standard is 0
                                            amsgrad=False,              # standard is False
                                            )
        
        
        # print('self.make_optimizer', self.make_optimizer)
        # other parameters
        self.PerceptualLoss = networks.PerceptualLoss(requires_grad=False)
        self.other_param_names = ['PerceptualLoss']
        # self.other_param_names = None
        # print('self.network_names', self.network_names)
        # -------------------------------------Beny End-------------------------------------------

    # --------------------------------------Beny Start----------------------------------------------------
    def init_optimizers(self):
        self.optimizer_names = []
        for net_name in self.network_names:
            optimizer = self.make_optimizer(getattr(self, net_name))
            optim_name = net_name.replace('net', 'optimizer')
            setattr(self, optim_name, optimizer)
            self.optimizer_names += [optim_name]
            # self.optimizer_names = self.optimizer_names + [optim_name]
            # self.optimizer_names.append(optim_name)

    
    def load_model_state(self, cp):
        for k in cp:
            if k and k in self.network_names:
                getattr(self, k).load_state_dict(cp[k])

    def load_optimizer_state(self, cp):
        for k in cp:
            if k and k in self.optimizer_names:
                getattr(self, k).load_state_dict(cp[k])

    def get_model_state(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        return states

    def get_optimizer_state(self):
        states = {}
        for optim_name in self.optimizer_names:
            states[optim_name] = getattr(self, optim_name).state_dict()
        return states

    def to_device(self, device):
        self.device = device
        for net_name in self.network_names:
            setattr(self, net_name, getattr(self, net_name).to(device))
        if self.other_param_names:
            for param_name in self.other_param_names:
                setattr(self, param_name, getattr(self, param_name).to(device))

    def set_train(self):
        for net_name in self.network_names:
            getattr(self, net_name).train()

    def set_eval(self):
        for net_name in self.network_names:
            getattr(self, net_name).eval()

    def photometric_loss(self, im1, im2, mask=None, conf_sigma=None, format='L1'):
        if format == 'L1':
            loss = (im1-im2).abs()
        elif format == 'L2':
            loss = ((im1-im2).abs()) **2
        # print('loss.shape',loss.shape)
        # print('mask.shape', mask.shape)
        if conf_sigma is not None:
            loss = loss * 2**0.5 / (conf_sigma + EPS) + \
                (conf_sigma + EPS).log()
        if mask is not None:
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss
    
    def criterionSimilarity(self, im1, im2, lambda_norm,lambda_perc,mask=None, conf_sigma=None, format='L1'):
        loss_photometric = lambda_norm * self.photometric_loss(im1, im2, mask=mask, conf_sigma=conf_sigma, format=format)
        loss_perceptual = lambda_perc * self.PerceptualLoss(im1, im2, mask=mask, conf_sigma=conf_sigma)
        # loss_perceptual = 0
        loss_all = loss_photometric + loss_perceptual
        
        return loss_photometric, loss_perceptual, loss_all
    
    def gradient_loss_l1(self, im1, im2, mask=None, conf_sigma=None):
        grad = im1[:,:,:-1,:-1]
        grad_p = im2[:,:,1:,1:]
        
        loss = (grad_p-grad).abs()
        if conf_sigma is not None:
            loss = loss * 2**0.5 / (conf_sigma + EPS) + \
                (conf_sigma + EPS).log()
        if mask is not None:
            mask = mask[:,:,1:,1:]
            mask = mask.expand_as(loss)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    def forward_G(self):
        ## G1 and G2, normal and albedo
        if self.no_source_illumination:
            if not self.no_deSpecular:
                input_G1_2 = torch.cat((
                                    self.real_image_mix_normalized,
                                    self.fake_image_diffuse_normalized.detach(),
                                    self.coord_normalized), dim=1)
            else:
                input_G1_2 = torch.cat((
                                    self.real_image_mix_normalized,
                                    self.coord_normalized), dim=1)
        else:
            if not self.no_deSpecular:
                input_G1_2 = torch.cat((
                                    self.real_image_mix_normalized,
                                    self.real_dirLight,
                                    self.fake_image_diffuse_normalized.detach(),
                                    self.coord_normalized), dim=1)
            else:
                input_G1_2 = torch.cat((
                                    self.real_image_mix_normalized,
                                    self.real_dirLight,
                                    self.coord_normalized), dim=1)
        
        
        self.fake_map_normal_normalized = self.net_G1(input_G1_2)
        norm = self.fake_map_normal_normalized.norm(p=2, dim=1, keepdim=True)
        self.fake_map_normal_normalized = self.fake_map_normal_normalized / norm
        self.fake_map_normal = self.fake_map_normal_normalized *0.5+0.5
        
        # calculate shading from normal
        self.real_shading = torch.clamp(torch.einsum('ijkl,ij->ikl', (self.real_map_normal_normalized, self.lightinfo[:,0:3]))[:, None, :, :], min=0)
        self.fake_shading = torch.clamp(torch.einsum('ijkl,ij->ikl', (self.fake_map_normal_normalized, self.lightinfo[:,0:3]))[:, None, :, :], min=0)
        self.real_shading_normalized = self.real_shading *2-1
        self.fake_shading_normalized = self.fake_shading *2-1

        self.fake_map_albedo_normalized = self.net_G2(input_G1_2)
        self.fake_map_albedo = self.fake_map_albedo_normalized *0.5+0.5
        # calculate diffuse
        self.real_image_lambertian = self.real_map_albedo * self.real_shading
        self.fake_image_lambertian = self.fake_map_albedo * self.fake_shading
        
        self.real_image_lambertian_normalized = self.real_image_lambertian *2-1
        self.fake_image_lambertian_normalized = self.fake_image_lambertian *2-1
        self.fake_image_residue_temp = (self.real_image_mix - self.fake_image_lambertian).clamp(0,1)
        self.fake_image_residue_temp_normalized = self.fake_image_residue_temp *2-1
        ## G4, mask
        if self.no_source_illumination:
            if not self.no_deSpecular:
                input_G4 = torch.cat((
                                    # self.real_map_normal_normalized,
                                    self.fake_map_normal_normalized,
                                    self.fake_image_diffuse_normalized.detach(),
                                    self.coord_normalized), dim=1)
            else:
                input_G4 = torch.cat((
                                    # self.real_map_normal_normalized,
                                    self.fake_map_normal_normalized,
                                    self.coord_normalized), dim=1)
        else:
            if not self.no_deSpecular:
                input_G4 = torch.cat((
                                    # self.real_map_normal_normalized,
                                    self.real_dirLight,
                                    self.fake_map_normal_normalized,
                                    self.fake_shading_normalized,
                                    self.fake_image_diffuse_normalized.detach(),
                                    self.coord_normalized), dim=1)
            else:
                input_G4 = torch.cat((
                                    # self.real_map_normal_normalized,
                                    self.real_dirLight,
                                    self.fake_map_normal_normalized,
                                    self.fake_shading_normalized,
                                    self.coord_normalized), dim=1)
            
        self.fake_image_mask_normalized = self.net_G4(input_G4)
        self.fake_image_mask = self.fake_image_mask_normalized *0.5+0.5
        
        ## G3
        if self.no_source_illumination:
            if not self.no_deSpecular:
                input_G3 = torch.cat((
                                    # self.real_map_normal_normalized,
                                    self.fake_map_normal_normalized,
                                    self.fake_map_albedo_normalized,
                                    self.fake_image_diffuse_normalized.detach(),
                                    self.coord_normalized), dim=1)
            else:
                input_G3 = torch.cat((
                                    # self.real_map_normal_normalized,
                                    self.fake_map_normal_normalized,
                                    self.fake_map_albedo_normalized,
                                    self.coord_normalized), dim=1)
        else:
            if not self.no_deSpecular:
                input_G3 = torch.cat((
                                    # self.real_map_normal_normalized,
                                    self.real_dirLight,
                                    self.fake_map_normal_normalized,
                                    self.fake_map_albedo_normalized,
                                    self.fake_shading_normalized,
                                    self.fake_image_lambertian_normalized,
                                    self.fake_image_residue_temp_normalized,
                                    self.fake_image_diffuse_normalized.detach(),
                                    self.coord_normalized), dim=1)
            else:
                input_G3 = torch.cat((
                                    # self.real_map_normal_normalized,
                                    self.real_dirLight,
                                    self.fake_map_normal_normalized,
                                    self.fake_map_albedo_normalized,
                                    self.fake_shading_normalized,
                                    self.fake_image_lambertian_normalized,
                                    self.fake_image_residue_temp_normalized,
                                    self.coord_normalized), dim=1)
        
        self.fake_image_residue_normalized = self.net_G3(input_G3)
        self.fake_image_residue = self.fake_image_residue_normalized *0.5+0.5
        self.fake_image_mix = (self.fake_image_lambertian + self.fake_image_residue).clamp(0,1)
        # self.fake_image_mix_normalized = self.fake_image_mix *2-1
        self.fake_image_mix = self.fake_image_mix * self.real_image_mask
        
        
    def forward(self, input):
        """Feedforward once."""
        
        image_mix, image_diffuse, image_mask, coord3, map_mask, map_normal_diffuse28, map_albedo_diffuse28, lightinfo = input
        # image_mix, image_diffuse, image_mask, coord3, map_normal_mix28, map_albedo_mix28, lightinfo = input
        
        self.mask = map_mask.to(self.device)
        self.mask_gray = self.mask[:,0,:,:].unsqueeze(dim=1)
        
        self.real_image_mix = image_mix.to(self.device) *self.mask
        self.real_image_diffuse = image_diffuse.to(self.device) *self.mask
        self.real_image_residue = (self.real_image_mix - self.real_image_diffuse).clamp(0,1) *self.mask
        self.real_image_mask = image_mask.to(self.device) *self.mask_gray
        self.coord = coord3[:,0:2,:,:].to(self.device)
        
        self.real_map_normal = map_normal_diffuse28.to(self.device) *self.mask    # 使用diffuse模式的表明normal
        self.real_map_albedo = map_albedo_diffuse28.to(self.device) *self.mask    # 使用diffuse模式的表明albedo
        # self.real_map_normal = map_normal_mix28.to(self.device) *self.mask      # 使用mix模式的表明normal
        # self.real_map_albedo = map_albedo_mix28.to(self.device) *self.mask      # 使用mix模式的表明albedo
        
        # print('self.real_image_mix.max()', self.real_image_mix.max())
        # print('self.real_image_min.max()', self.real_image_mix.min())
        
        
        self.real_image_mix_normalized = self.real_image_mix *2.-1.
        self.real_image_diffuse_normalized = self.real_image_diffuse *2.-1.
        self.real_image_residue_normalized = self.real_image_residue *2.-1.
        self.coord_normalized = self.coord *2.-1.
        self.real_map_normal_normalized = self.real_map_normal *2.-1.
        self.real_map_albedo_normalized = self.real_map_albedo *2.-1.
        # self.real_map_normal_normalized = self.real_map_normal *2.-1.
        # self.real_map_albedo_normalized = self.real_map_albedo  *2.-1.
        
        self.lightinfo = lightinfo.type(torch.FloatTensor).to(self.device)
        self.real_dirLight = self.lightinfo[:,0:3,None,None].expand_as(self.real_image_mix)
        self.real_dirView = self.lightinfo[:,3:6,None,None].expand_as(self.real_image_mix)
        
        ## C0 diffuse
        if self.no_source_illumination:
            input_C0 = torch.cat((self.real_image_mix_normalized,
                                #   self.real_dirLight,
                                    self.coord), dim=1)
        else:
            input_C0 = torch.cat((self.real_image_mix_normalized,
                                    self.real_dirLight,
                                    self.coord), dim=1)
        if self.model_name == 'FaceSep_CNN':
            self.fake_image_diffuse_normalized = self.net_C(input_C0)
            self.fake_image_diffuse = self.fake_image_diffuse_normalized *0.5+0.5
            with torch.no_grad():
                self.forward_G()
        
        if self.model_name == 'FaceSep_INS':
            with torch.no_grad():
                self.fake_image_diffuse_normalized = self.net_C(input_C0)
                self.fake_image_diffuse = self.fake_image_diffuse_normalized *0.5+0.5
            self.forward_G()
        
        if self.model_name == 'FaceSep_ALL':
            self.fake_image_diffuse_normalized = self.net_C(input_C0)
            self.fake_image_diffuse = self.fake_image_diffuse_normalized *0.5+0.5
            self.forward_G()
            
        ## loss
        # C0(A) = diffuse
        self.loss_Gdiffuse_norm,self.loss_Gdiffuse_perc,self.loss_Gdiffuse = self.criterionSimilarity(self.fake_image_diffuse*self.mask, self.real_image_diffuse, self.lam_norm, self.lam_perc)
        self.loss_Gdiffuse = self.loss_Gdiffuse *self.lam_deSpecular
        # print('self.loss_Gdiffuse_norm', self.loss_Gdiffuse_norm)
        # print('self.loss_Gdiffuse_perc', self.loss_Gdiffuse_perc)
        # print('self.loss_Gdiffuse', self.loss_Gdiffuse)
        
        
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_image_mix, self.fake_image_mix*self.mask), 1)
        pred_fake = self.net_D(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # G1(A) = normal
        self.loss_Gnormal_norm,self.loss_Gnormal_perc,self.loss_Gnormal = self.criterionSimilarity(self.fake_map_normal*self.mask, self.real_map_normal, self.lam_norm, self.lam_perc)
        self.loss_Gnormal =self.loss_Gnormal *self.lam_intrinsics
        
        
        # G2(A) = albedo
        self.loss_Galbedo = self.criterionSimilarity(self.fake_map_albedo*self.mask, self.real_map_albedo, self.lam_norm, self.lam_perc)[-1] *self.lam_intrinsics
        
        # G3(A) = error
        self.loss_Gerror = self.criterionSimilarity(self.fake_image_residue*self.mask, self.real_image_residue, self.lam_norm, self.lam_perc)[-1] *self.lam_error
        
        # G4(A) = visibility
        # print('self.mask_gray.shape', self.mask_gray.shape)
        # print('self.fake_image_mask.shape', self.fake_image_mask.shape)
        # print('self.real_image_mask.shape', self.real_image_mask.shape)
        self.loss_Gmask = self.criterionSimilarity(self.fake_image_mask*self.mask_gray, self.real_image_mask, self.lam_norm, self.lam_perc)[-1] *self.lam_recon
        
        # reconstruction
        self.loss_Grecon = self.criterionSimilarity(self.fake_image_lambertian*self.mask, self.real_image_lambertian, self.lam_norm, self.lam_perc)[-1] *self.lam_recon
        self.loss_Gtotal = self.criterionSimilarity(self.fake_image_mix*self.mask, self.real_image_mix, self.lam_norm, self.lam_perc)[-1] *self.lam_recon
       
        self.loss_Gintrinsics = self.loss_Gnormal + self.loss_Galbedo + self.loss_Gerror + self.loss_Gmask + self.loss_Grecon + self.loss_Gtotal
        # self.loss_G = self.loss_Gdiffuse + self.loss_Gintrinsics 
        metrics = {'loss_Gdiffuse': self.loss_Gdiffuse}
        metrics['loss_Gdiffuse_norm'] = self.loss_Gdiffuse_norm
        metrics['loss_Gnormal'] = self.loss_Gnormal
        metrics['loss_Gnormal_norm'] = self.loss_Gnormal_norm        
        metrics['loss_Galbedo'] = self.loss_Galbedo
        metrics['loss_Gerror'] = self.loss_Gerror
        metrics['loss_Gmask'] = self.loss_Gmask
        metrics['loss_Grecon'] = self.loss_Grecon
        metrics['loss_Gtotal'] = self.loss_Gtotal
        return metrics
        
    def backward_D(self,retain_graph=False):
        # Fake
        # stop backprop to the generator by detaching fake_AB
        fake_AB = torch.cat((self.real_image_mix, self.fake_image_mix*self.mask), dim=1)
        pred_fake = self.net_D(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        real_AB = torch.cat((self.real_image_mix, self.real_image_mix), dim=1)
        pred_real = self.net_D(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward(retain_graph=retain_graph)
        
    def backward_C(self):
        self.loss_C = self.loss_Gdiffuse
        self.loss_C.backward()
        
    def backward_G(self):
        self.loss_G = self.loss_Gintrinsics + self.loss_G_GAN
        # self.loss_G = self.loss_Gintrinsics
        self.loss_G.backward()
        
    
    def backward(self):
        if self.model_name == 'FaceSep_CNN':
            self.set_requires_grad(self.net_D, False)
            self.set_requires_grad(self.net_G1, False)
            self.set_requires_grad(self.net_G2, False)
            self.set_requires_grad(self.net_G3, False)
            self.set_requires_grad(self.net_G4, False)
            self.optimizer_C.zero_grad()
            self.backward_C()
            self.optimizer_C.step()
        
        if self.model_name == 'FaceSep_INS':
            # update D
            self.set_requires_grad(self.net_C, False)
            self.set_requires_grad(self.net_D, True)
            self.optimizer_D.zero_grad()
            # self.backward_D(retain_graph=True)
            self.backward_D()
            self.optimizer_D.step()

            # update C and G
            self.set_requires_grad(self.net_D, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            
        if self.model_name == 'FaceSep_ALL':
            # update D
            self.set_requires_grad(self.net_D, True)
            self.optimizer_D.zero_grad()
            # self.backward_D(retain_graph=True)
            self.backward_D()
            self.optimizer_D.step()

            # update C and G
            self.set_requires_grad(self.net_D, False)
            self.optimizer_C.zero_grad()
            self.backward_C()
            self.optimizer_C.step()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        

    # --------------------------------------Beny Start----------------------------------------
    # def visualize(self, logger, total_iter, max_bs=4):
    #     b0 = min(max_bs, self.batch_size)

    #     def log_grid_image(label, im, nrow=int(math.ceil(b0**0.5)), iter=total_iter):
    #         im_grid = torchvision.utils.make_grid(im, nrow=nrow)
    #         logger.add_image(label, im_grid, iter)
            
    #     real_img_mix = self.real_image_mix[:b0].detach().cpu() * 0.5+0.5
    #     real_img_diffuse = self.real_image_diffuse[:b0].detach().cpu() * 0.5+0.5
    #     real_img_mask = self.real_image_mask[:b0].detach().cpu() * 0.5+0.5
        
    #     fake_img_mix = 
    #     fake_img_diffuse = 
    #     fake_img_mask = 
        
    #     real_img_mix = self.real_image_mix[:b0].detach().cpu() * 0.5+0.5

    #     normal_gt = self.normal[:b0].detach().cpu() * 0.5+0.5
        
    #     # Prediction of S1
    #     com_diffuse_hist = self.com_diffuse.detach().cpu()
    #     com_diffuse = self.com_diffuse[:b0].detach().cpu() * 0.5+0.5
    #     com_normal_hist = self.com_normal.detach().cpu()
    #     com_normal = self.com_normal[:b0].detach().cpu() * 0.5+0.5

    #     # Prediction of S3
    #     com_dtL_hist = self.com_dtL.detach().cpu()
    #     com_dtL = self.com_dtL[:b0].detach().cpu() * 0.5+0.5
    #     com_ss_sp_dtV_hist = self.com_ss_sp_dtV.detach().cpu()
    #     com_ss_sp_dtV = self.com_ss_sp_dtV[:b0].detach().cpu() * 0.5+0.5
        
    #     img_recon_hist = self.img_recon.detach().cpu()
    #     img_recon = self.img_recon[:b0].detach().cpu() * 0.5+0.5
        
    #     ## gamma
    #     # gamma = self.gamma
    #     gamma = 1
    #     normal_gt = torch.pow(normal_gt, 1)
    #     img_input = torch.pow(input_im_s2[:,0:3,:,:], 1/gamma)
    #     img_diff = torch.pow(input_im_s2[:,3:6,:,:], 1/gamma)
        
    #     com_diffuse = torch.pow(com_diffuse, 1/gamma)
    #     com_dtL = torch.pow(com_dtL, 1/gamma)
    #     com_ss_sp_dtV = torch.pow(com_ss_sp_dtV, 1/gamma)
    #     com_normal = torch.pow(com_normal, 1)

    #     # write summary: histogram
    #     logger.add_histogram('diffuse/diffuse_hist',
    #                         com_diffuse_hist, total_iter)
    #     logger.add_histogram('dtL/specular_hist',
    #                         com_dtL_hist, total_iter)
    #     logger.add_histogram('ss_sp_dtV/ss_sp_dtV_hist',
    #                         com_ss_sp_dtV_hist, total_iter)
    #     logger.add_histogram('Normal/normal_hist', com_normal_hist, total_iter)

    #     # write summary: image
    #     log_grid_image('Image/com_diffuse', com_diffuse)
    #     log_grid_image('Image/com_dtL', com_dtL)
    #     log_grid_image('Image/com_ss_sp_dtV', com_ss_sp_dtV)
    #     log_grid_image('Image/com_normal', com_normal)
        
        
    #     img_recon = torch.pow(img_recon, 1/gamma)
    #     logger.add_histogram('Recon_image/img_recon_hist', img_recon_hist, total_iter)
    #     log_grid_image('Image/img_recon', img_recon)
        
        
    #     img_recon = torch.pow(img_recon, 1/gamma)
    #     logger.add_histogram('Recon_image/img_recon_hist', img_recon_hist, total_iter)
    #     log_grid_image('Image/img_recon', img_recon)
            
        
    #     log_grid_image('Image/normal_gt', normal_gt)
    #     log_grid_image('Image/img_input', img_input)
    #     log_grid_image('Image/img_diff', img_diff)
    #     # write summary: loss scale
    #     logger.add_scalar('Loss/loss_total', self.loss_total, total_iter)
    #     logger.add_scalar('Loss/loss_total_G', self.loss_total_G, total_iter)
    #     logger.add_scalar('Loss/loss_perc', self.loss_total_perc, total_iter)
    #     logger.add_scalar('Loss/loss_recon', self.loss_total_RE, total_iter)
        
    #     if self.ourRender:
    #         # Prediction of RE
    #         component_subsurface = self.component_subsurface[:b0].detach().cpu()
    #         component_sscatter = self.component_sscatter[:b0].detach().cpu()
    #         component_specular_ss = self.component_specular_ss[:b0].detach().cpu()
    #         component_specular = self.component_specular[:b0].detach().cpu()
    #         component_diffuse = self.component_diffuse[:b0].detach().cpu()
    #         # component_ambient = self.component_ambient[:b0].detach().cpu()

    #         com_component_subsurface = self.com_component_subsurface[:b0].detach().cpu()
    #         com_component_sscatter = self.com_component_sscatter[:b0].detach().cpu()
    #         com_component_specular_ss = self.com_component_specular_ss[:b0].detach().cpu()
    #         com_component_specular = self.com_component_specular[:b0].detach().cpu()
    #         com_component_diffuse = self.com_component_diffuse[:b0].detach().cpu()
    #         # com_component_ambient = self.com_component_ambient[:b0].detach().cpu()
            
    #         img_recon_0_hist = self.img_recon_0.detach().cpu()
    #         render_img_0_hist = self.render_img_0.detach().cpu()
    #         img_recon_0 = self.img_recon_0[:b0].detach().cpu() * 0.5+0.5
    #         render_img_0 = self.render_img_0[:b0].detach().cpu() * 0.5+0.5
            
    #         img_recon_1_hist = self.img_recon_1.detach().cpu()
    #         render_img_1_hist = self.render_img_1.detach().cpu()
    #         img_recon_1 = self.img_recon_1[:b0].detach().cpu() * 0.5+0.5
    #         render_img_1 = self.render_img_1[:b0].detach().cpu() * 0.5+0.5
        
    #         ## gamma
    #         component_sscatter = torch.pow(component_sscatter, 1/gamma)
    #         component_specular_ss = torch.pow(component_specular_ss, 1/gamma)
    #         component_subsurface = torch.pow(component_subsurface, 1/gamma)
    #         component_specular = torch.pow(component_specular, 1/gamma)
    #         component_diffuse = torch.pow(component_diffuse, 1/gamma)
    #         # component_ambient = torch.pow(component_ambient, 1/self.gamma)
    #         com_component_sscatter = torch.pow(com_component_sscatter, 1/gamma)
    #         com_component_specular_ss = torch.pow(com_component_specular_ss, 1/gamma)
    #         com_component_subsurface = torch.pow(com_component_subsurface, 1/gamma)
    #         com_component_specular = torch.pow(com_component_specular, 1/gamma)
    #         com_component_diffuse = torch.pow(com_component_diffuse, 1/gamma)
    #         # com_component_ambient = torch.pow(com_component_ambient, 1/self.gamma)
            
    #         img_recon_0 = torch.pow(img_recon_0, 1/gamma)
    #         img_recon_1 = torch.pow(img_recon_1, 1/gamma)
    #         render_img_0 = torch.pow(render_img_0, 1/gamma)
    #         render_img_1 = torch.pow(render_img_1, 1/gamma)

    #         logger.add_histogram(
    #             'Recon_image/img_recon_0_hist', img_recon_0_hist, total_iter)
    #         logger.add_histogram(
    #             'Recon_image/img_recon_1_hist', img_recon_1_hist, total_iter)
    #         logger.add_histogram(
    #             'Recon_image/render_img_0_hist', render_img_0_hist, total_iter)
    #         logger.add_histogram(
    #             'Recon_image/render_img_1_hist', render_img_1_hist, total_iter)

    #         log_grid_image('Image/component_subsurface', component_subsurface)
    #         log_grid_image('Image/component_sscatter', component_sscatter)
    #         log_grid_image('Image/component_specular_ss', component_specular_ss)
    #         log_grid_image('Image/component_specular', component_specular)
    #         log_grid_image('Image/component_diffuse', component_diffuse)
    #         # log_grid_image('Image/component_ambient', component_ambient)
    #         log_grid_image('Image/com_component_subsurface', com_component_subsurface)
    #         log_grid_image('Image/com_component_sscatter', com_component_sscatter)
    #         log_grid_image('Image/com_component_specular_ss', com_component_specular_ss)
    #         log_grid_image('Image/com_component_specular', com_component_specular)
    #         log_grid_image('Image/com_component_diffuse', com_component_diffuse)
    #         # log_grid_image('Image/com_component_ambient', com_component_ambient)
            
    #         log_grid_image('Image/img_recon_0', img_recon_0)
    #         log_grid_image('Image/img_recon_1', img_recon_1)
    #         log_grid_image('Image/render_img_0', render_img_0)
    #         log_grid_image('Image/render_img_1', render_img_1)
                
        
    def save_results(self, save_dir):
        mask = self.mask.detach().cpu().numpy()
        real_image_mix = self.real_image_mix.detach().cpu().numpy()
        real_image_diffuse = self.real_image_diffuse.detach().cpu().numpy()
        real_image_residue = self.real_image_residue.detach().cpu().numpy()
        real_image_mask = self.real_image_mask.detach().cpu().numpy()
        real_map_normal = self.real_map_normal.detach().cpu().numpy()
        real_map_albedo = self.real_map_albedo.detach().cpu().numpy()
        real_shading = self.real_shading.detach().cpu().numpy() *mask
        real_image_lambertian = self.real_image_lambertian.detach().cpu().numpy()
        
        
        
        fake_image_mix = self.fake_image_mix.detach().cpu().numpy() *mask 
        fake_image_diffuse = self.fake_image_diffuse.detach().cpu().numpy() *mask 
        fake_image_residue = self.fake_image_residue.detach().cpu().numpy() *mask 
        fake_image_mask = self.fake_image_mask.detach().cpu().numpy() *mask 
        # fake_image_mask = self.fake_image_mask.detach().cpu().numpy() *mask 
        fake_map_normal = self.fake_map_normal.detach().cpu().numpy() *mask 
        fake_map_albedo = self.fake_map_albedo.detach().cpu().numpy() *mask 
        fake_shading = self.fake_shading.detach().cpu().numpy() *mask 
        fake_image_lambertian = self.fake_image_lambertian.detach().cpu().numpy() *mask 
        
        if self.gamma != 1:
            real_image_mix = utils.linearToSrgb(real_image_mix)
            real_image_diffuse = utils.linearToSrgb(real_image_diffuse)
            real_image_residue = utils.linearToSrgb(real_image_residue)
            # real_image_mask = utils.linearToSrgb(real_image_mask)
            # real_map_normal = utils.linearToSrgb(real_map_normal)
            real_map_albedo = utils.linearToSrgb(real_map_albedo)
            real_shading = utils.linearToSrgb(real_shading)
            real_image_lambertian = utils.linearToSrgb(real_image_lambertian)
            
            fake_image_mix = utils.linearToSrgb(fake_image_mix)
            fake_image_diffuse = utils.linearToSrgb(fake_image_diffuse)
            fake_image_residue = utils.linearToSrgb(fake_image_residue)
            # fake_image_mask = utils.linearToSrgb(fake_image_mask)
            # fake_map_normal = utils.linearToSrgb(fake_map_normal)
            fake_map_albedo = utils.linearToSrgb(fake_map_albedo)
            fake_shading = utils.linearToSrgb(fake_shading)
            fake_image_lambertian = utils.linearToSrgb(fake_image_lambertian)
        
        # save images
        sep_folder = True
        
        utils.save_images(save_dir, real_image_mix, self.resize,
                          suffix='real_image_mix', sep_folder=sep_folder)
        utils.save_images(save_dir, real_image_diffuse, self.resize,
                          suffix='real_image_diffuse', sep_folder=sep_folder)
        utils.save_images(save_dir, real_image_residue, self.resize,
                          suffix='real_image_residue', sep_folder=sep_folder)
        utils.save_images(save_dir, real_image_mask, self.resize,
                          suffix='real_image_mask', sep_folder=sep_folder)
        utils.save_images(save_dir, real_map_normal, self.resize,
                          suffix='real_map_normal', sep_folder=sep_folder)
        utils.save_images(save_dir, real_map_albedo, self.resize,
                          suffix='real_map_albedo', sep_folder=sep_folder)
            
        utils.save_images(save_dir, real_shading, self.resize,
                        suffix='real_shading', sep_folder=sep_folder)
        utils.save_images(save_dir, real_image_lambertian, self.resize,
                        suffix='real_image_lambertian', sep_folder=sep_folder)
        
        utils.save_images(save_dir, fake_image_mix, self.resize,
                        suffix='fake_image_mix', sep_folder=sep_folder)
        utils.save_images(save_dir, fake_image_diffuse, self.resize,
                          suffix='fake_image_diffuse', sep_folder=sep_folder)
        utils.save_images(save_dir, fake_image_residue, self.resize,
                        suffix='fake_image_residue', sep_folder=sep_folder)
        utils.save_images(save_dir, fake_image_mask, self.resize,
                        suffix='fake_image_mask', sep_folder=sep_folder)
        utils.save_images(save_dir, fake_map_normal, self.resize,
                        suffix='fake_map_normal', sep_folder=sep_folder)
        utils.save_images(save_dir, fake_map_albedo, self.resize,
                        suffix='fake_map_albedo', sep_folder=sep_folder)
        utils.save_images(save_dir, fake_shading, self.resize,
                        suffix='fake_shading', sep_folder=sep_folder)
        utils.save_images(save_dir, fake_image_lambertian, self.resize,
                        suffix='fake_image_lambertian', sep_folder=sep_folder)
        
            