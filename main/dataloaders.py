import os
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image


def get_data_loaders(cfgs):
    # -------------------------------------Beny Start-------------------------------------------
    batch_size = cfgs.get('batch_size', 64)
    num_workers = cfgs.get('num_workers', 4)
    image_size = cfgs.get('image_size', 256)
    crop = cfgs.get('crop', None)
    crop_height = cfgs.get('crop_height', 256)
    crop_width = cfgs.get('crop_width', 256)
    

    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', './data')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data/test')
    model_name = cfgs.get('model_name', 'RE')
    train_loader = val_loader = test_loader = None

    AB_dnames = cfgs.get('paired_data_dir_names', ['A', 'B'])
    AB_fnames = cfgs.get('paired_data_filename_diff', None)

    get_loader = lambda **kargs: get_paired_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop, crop_height=crop_height, crop_width=crop_width, AB_dnames=AB_dnames, AB_fnames=AB_fnames, model_name=model_name)

    if run_train:
        train_data_dir = os.path.join(train_val_data_dir, "train")
        val_data_dir = os.path.join(train_val_data_dir, "val")
        assert os.path.isdir(train_data_dir), "Training data directory does not exist: %s" %train_data_dir
        assert os.path.isdir(val_data_dir), "Validation data directory does not exist: %s" %val_data_dir
        print(f"Loading training data from {train_data_dir}")
        train_loader = get_loader(data_dir=train_data_dir, is_validation=False)
        print(f"Loading validation data from {val_data_dir}")
        val_loader = get_loader(data_dir=val_data_dir, is_validation=True)

    if run_test:
        assert os.path.isdir(test_data_dir), "Testing data directory does not exist: %s" %test_data_dir
        print(f"Loading testing data from {test_data_dir}")
        test_loader = get_loader(data_dir=test_data_dir, is_validation=True)
    
    return train_loader, val_loader, test_loader

    
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


## paired AB image dataset ##
def make_paied_dataset(dir, AB_dnames=None, AB_fnames=None):
    # -------------------------------------Beny Strat-----------------------------------------
    A_dname, A_dname_diff, B_dname_m, B_dname_co, B_dname_mn,B_dname_n, B_dname_al, B_dname_li = AB_dnames or ('A', 'B')
    dir_A = os.path.join(dir, A_dname)
    dir_A_diff = os.path.join(dir, A_dname_diff)
    dir_B_m = os.path.join(dir, B_dname_m)
    dir_B_co = os.path.join(dir, B_dname_co)
    dir_B_mn = os.path.join(dir, B_dname_mn)
    dir_B_n = os.path.join(dir, B_dname_n)
    dir_B_al = os.path.join(dir, B_dname_al)
    dir_B_li = os.path.join(dir, B_dname_li)
    assert os.path.isdir(dir_A), '%s is not a valid directory' % dir_A
    assert os.path.isdir(dir_A_diff), '%s is not a valid directory' % dir_A_diff
    assert os.path.isdir(dir_B_m), '%s is not a valid directory' % dir_B_m
    assert os.path.isdir(dir_B_co), '%s is not a valid directory' % dir_B_co
    assert os.path.isdir(dir_B_mn), '%s is not a valid directory' % dir_B_mn
    assert os.path.isdir(dir_B_n), '%s is not a valid directory' % dir_B_n
    assert os.path.isdir(dir_B_al), '%s is not a valid directory' % dir_B_al
    assert os.path.isdir(dir_B_li), '%s is not a valid directory' % dir_B_li
    
    images = []
    for root_A, _, fnames_A in sorted(os.walk(dir_A)):
        for fname_A in sorted(fnames_A):
            if is_image_file(fname_A):
                path_A = os.path.join(root_A, fname_A)
                root_A_diff = root_A.replace(dir_A, dir_A_diff, 1)
                root_B_m = root_A.replace(dir_A, dir_B_m, 1)
                root_B_co = root_A.replace(dir_A, dir_B_co, 1)
                root_B_mn = root_A.replace(dir_A, dir_B_mn, 1)
                root_B_n = root_A.replace(dir_A, dir_B_n, 1)
                root_B_al = root_A.replace(dir_A, dir_B_al, 1)
                root_B_li = root_A.replace(dir_A, dir_B_li, 1)
                if AB_fnames is not None:
                    fname_A_diff = fname_A.replace(AB_fnames[0], AB_fnames[1], 1)
                    fname_B_m = fname_A.replace(AB_fnames[0], AB_fnames[2], 1)
                    fname_B_co = fname_A.replace(AB_fnames[0], AB_fnames[3], 1)
                    fname_B_mn = fname_A.replace(AB_fnames[0], AB_fnames[4], 1)
                    fname_B_n = fname_A.replace(AB_fnames[0], AB_fnames[5], 1)
                    fname_B_al = fname_A.replace(AB_fnames[0], AB_fnames[6], 1)
                    fname_B_li = fname_A.replace(AB_fnames[0], AB_fnames[7], 1)
                else:
                    fname_A_diff = fname_A
                    fname_B_m = fname_A
                    fname_B_co = fname_A
                    fname_B_mn = fname_A
                    fname_B_n = fname_A
                    fname_B_al = fname_A
                    fname_B_li = fname_A
                path_A_diff = os.path.join(root_A_diff, fname_A_diff)
                path_B_m = os.path.join(root_B_m, fname_B_m)
                path_B_co = os.path.join(root_B_co, fname_B_co)
                path_B_mn = os.path.join(root_B_mn, fname_B_mn)
                path_B_n = os.path.join(root_B_n, fname_B_n)
                path_B_al = os.path.join(root_B_al, fname_B_al)
                path_B_li = os.path.join(root_B_li, fname_B_li).replace('.png', '.txt', 1)
                images.append((path_A, path_A_diff, path_B_m, path_B_co, path_B_mn,path_B_n, path_B_al, path_B_li))
    return images


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=128, crop=None, crop_height=None, crop_width=None, is_validation=False, AB_dnames=None, AB_fnames=None, model_name=None):
        super(PairedDataset, self).__init__()
        self.root = data_dir
        self.model_name = model_name
        self.paths = make_paied_dataset(data_dir, AB_dnames=AB_dnames, AB_fnames=AB_fnames)

        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.is_validation = is_validation

    # def transform(self, img, hflip=False):
    #     if self.crop is not None:
    #         if isinstance(self.crop, int):
    #             img = tfs.CenterCrop(self.crop)(img)
    #         else:
    #             assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
    #             img = tfs.functional.crop(img, *self.crop)
    #     if img.size[0] != self.image_size:
    #         # img = tfs.functional.resize(img, (self.image_size, self.image_size), Image.LANCZOS)
    #         img = tfs.functional.resize(img, (self.image_size, self.image_size))
    #     #     # print('resize')
    #     # img = tfs.functional.resize(img, (self.image_size, self.image_size), Image.LANCZOS)
    #     if hflip:
    #         img = tfs.functional.hflip(img)
    #     return tfs.functional.to_tensor(img)
    
    def transform(self, img, crop=None, hflip=False):
        if crop is not None:
            if isinstance(crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *crop)
        if img.size[0] != self.image_size:
            # img = tfs.functional.resize(img, (self.image_size, self.image_size), Image.LANCZOS)
            img = tfs.functional.resize(img, (self.image_size, self.image_size))
        #     # print('resize')
        # img = tfs.functional.resize(img, (self.image_size, self.image_size), Image.LANCZOS)
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        path_A, path_A_diff, path_B_m, path_B_co, path_B_mn,path_B_n, path_B_al, path_B_li = self.paths[index % self.size]
        img_A = Image.open(path_A).convert('RGB')
        img_A_diff = Image.open(path_A_diff).convert('RGB')
        img_B_m = Image.open(path_B_m).convert('L')
        img_B_co = Image.open(path_B_co).convert('RGB')
        img_B_mn = Image.open(path_B_mn).convert('RGB')
        img_B_n = Image.open(path_B_n).convert('RGB')
        img_B_al = Image.open(path_B_al).convert('RGB')
        txt_B_li = np.loadtxt(path_B_li)
        # hflip = not self.is_validation and np.random.rand()>0.5
        hflip = False
        
        '''
        ## method_1: 验证集不切分，训练集除了中心裁剪外，还要随机裁剪
        if not self.is_validation:
            crop = self.crop
        else:
            crop = None
        if self.crop is not None and np.random.rand()>0.5:
            random_nums = np.random.randint(512, high=1024,size=2)
            height, width = self.crop_height, self.crop_width
            crop = (random_nums[0], random_nums[1], height, width)
            
        ## method_2: 验证集不切分，训练集只有中心裁剪
        if not self.is_validation:
            crop = self.crop
        else:
            crop = None
        
        ## method_3: 验证集和训练集都要切分，且只有中心裁剪
        crop = self.crop
        '''
        crop = self.crop
        
        img_A = self.transform(img_A, crop=crop, hflip=hflip)
        img_A_diff = self.transform(img_A_diff, crop=crop, hflip=hflip)
        img_B_m = self.transform(img_B_m, crop=crop, hflip=hflip)
        img_B_co = self.transform(img_B_co, crop=crop, hflip=hflip)
        img_B_mn = self.transform(img_B_mn, crop=crop, hflip=hflip)
        img_B_n = self.transform(img_B_n, crop=crop, hflip=hflip)
        img_B_al = self.transform(img_B_al, crop=crop, hflip=hflip)
        txt_B_li = torch.from_numpy(txt_B_li)
        return img_A, img_A_diff, img_B_m, img_B_co, img_B_mn,img_B_n,img_B_al, txt_B_li

    def __len__(self):
        return self.size

    def name(self):
        return 'PairedDataset'


def get_paired_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=128, crop=None,crop_height=None,crop_width=None, AB_dnames=None, AB_fnames=None, model_name=None):

    dataset = PairedDataset(data_dir, image_size=image_size, crop=crop, crop_height=crop_height,crop_width=crop_width,\
        is_validation=is_validation, AB_dnames=AB_dnames, AB_fnames=AB_fnames, model_name=model_name)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
