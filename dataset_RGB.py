import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex       = len(self.inp_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        if('blur' in inp_path):
            tar_path = self.tar_filenames[index_]
        elif ('artifact' in inp_path):
            tar_path = self.tar_filenames[index_].replace('input', 'target').replace('blur', 'sharp')
        elif ('rain-' in inp_path):
            tar_path = self.inp_filenames[index_].replace('input', 'target').replace('rain-', 'norain-')
        else :
            tar_path = self.inp_filenames[index_].replace('input', 'target')


        inp_img = Image.open(inp_path)
        #inp_img = inp_img.convert('L')
        tar_img = Image.open(tar_path)
        #tar_img = tar_img.convert('L')
        
        # 这部分代码获取目标图像的宽度和高度，并计算需要进行填充的宽度和高度。
        w,h = tar_img.size
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        # 如果需要进行填充，则使用torchvision.transforms.functional.pad函数对输入图像和目标图像进行反射填充。
        if padw!=0 or padh!=0:
            inp_img = TF.pad(inp_img, (0,0,padw,padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0,0,padw,padh), padding_mode='reflect')

        # 随机选择是否对输入图像和目标图像进行gamma调整。
        aug    = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)

        # 随机选择是否对输入图像和目标图像进行饱和度调整。
        aug    = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)

        # 将输入图像和目标图像转换为张量。
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        # 获取目标图像的高度和宽度，并随机选择裁剪的起始位置和数据增强方式
        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr     = random.randint(0, hh-ps) # 随机返回一个 0与hh-ps之间的值
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        # 根据裁剪的起始位置和给定的裁剪大小，对输入图像和目标图像进行裁剪操作。
        # 这里使用了切片操作[:, rr:rr+ps, cc:cc+ps]，其中rr和cc是随机选择的起始位置，ps是给定的裁剪大小。
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if aug==1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug==2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug==3:
            inp_img = torch.rot90(inp_img,dims=(1,2))
            tar_img = torch.rot90(tar_img,dims=(1,2))
        elif aug==4:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=2)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=2)
        elif aug==5:
            inp_img = torch.rot90(inp_img,dims=(1,2), k=3)
            tar_img = torch.rot90(tar_img,dims=(1,2), k=3)
        elif aug==6:
            inp_img = torch.rot90(inp_img.flip(1),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(1),dims=(1,2))
        elif aug==7:
            inp_img = torch.rot90(inp_img.flip(2),dims=(1,2))
            tar_img = torch.rot90(tar_img.flip(2),dims=(1,2))
        
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        if('blur' in inp_path):
            tar_path = self.tar_filenames[index_]
        elif ('artifact' in inp_path):
            tar_path = self.tar_filenames[index_].replace('input', 'target').replace('blur', 'sharp')
        elif ('rain-' in inp_path):
            tar_path = self.inp_filenames[index_].replace('input', 'target').replace('rain-', 'norain-')
        else :
            tar_path = self.inp_filenames[index_].replace('input', 'target')

        # inp_img = Image.open(inp_path)
        # tar_img = Image.open(tar_path)
        inp_img = Image.open(inp_path)
        # inp_img = inp_img.convert('L')
        tar_img = Image.open(tar_path)
        # tar_img = tar_img.convert('L')

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))
            tar_img = TF.center_crop(tar_img, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        tar_dir = inp_dir.replace('input','target')
        inp_files = sorted(os.listdir(inp_dir))
        tar_files = sorted(os.listdir(inp_dir.replace('input','target')))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(tar_dir, x) for x in tar_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        path_tar = self.tar_filenames[index]
        filename_inp = os.path.splitext(os.path.split(path_inp)[-1])[0]
        filename_tar = os.path.splitext(os.path.split(path_tar)[-1])[0]
        inp = Image.open(path_inp)
        tar = Image.open(path_tar)

        inp = TF.to_tensor(inp)
        tar = TF.to_tensor(tar)
        return inp, filename_inp, tar, filename_tar
