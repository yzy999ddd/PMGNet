import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_test_data
from PMGNet import PMGNet
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage.metrics import structural_similarity as compare_ssim
import csv

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='./test/motion/input', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='./checkpoints/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='OASIS', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = PMGNet()


utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset

rgb_dir_test = os.path.join(args.input_dir)
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

#####################################################
psnr_test = []
rmse_test = []
ssim_test = []
f = open('results-PMGNet-OASIS.csv', 'w', encoding='utf-8')
csv_writer = csv.writer(f)
csv_writer.writerow(["methods", "data", "psnr", "ssim"])
#####################################################

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_    = data_test[0].cuda()
        filenames = data_test[1]
        target_ = data_test[2].cuda()
        filenames_tar = data_test[3]
        # Padding in case images are not multiples of 8
        # if dataset == 'Rain100L' or dataset == 'BSD400':
        #     factor = 16
        #     h,w = input_.shape[2], input_.shape[3]
        #     H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        #     padh = H-h if h%factor!=0 else 0
        #     padw = W-w if w%factor!=0 else 0
        #     input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

        restored = model_restoration(input_)
        restored = torch.clamp(restored[0],0,1)

        psnr = utils.torchPSNR(restored, target_)
        rmse = utils.torchRMSE(restored, target_)
        psnr_test.append(psnr)
        rmse_test.append(rmse)

        restored = restored.squeeze(0)
        restored = torch.transpose(restored, 0, -1)
        target_ = target_.squeeze(0)
        target_ = torch.transpose(target_, 0, -1)
        restored = restored.cpu().numpy()
        target_ = target_.cpu().numpy()

        ssim = compare_ssim(restored, target_, multichannel=True)
        ssim_test.append(ssim)

        csv_writer.writerow(["PMGNet", filenames, psnr, ssim])

psnr_test = torch.stack(psnr_test).mean().item()
ssim_test = np.stack(ssim_test, 0)
ssim_test = torch.from_numpy(ssim_test).mean().item()
csv_writer.writerow(["avg", dataset, psnr_test, ssim_test])

f.close()