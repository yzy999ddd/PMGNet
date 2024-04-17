import torch
import numpy as np
import cv2
# from skimage.measure import compare_ssim
import math
# import phasepack.phasecong as pc

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

def torchRMSE(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    return rmse

 
    
def _similarity_measure(x, y, constant):
    """
    Calculate feature similarity measurement between two images
    """
    numerator = 2 * x * y + constant
    denominator = x ** 2 + y ** 2 + constant

    return numerator / denominator


def _gradient_magnitude(img, img_depth):
    """
    Calculate gradient magnitude based on Scharr operator.
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)

    return np.sqrt(scharrx ** 2 + scharry ** 2)
  
def fsim(org_img, pred_img, T1 = 0.85, T2 = 160):
    # _assert_image_shapes_equal(org_img, pred_img, "FSIM")

    alpha = (
        beta
    ) = 1  # parameters used to adjust the relative importance of PC and GM features
    fsim_list = []
    for i in range(org_img.shape[2]):
        # Calculate the PC for original and predicted images
        pc1_2dim = pc(
            org_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )
        pc2_2dim = pc(
            pred_img[:, :, i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978
        )

        # pc1_2dim and pc2_2dim are tuples with the length 7, we only need the 4th element which is the PC.
        # The PC itself is a list with the size of 6 (number of orientation). Therefore, we need to
        # calculate the sum of all these 6 arrays.
        pc1_2dim_sum = np.zeros((org_img.shape[0], org_img.shape[1]), dtype=np.float64)
        pc2_2dim_sum = np.zeros(
            (pred_img.shape[0], pred_img.shape[1]), dtype=np.float64
        )
        for orientation in range(6):
            pc1_2dim_sum += pc1_2dim[4][orientation]
            pc2_2dim_sum += pc2_2dim[4][orientation]

        # Calculate GM for original and predicted images based on Scharr operator
        gm1 = _gradient_magnitude(org_img[:, :, i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[:, :, i], cv2.CV_16U)

        # Calculate similarity measure for PC1 and PC2
        S_pc = _similarity_measure(pc1_2dim_sum, pc2_2dim_sum, T1)
        # Calculate similarity measure for GM1 and GM2
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc ** alpha) * (S_g ** beta)

        numerator = np.sum(S_l * np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        denominator = np.sum(np.maximum(pc1_2dim_sum, pc2_2dim_sum))
        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)

# def torchSSIM(tar_img, prd_img, cr1):
#     # ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
#     ssim_pre, ssim_map = compare_ssim(tar_img, prd_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False, data_range = 1.0, full=True)
#     ssim_map = ssim_map * cr1
#     r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
#     win_size = 2 * r + 1
#     pad = (win_size - 1) // 2
#     ssim = ssim_map[pad:-pad,pad:-pad,:]
#     crop_cr1 = cr1[pad:-pad,pad:-pad,:]
#     ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
#     ssim = np.mean(ssim)
#     return ssim
