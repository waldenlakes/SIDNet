import os
import os.path as osp
import random
import shutil
import pandas as pd
import numpy as np
import json
from PIL import Image
import cv2
import torch

from pytorch_ssim import ssim as ssim_metric

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(file):
    """
    takes as input a file path and returns a configuration file
    that contains relevant information to the training of the NN
    :param file:
    :return:
    """

    # load the file as a raw file
    loaded_file = open(file)

    # conversion from json file to dictionary
    config = json.load(loaded_file)

    # returning the file to the caller
    return config

def mask_l1(output, target, mask, loss_f):
    loss = loss_f(output, target)
    loss = (loss * mask.float()).sum()  # gives \sigma_euclidean over unmasked elements

    if mask.shape[1] == 3:
        non_zero_elements = mask.sum()
    elif mask.shape[1] == 1:
        non_zero_elements = mask.sum() * 3
    else:
        raise RuntimeError('MaskChannelError')
    if non_zero_elements == 0:
        mse_loss_val = 0
    else:
        mse_loss_val = loss / non_zero_elements

    return mse_loss_val

class Mask_PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self, max_value):
        self.name = "Mask_PSNR"
        self.max_value = max_value # 255 or 1

    @staticmethod
    def __call__(img1, img2, mask, max_value):
        if mask.shape[1] == 1:
            mask = mask.repeat((1, 3, 1, 1))
        elif mask.shape[1] == 3:
            pass
        else:
            raise RuntimeError('MaskChannelError')
        pixelwise_loss = (img1 * mask.float() - img2 * mask.float()) ** 2
        non_zero_elements = mask.sum() 
        if non_zero_elements == 0:
            psnr = 30
        else:
            mse = pixelwise_loss.sum() / non_zero_elements
            psnr = 20 * torch.log10(max_value / torch.sqrt(mse))

        return psnr

def mask_ssim(image1, image2, mask):
    """ input tensor
    image1: B 3 H W
    image2: B 3 H W
    mask: B 1or3 H W
    """
    # ensure mask channel == 3
    if mask.shape[1] == 1:
        mask = mask.repeat((1, 3, 1, 1))
    elif mask.shape[1] == 3:
        pass
    else:
        raise RuntimeError('MaskChannelError')

    non_zero_elements = mask.sum() 
    if non_zero_elements == 0:
        ssim_value = 1
    else:
        ssim_value = ssim_metric(image1, image2, mask=mask)
    
    return ssim_value


def init_illum_labels(test_data_dir):
    illum_label_dir = os.path.join(test_data_dir, "illum_label")
    ILLUM_LABELS = {'sunny':[], 'sunset_sunrise':[], 'cloudy':[], 'night':[]}
    for index_name in ILLUM_LABELS:
        for did in open(os.path.join(illum_label_dir, index_name+'.txt'), 'r'):
            did = did.strip()
            ILLUM_LABELS[index_name].append(did)
    return ILLUM_LABELS

def subdatasets_statistics(ILLUM_LABELS, total_statistics, illum_name, single_results):
    labels = ['sunny', 'sunset_sunrise', 'cloudy', 'night']
    for illum_label in labels:
        if illum_name in ILLUM_LABELS[illum_label]:
            total_statistics[illum_label]['fMAE'].append(single_results[0])
            total_statistics[illum_label]['fPSNR'].append(single_results[1])
            total_statistics[illum_label]['fSSIM'].append(single_results[2])
            total_statistics[illum_label]['LPIPS'].append(single_results[3])
            total_statistics[illum_label]['name'].append(single_results[4])
            break

    return total_statistics

def save_subdataset_statistics(save_dir, total_statistics, which_type=''):
    labels = ['sunny', 'sunset_sunrise', 'cloudy', 'night']

    for illum_label in labels:
        subdataset_statistics = total_statistics[illum_label]
        fMAE, fPSNR, fSSIM, LPIPS_v = np.mean(subdataset_statistics['fMAE']), \
            np.mean(subdataset_statistics['fPSNR']), np.mean(subdataset_statistics['fSSIM']), np.mean(subdataset_statistics['LPIPS'])
        count = len(subdataset_statistics['fPSNR'])

        data=open(os.path.join(save_dir, "test_results.txt"),'a+')
        print(f'-----------------illum label : {illum_label} total number: {count} {which_type}-----------------', file=data)
        print(f'f-MAE : {fMAE}', file=data)
        print(f'f-PSNR : {fPSNR}', file=data)
        print(f'f-SSIM : {fSSIM}', file=data)
        print(f'LPIPS : {LPIPS_v}', file=data)
        data.close()


        data_frame = pd.DataFrame(
            data={'name': subdataset_statistics['name'], 'fMAE': subdataset_statistics['fMAE'], 'fPSNR': subdataset_statistics['fPSNR'], \
                'fSSIM': subdataset_statistics['fSSIM'], 'LPIPS': subdataset_statistics['LPIPS']}, \
            index=range(1, count + 1))

        if not os.path.exists(os.path.join(save_dir, "statistics")):
            os.makedirs(os.path.join(save_dir, "statistics"))
        data_frame.to_csv(os.path.join(save_dir, "statistics", f"illum_{illum_label}_results_{which_type}.csv"), index_label='ID')

def img_dilation(raw_harmonized_img, raw_fore_mask_np):
    kernel = np.ones((5,5), np.uint8)
    raw_harmonized_img_np = torch.squeeze(raw_harmonized_img, 0).permute(1, 2, 0).cpu().numpy()
    dilation = cv2.dilate(raw_harmonized_img_np, kernel, iterations=1)
    raw_harmonized_img_np = raw_harmonized_img_np * np.expand_dims(raw_fore_mask_np, 2) + dilation * (1 - np.expand_dims(raw_fore_mask_np, 2))
    return raw_harmonized_img_np

def extract_bounding_box(fg_img, fg_mask):
    ret, bin_fg_mask = cv2.threshold((fg_mask*255).astype(np.uint8), 127, 1, cv2.THRESH_BINARY)
    points = cv2.boundingRect(bin_fg_mask)

    bb_fg_mask = fg_mask[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]
    bb_fg_img = fg_img[points[1]:(points[1] + points[3]), points[0]:(points[0] + points[2])]

    return bb_fg_img, bb_fg_mask

def PIL_resize_with_antialiasing(img, shape):
    img[img<0] = 0
    img[img>1] = 1
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize(shape, Image.ANTIALIAS)
    img = np.array(img).astype(np.float32)/255.0

    return img


def do_composition(bb_fg_img, bb_fg_mask, bg_img, placement_config):
    OBJ_ATTRIBUTES = {"height":1.8}
    CAM_ATTRIBUTES = {"height":1.4, "HFOV":67.5/180*np.pi}

    # get ratio
    Pt_pixel_height = bg_img.shape[0] - placement_config['pt_h']
    tan_phi = (bg_img.shape[0]/2-Pt_pixel_height) / (bg_img.shape[0]*0.5) * np.tan(CAM_ATTRIBUTES["HFOV"]/2)
    OBJ_pixel_height = (tan_phi*np.tan(np.pi/2-CAM_ATTRIBUTES["HFOV"]/2)) / (1 - tan_phi*np.tan(np.pi/2-CAM_ATTRIBUTES["HFOV"]/2))  * OBJ_ATTRIBUTES['height'] / CAM_ATTRIBUTES['height'] * Pt_pixel_height
    RATIO = OBJ_pixel_height / bb_fg_mask.shape[0]

    if RATIO <= 0:
        RATIO = 0.5
    bb_h, bb_w = bb_fg_mask.shape
    nh, nw = int(RATIO * bb_h), int(RATIO * bb_w)
    # reshaped_fg_img = cv2.resize(bb_fg_img, (nw, nh), cv2.INTER_AREA)#cv2.INTER_CUBIC)
    # reshaped_bb_fg_mask = cv2.resize(bb_fg_mask, (nw, nh), cv2.INTER_AREA)#cv2.INTER_CUBIC)
    reshaped_fg_img = PIL_resize_with_antialiasing(bb_fg_img, (nw, nh))
    reshaped_bb_fg_mask = PIL_resize_with_antialiasing(bb_fg_mask, (nw, nh))
    reshaped_bb_fg_mask = np.expand_dims(reshaped_bb_fg_mask, 2)

    h_index, w_index = placement_config['pt_h'] - nh, placement_config['pt_w'] - nw//2
    # checking boundary
    if h_index < 0:
        h_index = 0
    elif h_index > (bg_img.shape[0]-nh):
        h_index = bg_img.shape[0]-nh
    if w_index < 0:
        w_index = 0
    elif w_index > (bg_img.shape[1]-nw):
        w_index = bg_img.shape[1]-nw
    
    new_fg_mask = np.zeros_like(bg_img)
    new_fg_mask[h_index:h_index + nh, w_index:w_index + nw] = reshaped_bb_fg_mask
    new_fg_img = np.zeros_like(bg_img)
    new_fg_img[h_index:h_index + nh, w_index:w_index + nw] = reshaped_fg_img

    # new_fg_mask = new_fg_mask / 255.0
    composited = new_fg_mask * new_fg_img + (1 - new_fg_mask) * bg_img

    # NORMALIZE
    composited[composited>1.0] = 1.0
    composited[composited<0.0] = 0.0
    # composited = composited / 255.0

    return composited.astype(np.float32), new_fg_mask.astype(np.float32)



def gamma_correction(image, gamma=2.2):
    return image ** (1 / gamma)


def do_tone_map(hdr):
    tonemapped = gamma_correction(hdr)  # tonemap(hdr, factor)
    return tonemapped# * 255


def create_or_recreate_folders(configs):
    """
    deletes existing folder if they already exist and
    recreates then. Only valid for training mode. does not work in
    resume mode
    :return:
    """

    folders = [configs['display_folder'],
               configs['summary'],
               configs['epoch_folder']]

    # iterate through the folders and delete them if they exist
    # then recreate them.
    # otherwise simply create them
    for i in range(len(folders)):
        folder = folders[i]
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)
        else:
            os.makedirs(folder)


def rotate_hdr(image, angle):
    # angle : 0 - 360 for rotation angle
    # (4096, 8192, 3)
    B, C, H, W = image.shape
    width = (angle / 360.0) * W

    front = image[:, :, :, 0:W - int(width)]
    back = image[:, :, :, W - int(width):]

    rotated = torch.cat([back, front], 3)
    return rotated

def render_shading(feature_shading, coeff):
    expanded_fs = torch.unsqueeze(feature_shading, 2).repeat(1, 1, 3, 1, 1)
    rendered = torch.sum((coeff * expanded_fs), dim=1)
    return rendered

def copy_file(src_path, tgt_path):
    if not osp.exists(tgt_path):
        filename = tgt_path.split('/')[-1]
        tgt_path_dir = tgt_path[:-1-len(filename)]
        if osp.exists(tgt_path_dir):
            shutil.copy(src_path, tgt_path)
        else:
            os.makedirs(tgt_path_dir)
            shutil.copy(src_path, tgt_path)
