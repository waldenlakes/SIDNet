import os
import numpy as np
import cv2
import torch
import torch.utils.data
import torch.nn as nn
import torchvision.transforms.functional as TF
import lpips

from model.test_model import SIDNet
from data.test_illumharmony_dataset import TestIllumHarmonyDataset
from utils import load_config, extract_bounding_box, do_composition, img_dilation, mask_l1, Mask_PSNR, \
    mask_ssim, init_illum_labels, subdatasets_statistics, save_subdataset_statistics


# ------------------------- configuration -------------------------
config = load_config('./configs/config_test.json')['config']
print(config)

devices = config['gpus']
os.environ["CUDA_VISIBLE_DEVICES"] = devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(config['save_dir']):
    os.mkdir(config['save_dir'])
# ------------------------- configuration -------------------------


# ------------------------- network setup -------------------------
sidnet = SIDNet(ckpts=config['ckpts'])
sidnet.to(device)
# # ------------------------- network setup -------------------------


# ------------------------- dataset -------------------------
dataset = TestIllumHarmonyDataset(config['test_data_path'])
print(f'Dataset Test: {len(dataset)}')

test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                        num_workers=config['val_workers'],
                                        batch_size=config['val_batch'],
                                        shuffle=config['val_shuffle'])

ILLUM_LABELS = init_illum_labels(config['test_data_path'])
# ------------------------- dataset -------------------------


# # ------------------------- metrics setup -------------------------
l1_loss = nn.L1Loss(reduction='none')
mask_psnr_metric = Mask_PSNR(max_value=1)
lpips_loss = lpips.LPIPS()
lpips_loss.to(device)
# # ------------------------- metrics setup -------------------------


if __name__ == '__main__':

    f_mae = 0
    f_psnr = 0
    ssim = 0
    total_lpips = 0
    sunny_statistics = {'fMAE':[], 'fPSNR':[], 'fSSIM':[], 'LPIPS':[], 'name':[]}
    sunset_sunrise_statistics = {'fMAE':[], 'fPSNR':[], 'fSSIM':[], 'LPIPS':[], 'name':[]}
    cloudy_statistics = {'fMAE':[], 'fPSNR':[], 'fSSIM':[], 'LPIPS':[], 'name':[]}
    night_statistics = {'fMAE':[], 'fPSNR':[], 'fSSIM':[], 'LPIPS':[], 'name':[]}
    total_statistics = {'sunny':sunny_statistics, 'sunset_sunrise':sunset_sunrise_statistics, 'cloudy':cloudy_statistics, 'night':night_statistics}

    pseudogt_ssim = 0
    pseudogt_f_mae = 0
    pseudogt_f_psnr = 0
    pseudogt_total_lpips = 0
    pseudogt_sunny_statistics = {'fMAE':[], 'fPSNR':[], 'fSSIM':[], 'LPIPS':[], 'name':[]}
    pseudogt_sunset_sunrise_statistics = {'fMAE':[], 'fPSNR':[], 'fSSIM':[], 'LPIPS':[], 'name':[]}
    pseudogt_cloudy_statistics = {'fMAE':[], 'fPSNR':[], 'fSSIM':[], 'LPIPS':[], 'name':[]}
    pseudogt_night_statistics = {'fMAE':[], 'fPSNR':[], 'fSSIM':[], 'LPIPS':[], 'name':[]}
    pseudogt_total_statistics = {'sunny':pseudogt_sunny_statistics, 'sunset_sunrise':pseudogt_sunset_sunrise_statistics, 'cloudy':pseudogt_cloudy_statistics, 'night':pseudogt_night_statistics}

    iter = 0
    for tensors_dic in test_loader:

        iter = iter + 1
        # print(f'Test [{iter}|{len(test_loader)}]')

        background_image = tensors_dic['background_image']
        hdr_image = tensors_dic['hdr']
        alpha_fore_mask = tensors_dic['alpha_foreground_mask']
        raw_bin_fore_mask = tensors_dic['bin_fore_mask']
        back_mask = tensors_dic['back_mask']
        raw_unharmonized_img = tensors_dic['masked_unharmonized_input']
        full_unharmonized_input = tensors_dic['full_unharmonized_input']
        full_color_img_gt = tensors_dic['full_gt_color_image']

        placement_config = tensors_dic['placement_config']
        illum_name = tensors_dic['hdr_name'][0]
        gt_img_path = tensors_dic['gt_img_path'][0]
        obj_name = gt_img_path.split('/')[-4]
        obj_rotation = gt_img_path.split('/')[-1].split('_')[1]
        hdr_rotation = gt_img_path.split('/')[-1].split('_')[-1].split('.')[0]

        if torch.cuda.is_available():
            background_image = background_image.to(device)
            hdr_image = hdr_image.to(device)
            raw_bin_fore_mask = raw_bin_fore_mask.to(device)
            back_mask = back_mask.to(device)
            raw_unharmonized_img = raw_unharmonized_img.to(device)
            alpha_fore_mask = alpha_fore_mask.to(device)

        with torch.no_grad():
            raw_harmonized_img, raw_pseudogt_harmonized_img = sidnet(raw_unharmonized_img, background_image, hdr_image)


        # ------------------------- object placement -------------------------# 
        # Step 1: convert tensor to numpy
        alpha_fore_mask_np = torch.squeeze(alpha_fore_mask[...,config['comp_crop'][0]:config['comp_crop'][0]+config['comp_crop'][1]]).cpu().numpy()
        raw_bin_fore_mask_np = torch.squeeze(raw_bin_fore_mask[...,config['comp_crop'][0]:config['comp_crop'][0]+config['comp_crop'][1]]).cpu().numpy()  # .permute(1, 2, 0)
        full_unharmonized_input_np = torch.squeeze(full_unharmonized_input[...,config['comp_crop'][0]:config['comp_crop'][0]+config['comp_crop'][1]], 0).permute(1, 2, 0).cpu().numpy()
        full_color_img_gt_np = torch.squeeze(full_color_img_gt[...,config['comp_crop'][0]:config['comp_crop'][0]+config['comp_crop'][1]], 0).permute(1, 2, 0).cpu().numpy()
        bg_img_np = torch.squeeze(background_image, 0).permute(1, 2, 0).cpu().numpy()
        placemask_np = torch.squeeze(back_mask).cpu().numpy()
        raw_harmonized_img_np = img_dilation(raw_harmonized_img, raw_bin_fore_mask_np) # to remove foreground black edges 
        raw_pseudogt_harmonized_img_np = img_dilation(raw_pseudogt_harmonized_img, raw_bin_fore_mask_np)

        # Step 2: image composition for unharmonized img, harmonized img and GT.
        bb_gt_img_np, bb_mask_np = extract_bounding_box(full_color_img_gt_np, alpha_fore_mask_np)
        gt_img_np, mask_np = do_composition(bb_gt_img_np, bb_mask_np, bg_img_np, placement_config)
        bb_unharmonized_img_np, bb_mask_np = extract_bounding_box(full_unharmonized_input_np, alpha_fore_mask_np)
        unharmonized_img, mask_np = do_composition(bb_unharmonized_img_np, bb_mask_np, bg_img_np, placement_config)
        bb_harmonized_img_np, bb_mask_np = extract_bounding_box(raw_harmonized_img_np, alpha_fore_mask_np)
        harmonized_img, mask_np = do_composition(bb_harmonized_img_np, bb_mask_np, bg_img_np, placement_config)
        bb_pseudogt_harmonized_img_np, bb_mask_np = extract_bounding_box(raw_pseudogt_harmonized_img_np, alpha_fore_mask_np)
        pseudogt_harmonized_img, mask_np = do_composition(bb_pseudogt_harmonized_img_np, bb_mask_np, bg_img_np, placement_config)

        if config['save_img']:
            display_data = np.concatenate((unharmonized_img, harmonized_img, pseudogt_harmonized_img, gt_img_np, mask_np), 1)
            display_data = np.flip(display_data, -1)
            save_img_dir = os.path.join(config['save_dir'], "img_results")
            if not os.path.exists(save_img_dir):
                os.mkdir(save_img_dir)
            cv2.imwrite(os.path.join(save_img_dir, f"{iter}.png"), display_data*255.0)
        # ------------------------- object placement -------------------------#


        # ------------------------- cal results -------------------------#
        # reconvert the rensors back to torch then compute metrics
        f_comp_un_torch = torch.unsqueeze(torch.from_numpy(unharmonized_img).permute(2, 0, 1), 0).to(device)
        f_comp_ha_torch = torch.unsqueeze(torch.from_numpy(harmonized_img).permute(2, 0, 1), 0).to(device)
        f_pseudogt_comp_ha_torch = torch.unsqueeze(torch.from_numpy(pseudogt_harmonized_img).permute(2, 0, 1), 0).to(device)
        f_comp_gt_torch = torch.unsqueeze(torch.from_numpy(gt_img_np).permute(2, 0, 1), 0).to(device)
        new_mask_torch = torch.unsqueeze(torch.from_numpy(mask_np).permute(2, 0, 1), 0).to(device)

        # resize all images to 512x512
        f_comp_ha_torch = TF.resize(f_comp_ha_torch, (config['metrics_imgsize'], config['metrics_imgsize']))
        f_pseudogt_comp_ha_torch = TF.resize(f_pseudogt_comp_ha_torch, (config['metrics_imgsize'], config['metrics_imgsize']))
        f_comp_gt_torch = TF.resize(f_comp_gt_torch, (config['metrics_imgsize'], config['metrics_imgsize']))
        new_mask_torch = TF.resize(new_mask_torch, (config['metrics_imgsize'], config['metrics_imgsize']))

        # compute metrics for our SIDNet
        f_mae_loss = mask_l1(new_mask_torch * f_comp_ha_torch, new_mask_torch * f_comp_gt_torch, new_mask_torch, l1_loss)
        f_psnr_loss = mask_psnr_metric(f_comp_ha_torch, f_comp_gt_torch, new_mask_torch, max_value=1)
        ssim_loss = mask_ssim(new_mask_torch * f_comp_ha_torch, new_mask_torch * f_comp_gt_torch, new_mask_torch)
        lpips_loss_v = lpips_loss.forward(f_comp_ha_torch, f_comp_gt_torch, normalize=True)

        # compute and save metrics for our SIDNet with gt illum maps
        pseudogt_ssim_loss = mask_ssim(new_mask_torch * f_pseudogt_comp_ha_torch, new_mask_torch * f_comp_gt_torch, new_mask_torch)
        pseudogt_f_mae_loss = mask_l1(new_mask_torch * f_pseudogt_comp_ha_torch, new_mask_torch * f_comp_gt_torch, new_mask_torch, l1_loss)
        pseudogt_f_psnr_loss = mask_psnr_metric(f_pseudogt_comp_ha_torch, f_comp_gt_torch, new_mask_torch, max_value=1)
        pseudogt_lpips_loss_v = lpips_loss.forward(f_pseudogt_comp_ha_torch, f_comp_gt_torch, normalize=True)

        f_mae += f_mae_loss.item()
        f_psnr += f_psnr_loss.item()
        ssim += ssim_loss.item()
        total_lpips += lpips_loss_v.item()

        pseudogt_f_mae += pseudogt_f_mae_loss.item()
        pseudogt_f_psnr += pseudogt_f_psnr_loss.item()
        pseudogt_ssim += pseudogt_ssim_loss.item()
        pseudogt_total_lpips += pseudogt_lpips_loss_v.item()

        print(f'Test [{iter}|{len(test_loader)}] fMAE: {f_mae / iter} PSNR: {f_psnr / iter} fSSIM: {ssim / iter} lpips: {total_lpips / iter}')

        total_statistics = subdatasets_statistics(ILLUM_LABELS, total_statistics, illum_name, [f_mae_loss.item(), f_psnr_loss.item(), ssim_loss.item(), lpips_loss_v.item(), f"{obj_name}_{obj_rotation}_{illum_name}_{hdr_rotation}_{iter}"])
        pseudogt_total_statistics = subdatasets_statistics(ILLUM_LABELS, pseudogt_total_statistics, illum_name, [pseudogt_f_mae_loss.item(), pseudogt_f_psnr_loss.item(), pseudogt_ssim_loss.item(), pseudogt_lpips_loss_v.item(), f"{obj_name}_{obj_rotation}_{illum_name}_{hdr_rotation}_{iter}"])
    
    # average
    f_mae = f_mae / iter
    f_psnr = f_psnr / iter
    ssim = ssim / iter
    total_lpips = total_lpips / iter

    pseudogt_f_mae = pseudogt_f_mae / iter
    pseudogt_f_psnr = pseudogt_f_psnr / iter
    pseudogt_ssim = pseudogt_ssim / iter
    pseudogt_total_lpips = pseudogt_total_lpips / iter

    # save statistics
    save_subdataset_statistics(config['save_dir'], total_statistics)
    save_subdataset_statistics(config['save_dir'], pseudogt_total_statistics, which_type='Our_w_illum')
    data=open(os.path.join(config['save_dir'], "test_results.txt"),'a+')
    print(f'-----------------overall number: {iter}-----------------', file=data)
    print(f'f-MAE : {f_mae}', file=data)
    print(f'f-PSNR : {f_psnr}', file=data)
    print(f'f-SSIM : {ssim}', file=data)
    print(f'LPIPS : {total_lpips}', file=data)

    print(f'f-MAE-withillum : {pseudogt_f_mae}', file=data)
    print(f'f-PSNR-withillum : {pseudogt_f_psnr}', file=data)
    print(f'F-SSIM-withillum : {pseudogt_ssim}', file=data)
    print(f'LPIPS-withillum : {pseudogt_total_lpips}', file=data)
    