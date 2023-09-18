import os
import os.path as osp
import time
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

from data.dataset_full_biem import HarmonyDatasetV2 as HarmonyDataset
from model.shading_bases_module import ShadingBasesModule
from model.illlum_module import IlluminationModule
from utils import load_config, create_or_recreate_folders, rotate_hdr, render_shading, \
    do_tone_map, mask_l1, Mask_PSNR, mask_ssim, set_seed


# ------------------------- configuration -------------------------
config = load_config('configs/config_train_stage2_biem.json')['config']
print(config)

display_folder = config['display_folder']
epoch_folder = config['epoch_folder']
train_mode = config['training']['mode']
NUM_EPOCHS = config['training']['epochs']
display_iter = config['training']['display']
data_path_train = config['data_path_train']
lr_s = config['training']['lr_s']
lr_i = config['training']['lr_i']
lr_d = config['training']['lr_d']
display_shading = config['training']['display_iter_light_dir']
record_train_iter_loss = config['training']['record_train_iter_loss']

devices = config['gpus']
os.environ["CUDA_VISIBLE_DEVICES"] = devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------- configuration -------------------------


# ------------------------- network setup -------------------------
SBM = ShadingBasesModule(channels_in=3, channels_out=config['chns_of_shadingbases'])
print('# shading_bases_module parameters:', sum(param.numel() for param in SBM.parameters()))
opt = optim.Adam(SBM.parameters(), lr=lr_s)

IEM = IlluminationModule(channels_in=3, channels_out=config['chns_of_shadingbases']*3)
print('# illum_encoder_module parameters:', sum(param.numel() for param in IEM.parameters()))
opt2 = optim.Adam(IEM.parameters(), lr=lr_i)

BIEM = IlluminationModule(channels_in=3, channels_out=config['chns_of_shadingbases']*3)
print('# bg_illum_est_module parameters:', sum(param.numel() for param in BIEM.parameters()))
opt3 = optim.Adam(BIEM.parameters(), lr=lr_d)

SBM = nn.DataParallel(SBM)
SBM.to(device)

IEM = nn.DataParallel(IEM)
IEM.to(device)

BIEM = nn.DataParallel(BIEM)
BIEM.to(device)
# ------------------------- network setup -------------------------


# ------------------------- dataset -------------------------
set_seed(config['seed'])
train_dataset = HarmonyDataset(data_path_train, hflip=True)
print(f'Dataset Train : {len(train_dataset)}')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           num_workers=config['data_workers'],
                                           batch_size=config['data_batch'],
                                           shuffle=config['data_shuffle'])

# ------------------------- dataset -------------------------

# verify whether we want to continue with a training or start brand-new
if config['training']['continue']:
    # load weights
    print('------------------- Continue Training -------------------')
    weight = torch.load(f"{config['epoch_folder']}/ckpt_sbm{config['training']['epoch']}.pth")
    weight_illum = torch.load(f"{config['epoch_folder']}/ckpt_iem{config['training']['epoch']}.pth")
    weight_biem = torch.load(f"{config['epoch_folder']}/ckpt_biem{config['training']['epoch']}.pth")

    SBM.load_state_dict(weight)
    IEM.load_state_dict(weight_illum)
    BIEM.load_state_dict(weight_biem)

    # set current epoch
    epoch = config['training']['epoch']
    count_train_iter_loss = config['training']['count_train_iter_loss']
else:
    print('------------------- Starting Training -------------------')
    create_or_recreate_folders(config)
    writer = SummaryWriter(config['summary'])
    epoch = 0
    count_train_iter_loss = 1

    weight = torch.load(config['stage1_ckpts']['ckpt_sbm'])
    weight_illum = torch.load(config['stage1_ckpts']['ckpt_iem'])

    SBM.load_state_dict(weight, strict=True)
    IEM.load_state_dict(weight_illum, strict=True)

# ------------------------- loss functions setup -------------------------
l1_loss = nn.L1Loss(reduction='none')
l2_loss = nn.MSELoss()
mask_psnr_metric = Mask_PSNR(max_value=1)
# ------------------------- loss functions setup -------------------------


for epoch in range(1 + epoch, NUM_EPOCHS + 1):

    iteration = 0
    l1_loss_total = 0.0
    psnr_loss_total = 0.0
    ssim_loss_total = 0.0
    illum_descriptor_loss_total = 0.0
    shading_loss_total = 0.0
    shading_ssim_loss_total = 0.0
    SBM = SBM.train()
    IEM = IEM.train()
    BIEM = BIEM.train()

    for tensors_dic in train_loader:
        start_time = time.time()
        iteration = iteration + 1

        shading_gt = tensors_dic['shading_gt']
        foreground_mask = tensors_dic['fore_mask']
        hdr_image = tensors_dic['hdr']
        ohdr = tensors_dic['ohdr']
        unrotated_hdr = tensors_dic['unrotated']
        unharmonized_input = tensors_dic['unharmonized_input']
        background_image = tensors_dic['background_image']

        if torch.cuda.is_available():
            shading_gt = shading_gt.to(device)
            foreground_mask = foreground_mask.to(device)
            hdr_image = hdr_image.to(device)
            unharmonized_input = unharmonized_input.to(device)
            background_image = background_image.to(device)

        SBM.zero_grad()
        IEM.zero_grad()
        BIEM.zero_grad()

        # ------------------------- stage1 forward pass -------------------------
        shading_bases = SBM(unharmonized_input)

        # illumination encoder module forward pass
        illum_descriptor = IEM(hdr_image)
        shading_image = render_shading(shading_bases, illum_descriptor)

        # computing the loss function of shading bases module
        loss = mask_l1(shading_image, shading_gt, foreground_mask, l1_loss)
        psnr_loss = mask_psnr_metric(shading_image, shading_gt, foreground_mask, max_value=1)
        ssim_loss = mask_ssim(shading_image, shading_gt, foreground_mask)

        stage1_loss = loss + 1 - ssim_loss
        stage1_loss.backward()
        opt.step()
        opt2.step()
        # ------------------------- stage2 forward pass -------------------------
        predicted_illum_descriptor = BIEM(background_image)

        shading_ldr = render_shading(shading_bases.detach(), predicted_illum_descriptor)

        # computing the loss function of BIEM
        illum_descriptor_loss = l2_loss(predicted_illum_descriptor, illum_descriptor.detach())
        shading_loss = mask_l1(shading_ldr, shading_gt, foreground_mask, l1_loss)
        shading_ssim_loss = mask_ssim(shading_ldr, shading_gt, foreground_mask)

        stage2_loss = illum_descriptor_loss + shading_loss + (1 - shading_ssim_loss)
        stage2_loss.backward()
        opt3.step()

        final_loss = stage1_loss + stage2_loss

        # console print for the training in progress
        elapsed_time = time.time() - start_time
        print('[%d/%d][%d], L1 : %f Time: %.3f ' % (
            epoch, NUM_EPOCHS, iteration,
            loss.item(), elapsed_time / config['data_batch']))

        # display
        if iteration % display_iter == 0:
            resized_hdr = TF.resize(do_tone_map(hdr_image), (shading_gt.shape[2], shading_gt.shape[3]))
            resized_background_image = TF.resize(background_image, (shading_gt.shape[2], shading_gt.shape[3]))
            display_data = torch.cat([unharmonized_input, shading_image, shading_gt, resized_hdr, shading_ldr, resized_background_image], dim=0)
            utils.save_image(display_data,
                             display_folder + "/Epoch_%d Iter_%d.jpg" % (epoch, iteration),
                             nrow=unharmonized_input.shape[0], padding=2, normalize=False)

        # output images with illumination at different angles for display
        if iteration % display_shading == 0:
            images = []
            angles = [45, 90, 135, 180]
            for angle in angles:
                rotated_hdr = rotate_hdr(hdr_image, angle)
                with torch.no_grad():
                    shading_bases = SBM(unharmonized_input)
                    illum_descriptor = IEM(rotated_hdr)
                    shading_image = render_shading(shading_bases, illum_descriptor)
                    images.append(shading_image)

            display_data_rotation = torch.cat(images, dim=0)
            utils.save_image(display_data_rotation,
                             display_folder + "/Rotation_Epoch_%d Iter_%d.jpg" % (epoch, iteration),
                             nrow=unharmonized_input.shape[0], padding=2, normalize=False)

        # loss functions summary
        l1_loss_total += loss.item()
        psnr_loss_total += psnr_loss.item()
        ssim_loss_total += ssim_loss.item()
        illum_descriptor_loss_total += illum_descriptor_loss.item()
        shading_loss_total += shading_loss.item()
        shading_ssim_loss_total += shading_ssim_loss.item()

        if iteration % record_train_iter_loss == 0:
            writer.add_scalar('TrainIter/stage1_MAE', l1_loss_total / iteration, count_train_iter_loss)
            writer.add_scalar('TrainIter/stage1_PSNR', psnr_loss_total / iteration, count_train_iter_loss)
            writer.add_scalar('TrainIter/stage1_SSIM', ssim_loss_total / iteration, count_train_iter_loss)
            writer.add_scalar('TrainIter/stage2_MAE', illum_descriptor_loss_total / iteration, count_train_iter_loss)
            writer.add_scalar('TrainIter/stage2_PSNR', shading_loss_total / iteration, count_train_iter_loss)
            writer.add_scalar('TrainIter/stage2_SSIM', shading_ssim_loss_total / iteration, count_train_iter_loss)
            count_train_iter_loss += 1

    writer.add_scalar('Train/stage1_MAE', l1_loss_total / iteration, epoch)
    writer.add_scalar('Train/stage1_PSNR', psnr_loss_total / iteration, epoch)
    writer.add_scalar('Train/stage1_SSIM', ssim_loss_total / iteration, epoch)
    writer.add_scalar('Train/stage2_MAE', illum_descriptor_loss_total / iteration, epoch)
    writer.add_scalar('Train/stage2_PSNR', shading_loss_total / iteration, epoch)
    writer.add_scalar('Train/stage2_SSIM', shading_ssim_loss_total / iteration, epoch)
    torch.save(SBM.state_dict(), epoch_folder + '/ckpt_sbm%d.pth' % epoch)
    torch.save(IEM.state_dict(), epoch_folder + '/ckpt_iem%d.pth' % epoch)
    torch.save(BIEM.state_dict(), epoch_folder + '/ckpt_biem%d.pth' % epoch)
