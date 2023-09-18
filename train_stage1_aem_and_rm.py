import os
import json
import shutil
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import torchvision.utils as utils

from data.dataset_full_aem_and_rm import HarmonyDatasetV2 as HarmonyDataset
from model.albedo_est_module import AlbedoEstModule
from model.rendering_module import RenderingModule
from utils import load_config, create_or_recreate_folders, rotate_hdr, render_shading, \
    do_tone_map, mask_l1, Mask_PSNR, mask_ssim, set_seed

# ------------------------- configuration -------------------------
config = load_config('configs/config_train_stage1_aem_and_rm.json')['config']
print(config)

display_folder = config['display_folder']
epoch_folder = config['epoch_folder']
train_mode = config['training']['mode']
NUM_EPOCHS = config['training']['epochs']
display_iter = config['training']['display']
TRAIN_DATA_PATH = config['train_data_path']
lr_s = config['training']['lr_s']
lr_i = config['training']['lr_i']
display_shading = config['training']['display_iter_light_dir']
record_train_iter_loss = config['training']['record_train_iter_loss']

devices = config['gpus']
os.environ["CUDA_VISIBLE_DEVICES"] = devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------- configuration -------------------------

# ------------------------- network setup -------------------------
AEM = AlbedoEstModule(channels_in=3, channels_out=3)
RM = RenderingModule(channels_in=config['chns_of_albedofeats']+9, channels_out=3)

AEM = nn.DataParallel(AEM)
AEM.to(device)

RM = nn.DataParallel(RM)
RM.to(device)

print('# albedo_est_module parameters:', sum(param.numel() for param in AEM.parameters()))
opt = optim.Adam(AEM.parameters(), lr=lr_s)

print('# rendering_module parameters:', sum(param.numel() for param in RM.parameters()))
opt2 = optim.Adam(RM.parameters(), lr=lr_s)
# ------------------------- network setup -------------------------

# ------------------------- dataset -------------------------
set_seed(config['seed'])
dataset = HarmonyDataset(TRAIN_DATA_PATH, random_flipping=True, random_rotation=True, random_crop=True)
print(f'Dataset Train: {len(dataset)}')

train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           num_workers=config['data_workers'],
                                           batch_size=config['data_batch'],
                                           shuffle=config['data_shuffle'])
# ------------------------- dataset -------------------------

# verify whether we want to continue with a training or start brand-new
if config['training']['continue']:
    # load weights
    print('------------------- Continue Training -------------------')
    weight = torch.load(f"{config['epoch_folder']}/ckpt_aem_rm{config['training']['epoch']}.pth")
    render_weight = weight['rendering']
    albedo_weight = weight['albedo']

    RM.load_state_dict(render_weight)
    AEM.load_state_dict(albedo_weight)

    # set current epoch
    epoch = config['training']['epoch']
    count_train_iter_loss = config['training']['count_train_iter_loss']
else:
    print('------------------- Starting Training -------------------')
    create_or_recreate_folders(config)
    writer = SummaryWriter(config['summary'])
    epoch = 0
    count_train_iter_loss = 1

# # ------------------------- loss functions setup -------------------------
l1_loss = nn.L1Loss(reduction='none')
l2_loss = nn.MSELoss()
mask_psnr_metric = Mask_PSNR(max_value=1)
# # ------------------------- loss functions setup -------------------------


for epoch in range(1 + epoch, NUM_EPOCHS + 1):

    iteration = 0
    l1_loss_total = 0.0
    loss_image_total = 0.0
    loss_albedo_total = 0.0
    total_mae_loss_albedo = 0.0
    total_psnr_loss_albedo = 0.0
    total_ssim_loss_albedo = 0.0
    total_mae_loss_image = 0.0
    total_psnr_loss_image = 0.0
    total_ssim_loss_image = 0.0

    AEM = AEM.train()
    RM = RM.train()

    for tensors_dic in train_loader:

        iteration = iteration + 1

        background_image = tensors_dic['background_image']
        shading_gt = tensors_dic['shading_gt']
        unharmonized_image = tensors_dic['unharmonized_input']
        gt_color_image = tensors_dic['gt_color_image']
        gt_albedo = tensors_dic['gt_albedo']
        resized_hdr = tensors_dic['resized_hdr']
        foreground_mask = tensors_dic['fore_mask']
        background_image = tensors_dic['background_image']

        if torch.cuda.is_available():
            background_image = background_image.to(device)
            shading_gt = shading_gt.to(device)
            unharmonized_image = unharmonized_image.to(device)
            gt_color_image = gt_color_image.to(device)
            gt_albedo = gt_albedo.to(device)
            resized_hdr = resized_hdr.to(device)
            foreground_mask = foreground_mask.to(device)
            background_image = background_image.to(device)

        AEM.zero_grad()
        RM.zero_grad()

        shading_image = shading_gt

        albedo_image, albedo_feat = AEM(unharmonized_image)
        rendering_input = torch.cat([albedo_feat, shading_image, background_image, unharmonized_image], 1)
        rgb = RM(rendering_input)

        # computing the loss function
        loss_albedo = l2_loss(albedo_image, gt_albedo)
        loss_image = l2_loss(rgb, gt_color_image)
        ssim_loss_albedo = mask_ssim(albedo_image, gt_albedo, foreground_mask)
        ssim_loss_image = mask_ssim(rgb, gt_color_image, foreground_mask)

        final_loss = loss_albedo + loss_image + (1 - ssim_loss_albedo) + (1 - ssim_loss_image)
        final_loss.backward()
        opt.step()
        opt2.step()

        # console print for the training in progress
        print('[%d/%d][%d], L_albedo : %f, L_image : %f' % (
            epoch, NUM_EPOCHS, iteration,
            loss_albedo.item(), loss_image.item()))

        # display
        if iteration % display_iter == 0:
            resized_hdr = do_tone_map(resized_hdr)
            resized_hdr = TF.resize(resized_hdr, (256, 256))
            background_image = TF.resize(background_image, (256, 256))

            display_data = torch.cat([shading_image,
                    albedo_image,
                    gt_albedo,
                    rgb, gt_color_image, background_image, resized_hdr], dim=0)

            utils.save_image(display_data,
                             display_folder + "/Epoch_%d Iter_%d.jpg" % (epoch, iteration),
                             nrow=shading_image.shape[0], padding=2, normalize=False)


        # loss functions summary
        loss_image_total += loss_image.item()
        loss_albedo_total += loss_albedo.item()
        total_ssim_loss_albedo += ssim_loss_albedo.item()
        total_ssim_loss_image += ssim_loss_image.item()

        if iteration % record_train_iter_loss == 0:
            writer.add_scalar('MSE-image Iter', loss_image_total / iteration, count_train_iter_loss)
            writer.add_scalar('MSE-albedo Iter', loss_albedo_total / iteration, count_train_iter_loss)
            writer.add_scalar('TrainAlbedo Iter/SSIM', total_ssim_loss_albedo / iteration, count_train_iter_loss)
            writer.add_scalar('TrainRendering Iter/SSIM', total_ssim_loss_image / iteration, count_train_iter_loss)
            count_train_iter_loss += 1

    # one epoch finished, output training loss, save model
    writer.add_scalar('MSE-image Train', loss_image_total / iteration, epoch)
    writer.add_scalar('MSE-albedobTrain', loss_albedo_total / iteration, epoch)
    writer.add_scalar('TrainAlbedo/SSIM', total_ssim_loss_albedo / iteration, epoch)
    writer.add_scalar('TrainRendering/SSIM', total_ssim_loss_image / iteration, epoch)

    dico = {
        'albedo': AEM.state_dict(),
        'rendering': RM.state_dict()
    }

    torch.save(dico, epoch_folder + '/ckpt_aem_rm%d.pth' % epoch)
