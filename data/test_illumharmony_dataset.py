import os
import os.path as osp
import glob
import random
import shutil
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.utils as utils


class TestIllumHarmonyDataset(Dataset):
    def __init__(self, data_dir):
        super(TestIllumHarmonyDataset, self).__init__()
        self.data_dir = data_dir
        self.background_images_dir = os.path.join(self.data_dir, 'background_imgs')
        self.placement_masks_dir = os.path.join(self.data_dir, 'placement_binary_masks')
        self.hdr_dir = os.path.join(self.data_dir, 'illum_maps')
        self.objects_dir = os.path.join(self.data_dir, 'objects')

        self.unharmonized_img_paths = []
        self.gt_img_paths = []
        self.placement_points = []
        test_list_path = os.path.join(data_dir, "test_list.txt")
        for did in open(test_list_path):
            did = did.strip()
            self.unharmonized_img_paths.append(os.path.join(self.objects_dir, did.split(' ')[0]))
            self.gt_img_paths.append(os.path.join(self.objects_dir, did.split(' ')[1]))
            self.placement_points.append(did.split(' ')[2])

    def load_hdr(self, path):
        hdr = cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR)
        hdr = np.flip(hdr, -1)
        hdr = np.clip(hdr, 0, None)
        return hdr

    def rotate_hdr(self, image, angle):
        # angle : 0 - 360 for rotation angle
        H, W, C = image.shape
        width = (angle / 360.0) * W

        front = image[:, 0:W - int(width)]
        back = image[:, W - int(width):]

        rotated = np.concatenate((back, front), 1)
        return rotated

    def load_npy(self, npy_path):
        return np.load(npy_path)

    def get_corresponding_background(self, indexed_image):
        image_name = indexed_image.split('/')[-1]
        scene_name = indexed_image.split('/')[-2]
        hdr_theta = image_name.split('_')[2].split('.')[0]

        background_images = os.path.join(self.background_images_dir, f'{scene_name}_2.2_{hdr_theta}_0_90.png')
        return background_images

    def __getitem__(self, index):

        # obtain the current paths and object rotation angle,
        # illumination rotation angle and hdr illumination name
        indexed_image = self.gt_img_paths[index]
        indexed_background = self.get_corresponding_background(indexed_image)

        indexed_object_name = indexed_image.split('/')[-4]
        indexed_obj_angle = indexed_image.split('/')[-1].split('.')[0].split('_')[1]
        hdr_name = indexed_image.split('/')[-2]
        illumination_rotation = int(indexed_image.split('/')[-1].split('.')[0].split('_')[2])

        mask_fore_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'mask_foreground'))
        gt_albedo_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'albedo'))

        indexed_shading = indexed_image.replace("images", "shading")
        current_unharmonized_path = self.unharmonized_img_paths[index]
        current_mask_fore_path = glob.glob(os.path.join(mask_fore_dir, f'*_{indexed_obj_angle}.bmp'))[0]
        current_back_palcementmask_path = os.path.join(self.placement_masks_dir, hdr_name, f'background_2.2_{illumination_rotation}_0_90.png')
        current_albedo_path = glob.glob(os.path.join(gt_albedo_dir, f'*_{indexed_obj_angle}.png'))[0]

        # read images
        unharmonized_image = Image.open(current_unharmonized_path)
        foreground_mask = Image.open(current_mask_fore_path)
        color_image = Image.open(indexed_image)
        albedo_image = Image.open(current_albedo_path)
        shading_image = Image.open(indexed_shading)
        background_image = Image.open(indexed_background)
        background_mask = Image.open(current_back_palcementmask_path)

        shading_image = np.array(shading_image, dtype=np.float32)[:, :, :3] / 255.0
        foreground_mask = np.array(foreground_mask, dtype=np.float32)[:, :, 0] / 255.0
        unharmonized_image = np.array(unharmonized_image, dtype=np.float32)[:, :, :3] / 255.0
        color_image = np.array(color_image, dtype=np.float32)[:, :, :3] / 255.0
        albedo_image = np.array(albedo_image, dtype=np.float32)[:, :, :3] / 255.0
        background_image = np.array(background_image, dtype=np.float32)[:, :, :3] / 255.0
        background_mask = np.array(background_mask, dtype=np.float32) / 255.0

        alpha_foreground_mask = foreground_mask.copy()
        foreground_mask[foreground_mask < 0.5] = 0.0
        foreground_mask[foreground_mask > 0.5] = 1.0

        # masking foreground images
        masked_shading = np.expand_dims(foreground_mask, 2) * shading_image
        masked_albedo = np.expand_dims(foreground_mask, 2) * albedo_image
        masked_color = np.expand_dims(foreground_mask, 2) * color_image
        masked_unharmonized = np.expand_dims(foreground_mask, 2) * unharmonized_image

        # based on the shading gt illumination name pick the corresponding hdr image
        hdr_path = glob.glob(os.path.join(self.hdr_dir, f'{hdr_name}*'))[0]
        hdr_image = self.load_hdr(hdr_path)


        # rotation used to set the hdr image so that it corresponds with blender
        hdr_image = self.rotate_hdr(hdr_image, 180)
        unrotated = hdr_image.copy()

        # applying the rotation that corresponds to the shading image
        rotated_hdr = self.rotate_hdr(hdr_image, illumination_rotation)
        original_hdr = rotated_hdr.copy()

        # randomly choosing a point in the placement mask
        placement_point = {"pt_h":int(self.placement_points[index].split('_')[0]), "pt_w":int(self.placement_points[index].split('_')[1])}

        # convert all relevant tensors into pytorch tensors
        alpha_foreground_mask_tensor = torch.unsqueeze(torch.from_numpy(alpha_foreground_mask), 0)
        fore_mask_tensor = torch.unsqueeze(torch.from_numpy(foreground_mask), 0)
        shading_gt_tensor = torch.from_numpy(masked_shading).permute(2, 0, 1)
        albedo_gt_tensor = torch.from_numpy(masked_albedo).permute(2, 0, 1)
        color_gt_tensor = torch.from_numpy(masked_color).permute(2, 0, 1)
        background_tensor = torch.from_numpy(background_image).permute(2, 0, 1)
        unharmonized_tensor = torch.from_numpy(masked_unharmonized).permute(2, 0, 1)
        full_unharmonized_tensor = torch.from_numpy(unharmonized_image).permute(2, 0, 1)

        hdr_tensor = torch.from_numpy(rotated_hdr).permute(2, 0, 1)
        unrotated_tensor = torch.from_numpy(unrotated).permute(2, 0, 1)

        back_mask_tensor = torch.unsqueeze(torch.from_numpy(background_mask), 0)
        full_color_gt_tensor = torch.from_numpy(color_image).permute(2, 0, 1)

        return {
            'masked_unharmonized_input': unharmonized_tensor,
            'full_unharmonized_input': full_unharmonized_tensor,
            'gt_color_image': color_gt_tensor,
            'full_gt_color_image': full_color_gt_tensor,
            'bin_fore_mask': fore_mask_tensor,
            'alpha_foreground_mask': alpha_foreground_mask_tensor,
            'background_image': background_tensor,
            'back_mask': back_mask_tensor,
            'gt_albedo': albedo_gt_tensor,
            'shading_gt': shading_gt_tensor,
            'hdr': hdr_tensor,
            'ohdr': original_hdr,
            'unrotated': unrotated_tensor,
            'placement_config': placement_point,
            'hdr_name': hdr_name,
            "gt_img_path": indexed_image,
        }

    def __len__(self):
        return len(self.gt_img_paths)

if __name__ == '__main__':
    data_dir = ''
