import os
import glob
import numpy as np
import random
import cv2
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class HarmonyDatasetV2(Dataset):
    def __init__(self, data_dir, mode='train', random_flipping=False, random_rotation=False,
                 hflip=False):
        super(HarmonyDatasetV2, self).__init__()
        self.random_flipping = random_flipping
        self.random_rotation = random_rotation
        self.hflip = hflip

        self.data_dir = data_dir
        self.hdr_dir = os.path.join(self.data_dir, 'illum_maps')
        self.objects_dir = os.path.join(self.data_dir, 'objects')

        # ----------------------------------------------- hdr

        self.hdr_images_list = glob.glob(os.path.join(self.hdr_dir, '*.exr'))

        
        self.images_gt_images_paths = []
        train_list_path = os.path.join(self.data_dir, f'{mode}_list.txt')
        for did in open(train_list_path):
            did = did.strip()
            self.images_gt_images_paths.append(os.path.join(self.objects_dir, did))

    def load_hdr(self, path):
        hdr = cv2.imread(path, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR)
        hdr = np.flip(hdr, -1)
        hdr = np.clip(hdr, 0, None)
        return hdr

    def get_unharmonized_input2(self, indexed_image, images_gt_images_paths):

        # remove the indexed image ground truth from the list of all images
        new_paths = images_gt_images_paths.copy()
        new_paths.remove(indexed_image)

        # filter all the list such that only images that have the same object angle as that
        # of the ground truth image are retained
        unhamonized_inputs = []
        gt_angle = indexed_image.split('/')[-1].split('.')[0].split('_')[1]
        gt_name = indexed_image.split('/')[-4]

        for i in range(len(new_paths)):
            current_paths = new_paths[i]
            current_angle = current_paths.split('/')[-1].split('.')[0].split('_')[1]
            current_name = current_paths.split('/')[-4]

            if current_angle == gt_angle and current_name == gt_name:
                unhamonized_inputs.append(current_paths)

        # randomly select an image from the filtered list and return it
        unharmonized = unhamonized_inputs[random.randint(0, len(unhamonized_inputs) - 1)]
        return unharmonized

    def __getitem__(self, index):

        # obtain the current path and object rotation angle,
        # illumination rotation angle and hdr illumination name
        # indexed_shading = self.shading_gt_images_paths[index]
        indexed_image = self.images_gt_images_paths[index]
        indexed_shading = indexed_image.replace('images', 'shading')

        indexed_object_name = indexed_shading.split('/')[-4]
        indexed_obj_angle = indexed_shading.split('/')[-1].split('.')[0].split('_')[1]
        hdr_name = indexed_shading.split('/')[-2]
        illumination_rotation = int(indexed_shading.split('/')[-1].split('.')[0].split('_')[2])

        mask_fore_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'mask_foreground'))
        gt_albedo_dir = os.path.join(self.objects_dir, os.path.join(indexed_object_name, 'albedo'))

        current_unharmonized_path = self.get_unharmonized_input2(indexed_image, self.images_gt_images_paths)
        current_mask_fore_path = glob.glob(os.path.join(mask_fore_dir, f'*_{indexed_obj_angle}.bmp'))[0]
        current_albedo_path = glob.glob(os.path.join(gt_albedo_dir, f'*_{indexed_obj_angle}.png'))[0]

        # read images
        shading_image = Image.open(indexed_shading)
        foreground_mask = Image.open(current_mask_fore_path)
        color_image = Image.open(indexed_image)
        albedo_image = Image.open(current_albedo_path)
        unharmonized_image = Image.open(current_unharmonized_path)

        if self.random_flipping:

            # Random horizontal flipping
            if random.random() > 0.5:
                foreground_mask = TF.hflip(foreground_mask)
                color_image = TF.hflip(color_image)
                albedo_image = TF.hflip(albedo_image)
                shading_image = TF.hflip(shading_image)

            # Random vertical flipping
            if random.random() > 0.5:
                foreground_mask = TF.vflip(foreground_mask)
                color_image = TF.vflip(color_image)
                albedo_image = TF.vflip(albedo_image)
                shading_image = TF.hflip(shading_image)

        if self.random_rotation:
            # angles=[-90, 0, 90, 180]
            # angle = random.choice(angles)
            angle = random.randint(-180, 180)
            foreground_mask = TF.rotate(foreground_mask, angle)
            color_image = TF.rotate(color_image, angle)
            albedo_image = TF.rotate(albedo_image, angle)
            shading_image = TF.rotate(shading_image, angle)

        shading_image = np.array(shading_image, dtype=np.float32)[:, :, :3] / 255.0
        foreground_mask = np.array(foreground_mask, dtype=np.float32)[:, :, 0] / 255.0
        unharmonized_image = np.array(unharmonized_image, dtype=np.float32)[:, :, :3] / 255.0
        color_image = np.array(color_image, dtype=np.float32)[:, :, :3] / 255.0
        albedo_image = np.array(albedo_image, dtype=np.float32)[:, :, :3] / 255.0


        foreground_mask[foreground_mask < 0.5] = 0.0
        foreground_mask[foreground_mask > 0.5] = 1.0

        # masking the depth map and the normal map
        masked_shading = np.expand_dims(foreground_mask, 2) * shading_image
        masked_albedo = np.expand_dims(foreground_mask, 2) * albedo_image
        masked_color = np.expand_dims(foreground_mask, 2) * color_image
        masked_unharmonized = np.expand_dims(foreground_mask, 2) * unharmonized_image

        # based on the shading gt illumination name pick the corresponding hdr image
        hdr_path = glob.glob(os.path.join(os.path.join(self.hdr_dir, f'{hdr_name}*')))[0]
        hdr_image = self.load_hdr(hdr_path)


        # rotation used to set the hdr image so that it corresponds with blender
        hdr_image = rotate_hdr(hdr_image, 180)
        unrotated = hdr_image.copy()

        # applying the rotation that corresponds to the shading image
        rotated_hdr = rotate_hdr(hdr_image, illumination_rotation)
        original_hdr = rotated_hdr.copy()


        # convert all relevant tensors into pytorch tensors
        fore_mask_tensor = torch.unsqueeze(torch.from_numpy(foreground_mask), 0)
        shading_gt_tensor = torch.from_numpy(masked_shading).permute(2, 0, 1)
        albedo_gt_tensor = torch.from_numpy(masked_albedo).permute(2, 0, 1)
        color_gt_tensor = torch.from_numpy(masked_color).permute(2, 0, 1)
        hdr_tensor = torch.from_numpy(rotated_hdr).permute(2, 0, 1)
        unrotated_tensor = torch.from_numpy(unrotated).permute(2, 0, 1)
        unharmonized_tensor = torch.from_numpy(masked_unharmonized).permute(2, 0, 1)

        if self.hflip:
            if random.random() > 0.5:
                hdr_tensor = TF.hflip(hdr_tensor)
                shading_gt_tensor = TF.hflip(shading_gt_tensor)
                fore_mask_tensor = TF.hflip(fore_mask_tensor)
                unharmonized_tensor = TF.hflip(unharmonized_tensor)
                color_gt_tensor = TF.hflip(color_gt_tensor)

        return {
            'shading_gt': shading_gt_tensor,
            'fore_mask': fore_mask_tensor,
            'hdr': hdr_tensor,
            'ohdr': original_hdr,
            'unrotated': unrotated_tensor,
            'gt_albedo': albedo_gt_tensor,
            'gt_color_image': color_gt_tensor,
            'unharmonized_input': unharmonized_tensor,
            'name': f'{indexed_object_name}_{indexed_obj_angle}_{hdr_name}_{illumination_rotation}',
        }

    def __len__(self):
        return len(self.images_gt_images_paths)

def rotate_hdr(image, angle):
    # angle : 0 - 360 for rotation angle
    H, W, C = image.shape
    width = (angle / 360.0) * W

    front = image[:, 0:W - int(width)]
    back = image[:, W - int(width):]

    rotated = np.concatenate((back, front), 1)
    return rotated




if __name__ == '__main__':
    data_dir = ''

