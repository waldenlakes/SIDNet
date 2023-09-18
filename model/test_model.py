from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from model.illlum_module import IlluminationModule
from model.shading_bases_module import ShadingBasesModule
from model.albedo_est_module import AlbedoEstModule
from model.rendering_module import RenderingModule

class SIDNet(nn.Module):
    def __init__(self, ckpts=None, chns_of_albedofeats = 16, chns_of_shadingbases=32, input_size_of_biem=256, crop_info_of_rm=[120, 480]):
        super(SIDNet, self).__init__()
        self.chns_of_albedofeats = chns_of_albedofeats
        self.chns_of_shadingbases = chns_of_shadingbases
        self.input_size_of_biem = input_size_of_biem
        self.crop_info = crop_info_of_rm

        self.sbm = ShadingBasesModule(channels_in=3, channels_out=self.chns_of_shadingbases)
        self.aem = AlbedoEstModule(channels_in=3, channels_out=3)
        self.rm = RenderingModule(channels_in=self.chns_of_albedofeats+9, channels_out=3)
        self.iem = IlluminationModule(channels_in=3, channels_out=self.chns_of_shadingbases*3)
        self.biem = IlluminationModule(channels_in=3, channels_out=self.chns_of_shadingbases*3)

        if ckpts is not None:
            self.init_model(ckpts)

    def init_model(self, ckpts):
        weight = torch.load(ckpts['ckpt_aem_rm'])
        render_weight = weight['rendering']
        albedo_weight = weight['albedo']
        self.rm.load_state_dict(self.init_weights(render_weight), strict=True)
        self.aem.load_state_dict(self.init_weights(albedo_weight), strict=True)
        self.sbm.load_state_dict(self.init_weights(torch.load(ckpts['ckpt_sbm'])), strict=True)
        self.biem.load_state_dict(self.init_weights(torch.load(ckpts['ckpt_biem'])), strict=True)
        self.iem.load_state_dict(self.init_weights(torch.load(ckpts['ckpt_iem'])), strict=True)

        rendering_net_params = sum(param.numel() for param in self.rm.parameters())
        print('# rendering_module parameters:', rendering_net_params)
        total_params = rendering_net_params
        albedo_net_params = sum(param.numel() for param in self.aem.parameters())
        print('# albedo_est_module parameters:', albedo_net_params)
        total_params += albedo_net_params
        shading_net_params = sum(param.numel() for param in self.sbm.parameters())
        print('# shading_bases_module parameters:', shading_net_params)
        total_params += shading_net_params
        illum_net_params = sum(param.numel() for param in self.biem.parameters())
        print('# bg_illum_est_module parameters:', illum_net_params)
        total_params += illum_net_params
        print('# total parameters of SIDNet:', total_params)

        print('All modules weights loaded!')

    def init_weights(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v

        return new_state_dict

    def render_shading(self, shading_bases, coeff):
        expanded_fs = torch.unsqueeze(shading_bases, 2).repeat(1, 1, 3, 1, 1)
        rendered_shading = torch.sum((coeff * expanded_fs), dim=1)
        return rendered_shading

    def forward(self, raw_unharmonized_img, background_image, gt_illum_map=None):
        shading_bases = self.sbm(raw_unharmonized_img)
        illum_descriptor = self.biem(TF.resize(background_image, (self.input_size_of_biem, self.input_size_of_biem)))
        shading_image = self.render_shading(shading_bases, illum_descriptor)
        
        _, albedo_feat = self.aem(raw_unharmonized_img)
        crop_albedo_feat, crop_shading_image, crop_raw_unharmonized_img = albedo_feat[...,self.crop_info[0]:self.crop_info[0]+self.crop_info[1]], \
            shading_image[...,self.crop_info[0]:self.crop_info[0]+self.crop_info[1]], raw_unharmonized_img[...,self.crop_info[0]:self.crop_info[0]+self.crop_info[1]]
        resized_background_image = TF.resize(background_image, [crop_albedo_feat.shape[2], crop_albedo_feat.shape[3]])

        rendering_input = torch.cat([crop_albedo_feat, crop_shading_image, resized_background_image, crop_raw_unharmonized_img], 1)
        raw_harmonized_img = self.rm(rendering_input)

        if gt_illum_map is None:
            return raw_harmonized_img
        else:
            pseudo_gt_illum_descriptor = self.iem(gt_illum_map)
            pseudo_gt_shading_image = self.render_shading(shading_bases, pseudo_gt_illum_descriptor)
            crop_pseudo_gt_shading_image = pseudo_gt_shading_image[...,self.crop_info[0]:self.crop_info[0]+self.crop_info[1]]
            pseudogt_rendering_input = torch.cat([crop_albedo_feat, crop_pseudo_gt_shading_image, resized_background_image, crop_raw_unharmonized_img], 1)
            raw_pseudogt_harmonized_img = self.rm(pseudogt_rendering_input)

            return raw_harmonized_img, raw_pseudogt_harmonized_img
