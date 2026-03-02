# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional

import kornia.augmentation as K
from kornia.augmentation import AugmentationBase2D

import utils_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffJPEG(nn.Module):
    def __init__(self, quality=50):
        super().__init__()
        self.quality = quality
    
    def forward(self, x):
        with torch.no_grad():
            img_clip = utils_img.clamp_pixel(x)
            img_jpeg = utils_img.jpeg_compress(img_clip, self.quality)
            img_gap = img_jpeg - x
            img_gap = img_gap.detach()
        img_aug = x+img_gap
        return img_aug

class RandomDiffJPEG(AugmentationBase2D):
    def __init__(self, p, low=10, high=100) -> None:
        super().__init__(p=p)
        self.diff_jpegs = [DiffJPEG(quality=qf).to(device) for qf in range(low,high,10)]

    def generate_parameters(self, input_shape: torch.Size):
        qf = torch.randint(high=len(self.diff_jpegs), size=input_shape[0:1])
        return dict(qf=qf)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        qf = params['qf']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.diff_jpegs[qf[ii]](input[ii:ii+1])
        return output

class RandomBlur(AugmentationBase2D):
    def __init__(self, blur_size, p=1) -> None:
        super().__init__(p=p)
        self.gaussian_blurs = [K.RandomGaussianBlur(kernel_size=(kk,kk), sigma= (kk*0.15 + 0.35, kk*0.15 + 0.35)) for kk in range(1,int(blur_size),2)]

    def generate_parameters(self, input_shape: torch.Size):
        blur_strength = torch.randint(high=len(self.gaussian_blurs), size=input_shape[0:1])
        return dict(blur_strength=blur_strength)

    def compute_transformation(self, input, params, flags):
        return self.identity_matrix(input)

    def apply_transform(self, input, params, *args, **kwargs):
        B, C, H, W = input.shape
        blur_strength = params['blur_strength']
        output = torch.zeros_like(input)
        for ii in range(B):
            output[ii] = self.gaussian_blurs[blur_strength[ii]](input[ii:ii+1])
        return output


class HiddenAug(nn.Module):
    """Dropout p = 0.3,Dropout p = 0.7, Cropout p = 0.3, Cropout p = 0.7, Crop p = 0.3, Crop p = 0.7, Gaussian blur σ = 2, Gaussian blur σ = 4, JPEG-drop, JPEG-mask and the Identity layer"""
    def __init__(self, img_size, p_crop=0.3, p_blur=0.3, p_jpeg=0.3, p_rot=0.3, p_color_jitter=0.3, p_res=0.3):
        super().__init__()
        augmentations = []
        hflip = K.RandomHorizontalFlip(p=1)
        augmentations += [nn.Identity(), hflip]
        if p_crop > 0:
            crop1 = int(img_size * np.sqrt(0.3))
            crop2 = int(img_size * np.sqrt(0.7))
            crop1 = K.RandomCrop(size=(crop1, crop1), p=1) # Crop 0.3   
            crop2 = K.RandomCrop(size=(crop2, crop2), p=1) # Crop 0.7       
            augmentations += [crop1, crop2]
        if p_res > 0:
            res1 = int(img_size * np.sqrt(0.3))
            res2 = int(img_size * np.sqrt(0.7))
            res1 = K.RandomResizedCrop(size=(res1, res1), scale=(1.0,1.0), p=1) # Resize 0.3
            res2 = K.RandomResizedCrop(size=(res2, res2), scale=(1.0,1.0), p=1) # Resize 0.7
            augmentations += [res1, res2]
        if p_blur > 0:
            blur1 = K.RandomGaussianBlur(kernel_size=(11,11), sigma= (2.0, 2.0), p=1) # Gaussian blur σ = 2
            # blur2 = K.RandomGaussianBlur(kernel_size=(25,25), sigma= (4.0, 4.0), p=1) # Gaussian blur σ = 4
            augmentations += [blur1]
            # augmentations += [blur1, blur2]
        if p_jpeg > 0:
            diff_jpeg1 = DiffJPEG(quality=50)  # JPEG50
            diff_jpeg2 = DiffJPEG(quality=80)  # JPEG80
            augmentations += [diff_jpeg1, diff_jpeg2]
        if p_rot > 0:
            aff1 = K.RandomAffine(degrees=(-10,10), p=1)
            aff2 = K.RandomAffine(degrees=(90,90), p=1)
            aff3 = K.RandomAffine(degrees=(-90,-90), p=1)
            augmentations += [aff1]
            augmentations += [aff2, aff3]
        if p_color_jitter > 0:
            jitter1 = K.ColorJiggle(brightness=(1.5, 1.5), contrast=0, saturation=0, hue=0, p=1)
            jitter2 = K.ColorJiggle(brightness=0, contrast=(1.5, 1.5), saturation=0, hue=0, p=1)
            jitter3 = K.ColorJiggle(brightness=0, contrast=0, saturation=(1.5,1.5), hue=0, p=1)
            jitter4 = K.ColorJiggle(brightness=0, contrast=0, saturation=0, hue=(0.25, 0.25), p=1)
            augmentations += [jitter1, jitter2, jitter3, jitter4]
        self.hidden_aug = K.AugmentationSequential(*augmentations, random_apply=1).to(device)
        
    def forward(self, x):
        return self.hidden_aug(x)

class KorniaAug(nn.Module):
    def __init__(self, degrees=30, crop_scale=(0.2, 1.0), crop_ratio=(3/4, 4/3), blur_size=17, color_jitter=(1.0, 1.0, 1.0, 0.3), diff_jpeg=10,
                p_crop=0.5, p_aff=0.5, p_blur=0.5, p_color_jitter=0.5, p_diff_jpeg=0.5, 
                cropping_mode='slice', img_size=224
            ):
        super(KorniaAug, self).__init__()
        self.jitter = K.ColorJitter(*color_jitter, p=p_color_jitter).to(device)
        # self.jitter = K.RandomPlanckianJitter(p=p_color_jitter).to(device)
        self.aff = K.RandomAffine(degrees=degrees, p=p_aff).to(device)
        self.crop = K.RandomResizedCrop(size=(img_size,img_size),scale=crop_scale,ratio=crop_ratio, p=p_crop, cropping_mode=cropping_mode).to(device)
        self.hflip = K.RandomHorizontalFlip().to(device)
        self.blur = RandomBlur(blur_size, p_blur).to(device)
        self.diff_jpeg = RandomDiffJPEG(p=p_diff_jpeg, low=diff_jpeg).to(device)
    
    def forward(self, input):
        input = self.diff_jpeg(input)
        input = self.aff(input)
        input = self.crop(input)
        input = self.blur(input)
        input = self.jitter(input)
        input = self.hflip(input)
        return input

from augly.image import functional as aug_functional
normalize_img = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]) # Normalize (x - mean) / std
unnormalize_img = transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711],
                                        std=[1 / 0.26862954, 1 / 0.26130258, 1 / 0.27577711]) # Unnormalize (x * std) + mean

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)

def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))


def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)
