# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import shutil
import tqdm
import cv2
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms

from pytorch_fid.fid_score import InceptionV3, calculate_frechet_distance, \
    compute_statistics_of_path, calculate_activation_statistics, get_activations
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from run_image_editing import get_logger
from brisque import BRISQUE

class AverageMeter(object):
    # Computes and stores the average and current value.
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def prcd(path1, path2, device='cuda:0', k=3):
    # load inception
    inception = InceptionV3([3]).to(device)
    inception.eval()

    files_real = []
    for subset in ['set_A', 'set_B', 'set_C']:
        subpath = os.path.join(path2, subset)
        files_real += [os.path.join(subpath, f) for f in os.listdir(subpath)]
    files_real = sorted(files_real)
    embedding_real = get_activations(files_real, inception, device=device)
    embedding_real = torch.from_numpy(embedding_real).to(device)

    files_fake = [os.path.join(path1, f) for f in os.listdir(path1)]
    embedding_fake = get_activations(files_fake, inception, device=device)
    embedding_fake = torch.from_numpy(embedding_fake).to(device)

    pair_dist_real = torch.cdist(embedding_real, embedding_real, p=2)
    pair_dist_real = torch.sort(pair_dist_real, dim=1, descending=False)[0]
    pair_dist_fake = torch.cdist(embedding_fake, embedding_fake, p=2)
    pair_dist_fake = torch.sort(pair_dist_fake, dim=1, descending=False)[0]
    radius_real = pair_dist_real[:, k]
    radius_fake = pair_dist_fake[:, k]

    # Compute precision
    distances_fake_to_real = torch.cdist(embedding_fake, embedding_real, p=2)
    min_dist_fake_to_real, nn_real = distances_fake_to_real.min(dim=1)
    precision = (min_dist_fake_to_real <= radius_real[nn_real]).float().mean()
    precision = precision.cpu().item()

    # Compute recall
    distances_real_to_fake = torch.cdist(embedding_real, embedding_fake, p=2)
    min_dist_real_to_fake, nn_fake = distances_real_to_fake.min(dim=1)
    recall = (min_dist_real_to_fake <= radius_fake[nn_fake]).float().mean()
    recall = recall.cpu().item()

    # Compute density
    num_samples = distances_fake_to_real.shape[0]
    sphere_counter = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0).mean()
    density = sphere_counter / k
    density = density.cpu().item()

    # Compute coverage
    num_neighbors = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0)
    coverage = (num_neighbors > 0).float().mean()
    coverage = coverage.cpu().item()

    return precision, recall, density, coverage

def cached_fid(path1, path2, batch_size=32, device='cuda:0', dims=2048, num_workers=10):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    files = []
    for subset in ['set_A', 'set_B', 'set_C']:
        subpath = os.path.join(path2, subset)
        files += [os.path.join(subpath, f) for f in os.listdir(subpath)]
    files = sorted(files)
    m2, s2 = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)
    m1, s1 = compute_statistics_of_path(str(path1), model, batch_size, dims, device, num_workers)    
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def brisques(image_path):
    obj = BRISQUE(url=False)
    prompt_score = 0
    count = 0
    for img_name in os.listdir(image_path):
        if "png" in img_name or "jpg" in img_name:
            img_path = os.path.join(image_path, img_name)
            img = Image.open(img_path)
            brisque_score = obj.score(img)
            print(brisque_score)
            prompt_score += brisque_score
            count += 1
    return prompt_score/count

def eval_one(args, logger):
    brisque_wm = brisques(args.out_wm_path)
    brisque_cl = brisques(args.out_cl_path)
    logger.info(f'BRISQUE: cl {brisque_cl}, wm {brisque_wm}')

    fid_wm = cached_fid(args.out_wm_path, args.orig_path)
    fid_cl = cached_fid(args.out_cl_path, args.orig_path)
    logger.info(f'FID: cl {fid_cl}, wm {fid_wm}')

    p_wm, r_wm, c_wm, d_wm = prcd(args.out_wm_path, args.orig_path)
    p_cl, r_cl, c_cl, d_cl = prcd(args.out_cl_path, args.orig_path)
    logger.info(f'precision: cl {p_cl}, wm {p_wm}; recall: cl {r_cl}, wm {r_wm}')
    logger.info(f'coverage: cl {c_cl}, wm {c_wm}; density: cl {d_cl}, wm {d_wm}')

    return brisque_cl, brisque_wm, fid_cl, fid_wm, \
        p_wm, r_wm, c_wm, d_wm, p_cl, r_cl, c_cl, d_cl

def main(args):
    logger = get_logger(args.log_name)

    data_path = os.path.join('./data', args.dataset)

    brisque_cl_m, brisque_wm_m, fid_cl_m, fid_wm_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    p_wm_m, r_wm_m, c_wm_m, d_wm_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    p_cl_m, r_cl_m, c_cl_m, d_cl_m = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for i, subdir in enumerate(os.listdir(data_path)):
        logger.info(f'{i} {subdir}')

        # args.orig_path = os.path.join(data_path, subdir)
        # args.out_cl_path = os.path.join('./outputs', f'{args.dataset}_clean_{args.ei}',
        #     f'{subdir}_CLEAN', 'checkpoint-1000/dreambooth/a_photo_of_sks_person')
        # args.out_wm_path = os.path.join('./outputs', f'{args.dataset}_{args.wi}_{args.ei}',
        #     f'{subdir}_CLOAKED', 'checkpoint-1000/dreambooth/a_photo_of_sks_person')

        brisque_cl, brisque_wm, fid_cl, fid_wm, \
            p_wm, r_wm, c_wm, d_wm, p_cl, r_cl, c_cl, d_cl = eval_one(args, logger)
        
        brisque_cl_m.update(brisque_cl); brisque_wm_m.update(brisque_wm)
        fid_cl_m.update(fid_cl); fid_wm_m.update(fid_wm)
        p_wm_m.update(p_wm); r_wm_m.update(r_wm); c_wm_m.update(c_wm); d_wm_m.update(d_wm)
        p_cl_m.update(p_cl); r_cl_m.update(r_cl); c_cl_m.update(c_cl); d_cl_m.update(d_cl)
        break

    logger.info(f'Overall BRISQUE: cl {brisque_cl_m.avg}, wm {brisque_wm_m.avg}')
    logger.info(f'Overall FID: cl {fid_cl_m.avg}, wm {fid_wm_m.avg}')
    logger.info(f'Overall precision: cl {p_cl_m.avg}, wm {p_wm_m.avg}')
    logger.info(f'Overall recall: cl {r_cl_m.avg}, wm {r_wm_m.avg}')
    logger.info(f'Overall coverage: cl {c_cl_m.avg}, wm {c_wm_m.avg}')
    logger.info(f'Overall density: cl {d_cl_m.avg}, wm {d_wm_m.avg}')

    print(f'Overall BRISQUE: cl {brisque_cl_m.avg}, wm {brisque_wm_m.avg}')
    print(f'Overall FID: cl {fid_cl_m.avg}, wm {fid_wm_m.avg}')
    print(f'Overall precision: cl {p_cl_m.avg}, wm {p_wm_m.avg}')
    print(f'Overall recall: cl {r_cl_m.avg}, wm {r_wm_m.avg}')
    print(f'Overall coverage: cl {c_cl_m.avg}, wm {c_wm_m.avg}')
    print(f'Overall density: cl {d_cl_m.avg}, wm {d_wm_m.avg}')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default='test')
    parser.add_argument('--orig_path', type=str, default='./data/VGGFace2/n000050')
    parser.add_argument('--out_wm_path', type=str,
        default='./outputs/test/n00050_CLOAKED/checkpoint-1000/dreambooth/a_photo_of_sks_person')
    parser.add_argument('--out_cl_path', type=str,
        default='./outputs/test/n000050_CLEAN/checkpoint-1000/dreambooth/a_photo_of_sks_person')

    parser.add_argument('--dataset', type=str, default="VGGFace2") # VGGFace2
    parser.add_argument('--wi', type=str, default=None) # w0
    parser.add_argument('--ei', type=str, default=None) # e0
    
    args = parser.parse_args()

    main(args)