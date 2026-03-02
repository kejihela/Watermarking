import torch
import torch.nn as nn
import lpips
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torchvision.transforms as T

import argparse


def preprocess(image):
    w, h = image.size
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    # image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def glaze(x, x_trans, model, p=0.1, alpha=0.1, iters=500, lr=0.002):
    # x_adv = x.clone().detach()  + (torch.rand(*x.shape) * 2 * p - p).to(x.device)
    delta = (torch.rand(*x.shape) * 2 * p - p).to(x.device)
    # input_var = nn.Parameter(torch.rand(*x.shape) * 2 * p - p, requires_grad=True).to(x.device)
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([delta], lr=lr)
    loss_fn_alex = lpips.LPIPS(net='vgg').to(x.device)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters, eta_min=0.001)
    for i in pbar:
        # x_adv_image = x.clone().detach()

        delta.requires_grad_(True)
        x_adv = x + delta
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        x_emb = model(x_adv).latent_dist.sample()
        x_trans_emb = model(x_trans).latent_dist.sample()
        # x_trans_emb = model(x_trans).latent_dist.sample()
        optimizer.zero_grad()
        d = loss_fn_alex(x, x_adv)
        sim_loss = alpha * max(d-p, 0)
        loss = criterion(x_emb, x_trans_emb) + sim_loss
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        optimizer.step()
        # scheduler.step()
        pbar.set_description(f"[Running glaze]: Loss {loss.item():.5f} | sim loss {alpha * max(d.item()-p, 0):.5f} | dist {d.item():.5f}")
    x_adv = x + delta
    x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
    return x_adv

def main(args):
    # make sure you're logged in with `huggingface-cli login` - check https://github.com/huggingface/diffusers for more details
    to_pil = T.ToPILImage()
    pipe_img2img = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32)
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)
    # pipe_img2img.enable_xformers_memory_efficient_attention()
    for name, param in pipe_img2img.vae.encoder.named_parameters():
        param.requires_grad = False

    os.makedirs(args.save_dir, exist_ok=True)

    for file_name in os.listdir(args.clean_data_dir):

        init_image = Image.open(f"{args.clean_data_dir}/{file_name}").convert("RGB")
        trans_image = Image.open(f"{args.trans_data_dir}/{file_name}").convert("RGB")

        x = preprocess(init_image).to(device)
        x_t = preprocess(trans_image).to(device)
        # x = torch.squeeze(x)
        x_adv = glaze(x, x_t, model=pipe_img2img.vae.encode,
                        p=args.p, alpha=args.alpha, iters=args.glaze_iters, lr=args.lr)


        x_adv = (x_adv / 2 + 0.5).clamp(0, 1)
        adv_image = to_pil(x_adv[0]).convert("RGB")
        adv_image.save(f"{args.save_dir}/{file_name}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')

    # model_id_or_path = "runwayml/stable-diffusion-v1-5"
    # model_id_or_path = "CompVis/stable-diffusion-v1-4"
    # model_id_or_path = "CompVis/stable-diffusion-v1-3"
    # model_id_or_path = "CompVis/stable-diffusion-v1-2"
    # model_id_or_path = "CompVis/stable-diffusion-v1-1"
    # model_id = "stabilityai/stable-diffusion-2-1"
    parser.add_argument('--model', default='stabilityai/stable-diffusion-2-1-base', type=str,
                        help='stable diffusion weight')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train', default= 0, type=bool,
                        help='training(True) or testing(False)')

    # data
    parser.add_argument('--clean_data_dir', type=str, default='./data/n000050/set_A')
    parser.add_argument('--trans_data_dir', type=str, default='./data/n000050/set_A_edited')
    parser.add_argument('--save_dir', type=str, default='./data/n000050/set_A_cloaked')

    # pgd Hyperparameters
    parser.add_argument('--p', default=0.05, type=float, help='pgd Hyperparameters')
    parser.add_argument('--alpha', default=30, type=int, help='pgd Hyperparameters')
    parser.add_argument('--glaze_iters', default=500, type=int, help='pgd Hyperparameters')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate.')


    # Miscs
    parser.add_argument('--manual_seed', default=None, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device used for training')

    args = parser.parse_args()
    if args.manual_seed is not None:
        np.random.seed(seed = args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.backends.cudnn.deterministic=True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    main(args)