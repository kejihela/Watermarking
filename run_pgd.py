import torch
import torch.nn as nn

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


def pgd(x, x_trans, model, alpha, eps, iters=100):

    x_adv = x.clone().detach()

    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam([x_adv], lr=0.1)
    
    for i in pbar:

        x_adv.requires_grad_(True)

        x_emb = model(x_adv).latent_dist.sample()
        x_trans_emb = model(x_trans).latent_dist.sample()

        optimizer.zero_grad()

        loss = criterion(x_emb, x_trans_emb)
        # Getting gradients w.r.t. parameters
        loss.backward()
        # Updating parameters
        adv_images = x_adv - alpha * x_adv.grad.sign()
        eta = torch.clamp(adv_images - x, min=-eps, max=+eps)
        x_adv = torch.clamp(x + eta, min=-1, max=+1).detach_()
        # scheduler.step()
        pbar.set_description(f"[Running pgd]: Loss {loss.item():.5f} | noise norm {torch.norm(eta, p=2):.5f}")

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
        trans_image = Image.open(f"{args.trans_data_dir}/{file_name.replace('jpg', 'png')}").convert("RGB")

        x = preprocess(init_image).to(device)
        x_t = preprocess(trans_image).to(device)
        # x = torch.squeeze(x)
        x_adv = pgd(x, x_t, model=pipe_img2img.vae.encode,
                        alpha=args.pgd_alpha, eps=args.pgd_eps, iters=args.pgd_iters)


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
    parser.add_argument('--model', default='Manojb/stable-diffusion-2-1-base', type=str,
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
    parser.add_argument("--pgd_alpha", type=float, default=1.0 / 255, help="The step size for pgd.")
    parser.add_argument("--pgd_eps", type=float, default=0.1, help="The noise budget for pgd.")
    parser.add_argument("--pgd_iters", type=int, default=500, help="The training iterations for pgd.")

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