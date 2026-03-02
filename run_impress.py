import argparse
import torch
import numpy as np
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torch import nn
import lpips

def impress(X_adv, model, eps=0.1, iters=40, clamp_min=0, clamp_max=1, lr=0.001, pur_alpha=0.5, noise=0.1):
    # init purified X
    X_p = X_adv.clone().detach()  + (torch.randn(*X_adv.shape) * noise).to(X_adv.device).half()
    pbar = tqdm(range(iters))
    criterion = nn.MSELoss()
    loss_fn_alex = lpips.LPIPS(net='vgg').to(X_adv.device)
    optimizer = torch.optim.Adam([X_p], lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iters, eta_min=1e-5)
    for i in pbar:
        X_p.requires_grad_(True)
        _X_p = model(X_p).sample
        optimizer.zero_grad()
        lnorm_loss = criterion(_X_p, X_p)
        d = loss_fn_alex(X_p, X_adv)
        lpips_loss = max(d - eps, 0)
        loss = lnorm_loss + pur_alpha * lpips_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        X_p.data = torch.clamp(X_p, min=clamp_min, max=clamp_max)
        pbar.set_description(f"[Running purify]: Loss: {loss.item():.5f} | l2 dist: {lnorm_loss.item():.4} | lpips loss: {d.item():.4}")
    X_p.requires_grad_(False)
    return X_p

def preprocess(image):
    w, h = image.size
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    # image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

def main(args):
    to_pil = T.ToPILImage()
    pipe_img2img = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(pipe_img2img.scheduler.config)
    pipe_img2img = pipe_img2img.to(device)
    # pipe_img2img.enable_xformers_memory_efficient_attention()

    for name, param in pipe_img2img.vae.named_parameters():
        param.requires_grad = False

    load_path = os.path.join('./data', f"{args.dataset}_cloak", f"{args.wi}_{args.ei}", args.id, "set_A")
    save_path = os.path.join('./data', f"{args.dataset}_pur", f"{args.wi}_{args.ei}", args.id, "set_A")
    os.makedirs(save_path, exist_ok=True)

    # start
    for imgdir in os.listdir(load_path):
        adv_image = Image.open(os.path.join(load_path, imgdir)).convert("RGB")
        x_adv = preprocess(adv_image).to(device).half()
        x_purified = impress(x_adv,
                             model=pipe_img2img.vae,
                             clamp_min=-1,
                             clamp_max=1,
                             eps=args.pur_eps,
                             iters=args.pur_iters,
                             lr=args.pur_lr,
                             pur_alpha=args.pur_alpha,
                             noise=args.pur_noise, )

        x_purified = (x_purified / 2 + 0.5).clamp(0, 1)
        purified_image = to_pil(x_purified[0]).convert("RGB")
        purified_image.save(f"{save_path}/{imgdir}")
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='diffusion attack')


    # model_id = "stabilityai/stable-diffusion-2-1"
    parser.add_argument('--model', default='stabilityai/stable-diffusion-2-1-base', type=str,
                        help='stable diffusion weight')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train', default= 0, type=bool,
                        help='training(True) or testing(False)')

    # data
    parser.add_argument('--dataset', type=str, default="VGGFace2")
    parser.add_argument('--wi', type=str, default="w0")
    parser.add_argument('--ei', type=str, default="e0")
    parser.add_argument('--id', type=str, default=None)

    # ae Hyperparameters
    parser.add_argument('--pur_eps', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_iters', default=3000, type=int, help='ae Hyperparameters')
    parser.add_argument('--pur_lr', default=0.01, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_alpha', default=0.1, type=float, help='ae Hyperparameters')
    parser.add_argument('--pur_noise', default=0.1, type=float, help='ae Hyperparameters')



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