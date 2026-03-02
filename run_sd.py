import torch
from PIL import Image
from tqdm.auto import tqdm
import open_clip
import argparse
import os
import torch.nn.functional as F
import data_augmentation
from torch import nn
from torchvision import transforms
from copy import deepcopy
from run_image_editing import get_logger
from omegaconf import OmegaConf 
from diffusers import AutoencoderKL, StableDiffusionPipeline 
from diffusers.models.vae import DecoderOutput
# from utils_model import load_model_from_config 
import importlib
from run_hidden import msg2str, str2msg

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    for name in sd.keys():
        # if "proj_in.weight" in name:
        if ("proj_in.weight" in name or "proj_out.weight" in name) and "attn_1" not in name:
            sd[name] = sd[name].unsqueeze(-1).unsqueeze(-1)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def sd_one_round(args, vae, msg_extractor, transform_vae, 
                 transform_imnet, aug_test, device, logger):
    # params
    clean_imgs_path = args.clean_imgs_path
    save_path = args.save_path
    prompts = args.prompts
    to_pil = transforms.ToPILImage()

    # key
    msgs_bool = str2msg(args.key)
    args.num_bits = len(msgs_bool)
    # msgs = torch.tensor([1 if x else -1 for x in msgs_bool]).to(device)

    # prepare data
    prompt = prompts.split(";")[0]
    prompt = prompt.lower().replace(",", "").replace(" ", "_")
    clean_imgs_path = os.path.join(clean_imgs_path, "checkpoint-1000", "dreambooth", prompt)
    clean_imgs_paths = [os.path.join(clean_imgs_path, file_name) for file_name in os.listdir(clean_imgs_path)]
    images = [Image.open(image_path).convert("RGB") for image_path in clean_imgs_paths]
    images = [transform_vae(i).unsqueeze(0) for i in images]
    images = torch.concatenate(images).to(device)
    resize = transforms.Resize(512)

    # start stable signature
    save_path = os.path.join(save_path, "checkpoint-1000", "dreambooth", prompt)
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        # watermark encoding
        wm_images = vae(images).sample # torch.tensor: [16, 3, 512, 512]
        images, wm_images = (images / 2 + 0.5).clamp(0, 1), (wm_images / 2 + 0.5).clamp(0, 1)
        if args.save_images: # save watermarked images
            for i in range(wm_images.shape[0]):
                save_image = to_pil(wm_images[i]).convert("RGB")
                save_image.save(os.path.join(save_path, clean_imgs_paths[i].split('/')[-1]))
        images, wm_images = transform_imnet(images).to("cuda"), transform_imnet(wm_images).to("cuda")


        acc_clean_dict, acc_wm_dict = {}, {}
        # comput statistics and save images
        for name, aug in aug_test.items():
            acc_clean_dict[name], acc_wm_dict[name] = 0, 0

            # watermark extraction
            msg_clean = msg_extractor(resize(aug(images))) # b c h w -> b k
            msg_wm = msg_extractor(resize(aug(wm_images))) # b c h w -> b k

            msg_clean_bools = (msg_clean>0).cpu().numpy().tolist()
            msg_wm_bools = (msg_wm>0).cpu().numpy().tolist()
            for i in range(len(msg_clean_bools)):
                msg_clean_bool, msg_wm_bool = msg_clean_bools[i], msg_wm_bools[i]
                # clean
                diff = [msg_clean_bool[j] != msgs_bool[j] for j in range(len(msg_clean_bool))]
                bit_acc = 1 - sum(diff)/len(diff)
                acc_clean_dict[name] += bit_acc

                # wm
                diff = [msg_wm_bool[j] != msgs_bool[j] for j in range(len(msg_wm_bool))]
                bit_acc = 1 - sum(diff)/len(diff)
                acc_wm_dict[name] += bit_acc

    report_str = ""
    for name in acc_clean_dict.keys():
        acc_clean_dict[name] /= len(os.listdir(clean_imgs_path))
        acc_wm_dict[name] /= len(os.listdir(clean_imgs_path))

        report_str += f"{name}, acc_orig {acc_clean_dict[name]}, acc_wm {acc_wm_dict[name]}; "
    logger.info(report_str)

    return acc_clean_dict, acc_wm_dict

def main(args):
    logger = get_logger(args.log_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load stable signature decoder
    print(f'>>> Building LDM model with config {args.ldm_config} and weights from {args.ldm_ckpt}...')
    config = OmegaConf.load(f"{args.ldm_config}")
    ldm_ae = load_model_from_config(config, args.ldm_ckpt)
    ldm_aef = ldm_ae.first_stage_model
    ldm_aef.eval()

    # loading the fine-tuned decoder weights
    state_dict = torch.load(args.sd_decoder)
    unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)
    print(unexpected_keys)
    print("you should check that the decoder keys are correctly matched")

    # load VAE
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, 
                                        subfolder="vae", revision=None).to(device)
    vae.decode = (lambda x,  *args, **kwargs: DecoderOutput(sample=ldm_aef.decode(x)))
    transform_vae = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # watermark extractor
    msg_extractor = torch.jit.load(args.msg_extractor).to("cuda")
    transform_imnet = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])

    
    # load augmentation
    if args.transform_mode == 'all':
        aug_test = {
            'none': lambda x: x,
            'crop_05': lambda x: data_augmentation.center_crop(x, 0.5),
            'crop_01': lambda x: data_augmentation.center_crop(x, 0.1),
            'rot_25': lambda x: data_augmentation.rotate(x, 25),
            'rot_90': lambda x: data_augmentation.rotate(x, 90),
            'jpeg_80': lambda x: data_augmentation.jpeg_compress(x, 80),
            'jpeg_50': lambda x: data_augmentation.jpeg_compress(x, 50),
            'brightness_1p5': lambda x: data_augmentation.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: data_augmentation.adjust_brightness(x, 2),
            'contrast_1p5': lambda x: data_augmentation.adjust_contrast(x, 1.5),
            'contrast_2': lambda x: data_augmentation.adjust_contrast(x, 2),
            'saturation_1p5': lambda x: data_augmentation.adjust_saturation(x, 1.5),
            'saturation_2': lambda x: data_augmentation.adjust_saturation(x, 2),
            'sharpness_1p5': lambda x: data_augmentation.adjust_sharpness(x, 1.5),
            'sharpness_2': lambda x: data_augmentation.adjust_sharpness(x, 2),
            'resize_05': lambda x: data_augmentation.resize(x, 0.5),
            'resize_01': lambda x: data_augmentation.resize(x, 0.1),
            'overlay_text': lambda x: data_augmentation.overlay_text(x, [76,111,114,101,109,32,73,112,115,117,109]),
            'comb': lambda x: data_augmentation.jpeg_compress(data_augmentation.adjust_brightness(data_augmentation.center_crop(x, 0.5), 1.5), 80),
        }
    elif args.transform_mode == 'few':
        aug_test = {
            'none': lambda x: x,
            'crop_01': lambda x: data_augmentation.center_crop(x, 0.1),
            'brightness_2': lambda x: data_augmentation.adjust_brightness(x, 2),
            'contrast_2': lambda x: data_augmentation.adjust_contrast(x, 2),
            'jpeg_50': lambda x: data_augmentation.jpeg_compress(x, 50),
            'comb': lambda x: data_augmentation.jpeg_compress(data_augmentation.adjust_brightness(data_augmentation.center_crop(x, 0.5), 1.5), 80),
        }
    else:
        aug_test = {'none': lambda x: x}

    # load data
    data_path = os.path.join('./data', args.dataset)
    acc_clean_dict_total, acc_wm_dict_total = {}, {}
    for i, subdir in enumerate(os.listdir(data_path)):
        if 'w0' in subdir or 'w1' in subdir or 'all' in subdir:
            continue
        logger.info(f'{i} {subdir}')
        args.clean_imgs_path = os.path.join('./outputs', f'{args.dataset}_clean_{args.ei}', f'{subdir}_CLEAN')
        args.save_path = os.path.join('./outputs', f'{args.dataset}_sd_{args.ei}', f'{subdir}_CLEAN')

        acc_clean_dict, acc_wm_dict = sd_one_round(args, vae, msg_extractor, transform_vae,
                                        transform_imnet, aug_test, device, logger)
        if i == 0:
            acc_clean_dict_total = acc_clean_dict
            acc_wm_dict_total = acc_wm_dict
        else:
            for name in acc_clean_dict_total.keys():
                acc_clean_dict_total[name] += acc_clean_dict[name]
                acc_wm_dict_total[name] += acc_wm_dict[name]
    
    report_str = "Overall: "
    for name in acc_clean_dict_total.keys():
        acc_clean_dict_total[name] /= len(os.listdir(data_path))
        acc_wm_dict_total[name] /= len(os.listdir(data_path))
        report_str += f"{name}, acc_orig_total {acc_clean_dict_total[name]}, acc_edit_total {acc_wm_dict_total[name]}; "
    logger.info(report_str)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # params for stable signature
    parser.add_argument("--ldm_config", type=str, default="sd/stable-diffusion-2-1-base/v2-inference.yaml")
    parser.add_argument("--ldm_ckpt", type=str, default="sd/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt")
    parser.add_argument("--sd_decoder", type=str, default="sd/sd2_decoder.pth")
    parser.add_argument("--msg_extractor", type=str, default="./sd/dec_48b_whit.torchscript.pt")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-2-1-base")

    parser.add_argument('--log_name', type=str, default='VGGFace2_sd_e0')
    parser.add_argument('--prompts', type=str, default="a photo of sks person")
    parser.add_argument('--dataset', type=str, default="VGGFace2")
    parser.add_argument('--wi', type=str, default="sd")
    parser.add_argument('--ei', type=str, default="e0")

    # watermark labels
    parser.add_argument("--key", type=str, default='111010110101000001010111010011010100010000100111')

    # watermark params
    parser.add_argument("--num_bits", type=int, default=48, help="Number of bits of the watermark (Default: 48)")
    
    # eval params
    parser.add_argument("--transform_mode", type=str, default="few", help="'all', 'few' or 'none'")
    parser.add_argument("--save_images", type=int, default=0)
    args = parser.parse_args()

    main(args)