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
from torch.optim.lr_scheduler import CosineAnnealingLR
from run_image_editing import get_logger

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

def message_loss(fts, targets, m, loss_type='mse'):
    """
    Compute the message loss
    Args:
        dot products (b k*r): the dot products between the carriers and the feature
        targets (KxD): boolean message vectors or gaussian vectors
        m: margin of the Hinge loss or temperature of the sigmoid of the BCE loss
    """
    if loss_type == 'bce':
        return F.binary_cross_entropy(torch.sigmoid(fts/m), 0.5*(targets+1), reduction='mean')
    elif loss_type == 'cossim':
        return -torch.mean(torch.cosine_similarity(fts, targets, dim=-1))
    elif loss_type == 'mse':
        return F.mse_loss(fts, targets, reduction='mean')
    else:
        raise ValueError('Unknown loss type')

def train_and_test(model, transform_train, transform_test, aug_test, args, logger, device, fc_dict=None):

    # params
    orig_train_path = args.orig_train_path
    edit_train_path = args.edit_train_path
    orig_test_path = args.orig_test_path
    edit_test_path = args.edit_test_path
    prompts = args.prompts
    data_aug = data_augmentation.HiddenAug(224, args.p_crop, args.p_blur, args.p_jpeg, args.p_rot, args.p_color_jitter, args.p_res).to(device)
    resize = transforms.Resize(224)

    ###############
    #    TRAIN    #
    ###############
    # random target label
    if args.key is None:
        msgs = torch.rand((1, args.num_bits)) > 0.5 # b k, True/False
        msgs = 2 * msgs.type(torch.float).to(device) - 1 # b k, -1./1.
        msgs_bool = (msgs>0)[0].cpu().numpy().tolist()
    else:
        msgs_bool = str2msg(args.key)
        args.num_bits = len(msgs_bool)
        msgs = torch.tensor([1 if x else -1 for x in msgs_bool]).to(device)

    orig_image_paths = [os.path.join(orig_train_path, file_name) for file_name in os.listdir(orig_train_path) if '.png' in file_name]
    edit_image_paths = []
    for file_name in os.listdir(edit_train_path):
        if 'combined_image' not in file_name and 'null_inversion' not in file_name and '.png' in file_name:
            edit_image_paths.append(os.path.join(edit_train_path, file_name))
    images = [Image.open(image_path).convert("RGB") for image_path in orig_image_paths + edit_image_paths]
    fc = torch.nn.Linear(1024, args.num_bits).to(device)
    nn.init.normal_(fc.weight, 0, 0.01)
    nn.init.constant_(fc.bias, 0)

    if fc_dict is not None:
        fc.load_state_dict(fc_dict)
    else:
        optimizer = torch.optim.AdamW(fc.parameters(), lr=1e-2, weight_decay=1e-2) # 1e-2 for w0
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        iters = 100

        progress_bar = tqdm(range(iters))
        progress_bar.set_description("Steps")
        for _ in range(iters):
            imgs_train = [transform_train(i).unsqueeze(0) for i in images]
            imgs_train = torch.concatenate(imgs_train).to(device)
            imgs_train = resize(data_aug(imgs_train))
            with torch.no_grad():
                target_features = model.encode_image(imgs_train)
            fts = fc(target_features)
            if 'w1' in args.wi:
                msgs_rev = - msgs
                loss = message_loss(fts[:4], msgs_rev.repeat(4, 1),
                                    m=args.loss_margin, loss_type=args.loss_w_type) \
                    + message_loss(fts[4:], msgs.repeat(4, 1),
                                    m=args.loss_margin, loss_type=args.loss_w_type)
            else:
                loss = message_loss(fts[:4], msgs.repeat(4, 1),
                                    m=args.loss_margin, loss_type=args.loss_w_type) \
                    - message_loss(fts[4:], msgs.repeat(4, 1),
                                    m=args.loss_margin, loss_type=args.loss_w_type)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
            progress_bar.update(1)
            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)


    ##############
    #    TEST    #
    ##############
    prompt = prompts.split(";")[0]
    prompt = prompt.lower().replace(",", "").replace(" ", "_")

    orig_test_path = os.path.join(orig_test_path, "checkpoint-1000", "dreambooth", prompt)
    images = [Image.open(os.path.join(orig_test_path, image_path)).convert("RGB") \
                        for image_path in os.listdir(orig_test_path)]
    images = [transform_test(i).unsqueeze(0) for i in images]
    images = torch.concatenate(images).to(device)

    acc_orig_dict = {}
    for name, aug in aug_test.items():
        acc_orig_dict[name] = 0
        with torch.no_grad():
            target_features = model.encode_image(resize(aug(images)))
            fts = fc(target_features)
        fts_bools = (fts>0).cpu().numpy().tolist()
        for i, fts_bool in enumerate(fts_bools):
            diff = [fts_bool[j] != msgs_bool[j] for j in range(len(fts_bool))]
            bit_acc = 1 - sum(diff)/len(diff)
            acc_orig_dict[name] += bit_acc

            # if args.wi == 'w1':
            #     logger.info(f"w/o wm {os.listdir(orig_test_path)[i].split('/')[-1]}, bit acc: {bit_acc}")
            # else:
            #     logger.info(f"w/ wm {os.listdir(orig_test_path)[i].split('/')[-1]}, bit acc: {bit_acc}")


    edit_test_path = os.path.join(edit_test_path, "checkpoint-1000", "dreambooth", prompt)
    images = [Image.open(os.path.join(edit_test_path, image_path)).convert("RGB") \
                        for image_path in os.listdir(edit_test_path)]
    images = [transform_test(i).unsqueeze(0) for i in images]
    images = torch.concatenate(images).to(device)

    acc_edit_dict = {}
    for name, aug in aug_test.items():
        acc_edit_dict[name] = 0
        with torch.no_grad():
            target_features = model.encode_image(resize(aug(images)))
            fts = fc(target_features)
        fts_bools = (fts>0).cpu().numpy().tolist()

        for i, fts_bool in enumerate(fts_bools):
            diff = [fts_bool[j] != msgs_bool[j] for j in range(len(fts_bool))]
            bit_acc = 1 - sum(diff)/len(diff)
            acc_edit_dict[name] += bit_acc

            # if args.wi == 'w1':
            #     logger.info(f"w/ wm {os.listdir(edit_test_path)[i].split('/')[-1]}, bit acc: {bit_acc}")
            # else:
            #     logger.info(f"w/o wm {os.listdir(edit_test_path)[i].split('/')[-1]}, bit acc: {bit_acc}")
    
    report_str = ""
    for name in acc_edit_dict.keys():
        acc_orig_dict[name] /= len(os.listdir(orig_test_path))
        acc_edit_dict[name] /= len(os.listdir(edit_test_path))

        report_str += f"{name}, acc_orig {acc_orig_dict[name]}, acc_edit {acc_edit_dict[name]}; "
    logger.info(report_str)

    if args.resume_fc:
        return None, acc_orig_dict, acc_edit_dict
    else:
        return fc.state_dict(), acc_orig_dict, acc_edit_dict

def main(args):
    logger = get_logger(args.log_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)
    # preprocess: Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

    transform_train = deepcopy(preprocess)
    transform_test = deepcopy(preprocess)
    transform_train.transforms.insert(0, transforms.Compose([
        transforms.RandomResizedCrop(512),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(),]))

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

    if args.dataset is not None:
        data_path = os.path.join('./data', args.dataset)
        save_dict = {}
        acc_orig_dict_total, acc_edit_dict_total = {}, {}

        load_path = os.path.join('./outputs', f'{args.dataset}_{args.wi}_{args.ei}', 'fc.pth')
        if args.resume_fc and os.path.exists(load_path):
            save_dict = torch.load(load_path)

        for i, subdir in enumerate(os.listdir(data_path)):
            logger.info(f'{i} {subdir}')
            args.orig_train_path = os.path.join('./data', args.dataset, subdir, 'set_A')
            wi = 'w0' if 'w0' in args.wi else 'w1'
            args.edit_train_path = os.path.join('./data', f'{args.dataset}_edit', f'{wi}_{args.ei}', subdir, 'set_A')
            args.edit_test_path = os.path.join('./outputs', f'{args.dataset}_{wi}_{args.ei}', f'{subdir}_CLOAKED')
            args.orig_test_path = os.path.join('./outputs', f'{args.dataset}_clean_{args.ei}', f'{subdir}_CLEAN')
            # args.edit_test_path = os.path.join('./outputs', f'{args.dataset}_{args.wi}_{args.ei}', f'{subdir}_CLOAKED')
            
            if args.resume_fc and os.path.exists(load_path):
                _, acc_orig_dict, acc_edit_dict = train_and_test(model, transform_train, transform_test, aug_test,
                                        args, logger, device, save_dict[str(subdir)])
            else:
                save_dict[str(subdir)], acc_orig_dict, acc_edit_dict = train_and_test(model, transform_train,
                                        transform_test, aug_test, args, logger, device, None)

            if i == 0:
                acc_orig_dict_total = acc_orig_dict
                acc_edit_dict_total = acc_edit_dict
            else:
                for name in acc_orig_dict_total.keys():
                    acc_orig_dict_total[name] += acc_orig_dict[name]
                    acc_edit_dict_total[name] += acc_edit_dict[name]

        report_str = "Overall: "
        for name in acc_orig_dict_total.keys():
            acc_orig_dict_total[name] /= len(os.listdir(data_path))
            acc_edit_dict_total[name] /= len(os.listdir(data_path))
            report_str += f"{name}, acc_orig_total {acc_orig_dict_total[name]}, acc_edit_total {acc_edit_dict_total[name]}; "
        logger.info(report_str)
        if not args.resume_fc:
            wi = 'w0' if 'w0' in args.wi else 'w1'
            save_path = os.path.join('./outputs', f'{args.dataset}_{wi}_{args.ei}', 'fc.pth')
            torch.save(save_dict, save_path)
    else:
        if args.wi is None:
            args.wi = 'w0'
        load_path = './outputs/fc.pth'
        if args.resume_fc and os.path.exists(load_path):
            save_dict = torch.load(load_path)
        else:
            save_dict = None
        save_dict, _, _ = train_and_test(model, transform_train, transform_test, aug_test, args, logger, device, fc_dict=save_dict)
        if not args.resume_fc:
            torch.save(save_dict, load_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clip_model', type=str, default='ViT-H-14')
    parser.add_argument('--clip_pretrain', type=str, default='laion2b_s32b_b79k')
    parser.add_argument('--log_name', type=str, default='test')
    parser.add_argument('--resume_fc', type=int, default=0, help='whether resume extractor and skip the training')

    # customized params
    # parser.add_argument('--prompts', type=str, 
    #     default="a photo of sks person;a close-up photo of sks person, high details;a photo of sks person in front of eiffel tower;a photo of sks person looking at the mirror")
    parser.add_argument('--prompts', type=str, default="a photo of sks person")
    parser.add_argument('--orig_train_path', type=str, default="./data/n000050/set_A/")
    parser.add_argument('--edit_train_path', type=str, default="./data/n000050/set_A_edited/")
    parser.add_argument('--orig_test_path', type=str, default="./outputs/test/n000050_CLEAN/")
    parser.add_argument('--edit_test_path', type=str, default="./outputs/test/n000050_CLOAKED/")

    parser.add_argument('--dataset', type=str, default=None) # VGGFace2
    parser.add_argument('--wi', type=str, default=None) # w0
    parser.add_argument('--ei', type=str, default=None) # e0

    # watermark labels
    parser.add_argument("--key", type=str, default='111010110101000001010111010011010100010000100111')

    # watermark params
    parser.add_argument("--num_bits", type=int, default=48, help="Number of bits of the watermark (Default: 48)")
    parser.add_argument("--loss_margin", type=float, default=1,
        help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    parser.add_argument("--loss_w_type", type=str, default='bce',
        help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")
    parser.add_argument("--p_crop", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    parser.add_argument("--p_res", type=float, default=0.5, help="Probability of the crop augmentation. (Default: 0.5)")
    parser.add_argument("--p_blur", type=float, default=0.5, help="Probability of the blur augmentation. (Default: 0.5)")
    parser.add_argument("--p_jpeg", type=float, default=0.5, help="Probability of the diff JPEG augmentation. (Default: 0.5)")
    parser.add_argument("--p_rot", type=float, default=0.5, help="Probability of the rotation augmentation. (Default: 0.5)")
    parser.add_argument("--p_color_jitter", type=float, default=0.5, help="Probability of the color jitter augmentation. (Default: 0.5)")
    
    # eval params
    parser.add_argument("--transform_mode", type=str, default="few", help="'all', 'few' or 'none'")
    args = parser.parse_args()

    main(args)