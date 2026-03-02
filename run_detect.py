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
    clean_path = args.clean_path
    cloak_path = args.cloak_path

    prompts = args.prompts
    data_aug = data_augmentation.HiddenAug(224, args.p_crop, args.p_blur, args.p_jpeg, args.p_rot, args.p_color_jitter, args.p_res).to(device)
    resize = transforms.Resize(224)
    prompt = prompts.split(";")[0]
    prompt = prompt.lower().replace(",", "").replace(" ", "_")
    clean_path = os.path.join(clean_path, "checkpoint-1000", "dreambooth", prompt)
    cloak_path = os.path.join(cloak_path, "checkpoint-1000", "dreambooth", prompt)

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

    orig_image_paths = [os.path.join(clean_path, file_name) for file_name in os.listdir(clean_path)]
    edit_image_paths = [os.path.join(cloak_path, file_name) for file_name in os.listdir(cloak_path)]
    num_imgs = 16 // 2
    orig_train_images = [Image.open(image_path).convert("RGB") for image_path in orig_image_paths[:num_imgs]]
    orig_test_images = [Image.open(image_path).convert("RGB") for image_path in orig_image_paths[num_imgs:]]
    edit_train_images = [Image.open(image_path).convert("RGB") for image_path in edit_image_paths[:num_imgs]]
    edit_test_images = [Image.open(image_path).convert("RGB") for image_path in edit_image_paths[num_imgs:]]

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
            train_images = [transform_train(i).unsqueeze(0) for i in orig_train_images + edit_train_images]
            train_images = torch.concatenate(train_images).to(device)
            train_images = resize(data_aug(train_images))

            with torch.no_grad():
                target_features = model.encode_image(train_images)
            fts = fc(target_features)
            msgs_rev = - msgs
            loss = message_loss(fts[:num_imgs], msgs_rev.repeat(num_imgs, 1),
                                m=args.loss_margin, loss_type=args.loss_w_type) \
                + message_loss(fts[num_imgs:], msgs.repeat(num_imgs, 1),
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

    # cloak_path = "./a1_out/n000142/neutralized__stubble"
    # cloak_path = "./a2_out/n000142/checkpoint-1000/dreambooth/a_photo_of_sks_person"
    # cloak_path = "./a3_out_guided/n000142/regenerated"
    # cloak_path = "./a3_out_unguided/n000142/regenerated"
    # cloak_path = "./outputs/test/n000050_NOISYUPSCALE/checkpoint-1000/dreambooth/a_photo_of_sks_person"
    # edit_image_paths = [os.path.join(cloak_path, file_name) for file_name in os.listdir(cloak_path)]
    # edit_test_images = [Image.open(image_path).convert("RGB") for image_path in edit_image_paths[num_imgs:]]
    
    images = [transform_test(i).unsqueeze(0) for i in orig_test_images]
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
    
    images = [transform_test(i).unsqueeze(0) for i in edit_test_images]
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

    report_str = ""
    for name in acc_edit_dict.keys():
        acc_orig_dict[name] /= num_imgs
        acc_edit_dict[name] /= num_imgs

        report_str += f"{name}, acc_orig {acc_orig_dict[name]}, acc_edit {acc_edit_dict[name]}; "
    logger.info(report_str)

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

    train_and_test(model, transform_train, transform_test, aug_test, args, logger, device)


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
    parser.add_argument('--clean_path', type=str, default="./outputs/test/n000142_CLEAN/")
    parser.add_argument('--cloak_path', type=str, default="./outputs/test/n000142_CLOAKED/")
    parser.add_argument('--addi_train_path', type=str, default=None)

    # message labels
    parser.add_argument("--key", type=str, default='111010110101000001010111010011010100010000100111')

    # params
    parser.add_argument("--num_bits", type=int, default=48, help="Number of bits (Default: 48)")
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