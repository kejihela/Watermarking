import argparse
import os
import torch
import open_clip
import numpy as np
from PIL import Image
from torch import nn
from tqdm.auto import tqdm
# from tqdm import tqdm
from run_image_editing import get_logger
from run_hidden import str2msg, message_loss
from torchvision import transforms

def project(param_data, backup, epsilon):
    # If the perturbation exceeds the upper bound, project it back.
    r = param_data - backup
    r = epsilon * r

    return backup + r


class Decoder(nn.Module):
    def __init__(self, model, fc) -> None:
        super().__init__()
        self.model = model
        self.fc = fc
    def forward(self, images):
        features = self.model.encode_image(images)
        outputs = self.fc(features)
        return outputs


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


def wevade(args, decoder, device, preprocess, meter, logger):
    wm_imgs_path = args.wm_imgs_path
    prompts = args.prompts
    alpha = args.pgd_alpha
    eps = args.pgd_eps
    iters = args.pgd_iters
    epsilon = args.epsilon
    Bit_acc, Perturbation, Evasion_rate = meter

    # target bits
    msgs_bool = str2msg(args.key)

    # data type needs to be float, not int
    target_watermark = torch.tensor([-1. if x else 1. for x in msgs_bool]).to(device)

    # load watermarked dataset
    prompt = prompts.split(";")[0]
    prompt = prompt.lower().replace(",", "").replace(" ", "_")
    wm_imgs_path = os.path.join(wm_imgs_path, "checkpoint-1000", "dreambooth", prompt)
    wm_imgs_path = [os.path.join(wm_imgs_path, file_name) for file_name in os.listdir(wm_imgs_path)]
    images = [Image.open(image_path).convert("RGB") for image_path in wm_imgs_path]
    images = [preprocess(i).unsqueeze(0) for i in images]

    scaling_factor = args.scaling_factor
    for img_id, x in enumerate(images):

        x = x.to(device)
        x_adv = x.clone().detach()
        optimizer = torch.optim.Adam([x_adv], lr=0.1)

        progress_bar = tqdm(range(iters))
        progress_bar.set_description("Steps")
        for i in range(iters):
            x_adv.requires_grad_(True)
            optimizer.zero_grad()
            
            noise = torch.randn(x.size()).to(device) * scaling_factor
            x_per = x_adv + noise
            x_per = torch.clamp(x_per, min=-1.0, max=+1.0)
            decoded_watermark = decoder(x_per)

            # Post-process the watermarked image.
            if 'w0' in args.wi:
                loss = message_loss(decoded_watermark, - target_watermark.unsqueeze(0),
                                    m=args.loss_margin, loss_type=args.loss_w_type)
            else:
                loss = message_loss(decoded_watermark, target_watermark.unsqueeze(0),
                                    m=args.loss_margin, loss_type=args.loss_w_type)
            
            loss.backward()
            adv_images = x_adv - alpha * x_adv.grad.sign()
            eta = torch.clamp(adv_images - x, min=-eps, max=+eps)
            x_adv = torch.clamp(x + eta, min=-1, max=+1).detach_()

            with torch.no_grad():
                noise = torch.randn(x.size()).to(device) * scaling_factor
                x_per = x_adv + noise
                x_per = torch.clamp(x_per, min=-1.0, max=+1.0)
                decoded_watermark = decoder(x_per)
                decoded_watermark = (decoded_watermark>0).cpu().numpy().tolist()[0]
                diff = [decoded_watermark[j] != msgs_bool[j] for j in range(len(decoded_watermark))]
                bit_acc_target = 1 - sum(diff)/len(diff)


            # Early Stopping.
            # if perturbation_norm.cpu().detach().numpy() >= r:
            #     break
            criterion = (bit_acc_target >= 1 - epsilon) if 'w0' in args.wi else (bit_acc_target <= epsilon)
            if criterion:
                print(f'img_id: {img_id}, i: {i}; loss: {loss.detach().item()}, bit_acc: {bit_acc_target}')
                break
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "bit_acc": bit_acc_target}
            progress_bar.set_postfix(**logs)

        # save result for each image
        x_adv.data = torch.clamp(x_adv, min=-1.0, max=1.0)
        bound = torch.norm(x_adv - x, float('inf'))
        bound = bound.item()
        with torch.no_grad():
            decoded_watermark = decoder(x)
            decoded_watermark = (decoded_watermark>0).cpu().numpy().tolist()[0]
            diff = [decoded_watermark[j] != msgs_bool[j] for j in range(len(decoded_watermark))]
            bit_acc_orig = 1 - sum(diff)/len(diff)

            if 'w0' in args.wi or 'w1' in args.wi:
                bit_acc = 0
                num_repeat = 10
                for _ in range(num_repeat):
                    noise = torch.randn(x.size()).to(device) * scaling_factor
                    decoded_watermark = decoder(torch.clamp(x_adv + noise, min=-1.0, max=1.0))
                    decoded_watermark = (decoded_watermark>0).cpu().numpy().tolist()[0]
                    diff = [decoded_watermark[j] != msgs_bool[j] for j in range(len(decoded_watermark))]
                    bit_acc += 1 - sum(diff)/len(diff)
                bit_acc /= num_repeat
            else:
                decoded_watermark = decoder(x_adv)
                decoded_watermark = (decoded_watermark>0).cpu().numpy().tolist()[0]
                diff = [decoded_watermark[j] != msgs_bool[j] for j in range(len(decoded_watermark))]
                bit_acc = 1 - sum(diff)/len(diff)

        evasion = None
        if ('w0' in args.wi and bit_acc_orig < 1 - args.tau):
            evasion = (bit_acc >= 1 - args.tau)
        elif ('w1' in args.wi and bit_acc_orig > args.tau):
            evasion = (bit_acc <= args.tau)
        elif 'sd' in args.wi and bit_acc_orig > args.tau:
            evasion = (bit_acc <= args.tau)
        if evasion is not None:
            bound = bound / 2   # [-1,1]->[0,1]
            Bit_acc.update(bit_acc, x.shape[0])
            Perturbation.update(bound, x.shape[0])
            Evasion_rate.update(evasion, x.shape[0])

            logger.info(f'bit_acc {bit_acc}, bound {bound}, evasion {evasion}')

    return Bit_acc, Perturbation, Evasion_rate


def main(args):
    logger = get_logger(args.log_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if 'w0' in args.wi or 'w1' in args.wi:
        model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)
        load_path = os.path.join('./outputs', f'{args.dataset}_{args.wi}_{args.ei}', 'fc.pth')
        save_dict = torch.load(load_path)
    elif args.wi == 'sd':
        decoder = torch.jit.load(args.msg_extractor).to(device)
        preprocess = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])

    # training statistics
    data_path = os.path.join('./data', args.dataset)
    args.num_bits = len(str2msg(args.key))
    Bit_acc = AverageMeter()
    Perturbation = AverageMeter()
    Evasion_rate = AverageMeter()

    # start
    for i, subdir in enumerate(os.listdir(data_path)):

        logger.info(f'{args.wi} {i} {subdir}')
        suffix = "CLOAKED" if ('w0' in args.wi or 'w1' in args.wi) else "CLEAN"
        args.wm_imgs_path = os.path.join('./outputs', f'{args.dataset}_{args.wi}_{args.ei}', f'{subdir}_{suffix}')

        # load encoder
        if ('w0' in args.wi or 'w1' in args.wi):
            fc = torch.nn.Linear(1024, args.num_bits).to(device)
            fc.load_state_dict(save_dict[str(subdir)])
            decoder = Decoder(model, fc)

        Bit_acc, Perturbation, Evasion_rate = wevade(args, decoder, device, preprocess, (Bit_acc, Perturbation, Evasion_rate, ), logger)

    logger.info(f"Overall of Attacking {args.dataset}_{args.wi}_{args.ei} " + \
                "Average Bit_acc=%.4f\t Average Perturbation=%.4f\t Evasion rate=%.2f" % (Bit_acc.avg, Perturbation.avg, Evasion_rate.avg))
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # WEvade optimization params
    parser.add_argument('--tau', default=0.8, type=float, help='Detection threshold of the detector.')
    # parser.add_argument('--iteration', default=500, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--epsilon', default=0.01, type=float, help='Epsilon used in WEvdae-W.')
    # parser.add_argument('--alpha', default=0.1, type=float, help='Learning rate used in WEvade-W.')
    # parser.add_argument('--rb', default=2, type=float, help='Upper bound of perturbation.')

    parser.add_argument("--pgd_alpha", type=float, default=1.0 / 255, help="The step size for pgd.")
    parser.add_argument("--pgd_eps", type=float, default=0.1, help="The noise budget for pgd.")
    parser.add_argument("--pgd_iters", type=int, default=500, help="The training iterations for pgd.")

    # w0 and w1
    parser.add_argument('--clip_model', type=str, default='ViT-H-14')
    parser.add_argument('--clip_pretrain', type=str, default='laion2b_s32b_b79k')
    parser.add_argument('--log_name', type=str, default='VGGFace2_wevade_e0_evolve')
    parser.add_argument('--prompts', type=str, default="a photo of sks person")

    # sd
    parser.add_argument("--msg_extractor", type=str, default="./sd/dec_48b_whit.torchscript.pt")

    parser.add_argument('--dataset', type=str, default="VGGFace2")
    parser.add_argument('--wi', type=str, default="w0")
    parser.add_argument('--ei', type=str, default="e0")

    parser.add_argument("--loss_margin", type=float, default=1,
        help="Margin of the Hinge loss or temperature of the sigmoid of the BCE loss. (Default: 1.0)")
    parser.add_argument("--loss_w_type", type=str, default='bce',
        help="Loss type. 'bce' for binary cross entropy, 'cossim' for cosine similarity (Default: bce)")

    # watermark labels
    parser.add_argument("--key", type=str, default='111010110101000001010111010011010100010000100111')

    # watermark params
    parser.add_argument("--num_bits", type=int, default=48, help="Number of bits of the watermark (Default: 32)")
    parser.add_argument("--scaling_factor", type=float, default=0.1)
    args = parser.parse_args()
    main(args)