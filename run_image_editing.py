from optim_utils import * 
import os
import logging
import torch
import argparse
from tqdm.auto import tqdm
from prompt2prompt import *
from ptp_utils import load_512, load_stable_diffusion, run_and_display
from torchvision.utils import save_image, make_grid

from clip_interrogator import Config, Interrogator

class NullInversion:

    def __init__(self, args, model):
        # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
        #                           set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.guidance_scale = args.guidance_scale
        self.device = args.device
        self.num_ddim_steps = args.num_ddim_steps
        self.model.scheduler.set_timesteps(self.num_ddim_steps)
        self.prompt = None
        self.context = None

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        '''
        This function preform a one-step diffusion or denoising,
        which can be viewed as a combination of self.get_noise_pred_single() and
        [self.prev_step() or self.next_step()]
        '''
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else self.guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        '''
        latents: target latent trajectory from DDIM inversion
        '''
        # prepare the params
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]

        # set progress bar
        progress_bar = tqdm(range(num_inner_steps * self.num_ddim_steps))
        progress_bar.set_description("Steps")

        # start optimization, num_ddim_steps=50 (default)
        for i in range(self.num_ddim_steps):

            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True

            optimizer = torch.optim.Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))

            latent_prev = latents[len(latents) - i - 2]

            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)

            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)

                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)

                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if loss_item < epsilon + i * 2e-5:
                    break

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    # "lr": lr_scheduler.get_last_lr()[0]
                }
                progress_bar.set_postfix(**logs)

            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image, image_rec), ddim_latents[-1], uncond_embeddings

def get_logger(log_name):
    os.makedirs('./outputs/logger', exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'./outputs/logger/{log_name}.txt', mode='a')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main(args):

    logger = get_logger(args.log_name)

    # load clip model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

    os.makedirs(args.save_path, exist_ok=True)
    image_paths = [os.path.join(args.orig_path, sub_dir) for sub_dir in os.listdir(args.orig_path)]
    save_paths = [os.path.join(args.save_path, sub_dir) for sub_dir in os.listdir(args.orig_path)]

    orig_images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    offsets = (0,0,0,0)
    connect = " , "

    # resume prompt
    prefix, learned_prompt = [], None
    if args.resume_name is not None and os.path.exists(f'./outputs/logger/{args.resume_name}.txt'):
        for image_path in image_paths:
            with open(f'./outputs/logger/{args.resume_name}.txt') as f:
                for line in f:
                    if image_path in line:
                        prompt = line.split(' - ')[-1]
                        prefix.append(prompt.split(' , ')[0])
                        learned_prompt = prompt.split(' , ')[1]
                        print(f'Image {image_path} load prompt {prompt}')
                        break
    
    # optimize the prompt with clip interrogator
    if len(prefix) == 0:
        ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        logger.info(f'CLIP Interrogator for target {args.orig_path}')
        for orig_image in orig_images:
            prompt_clip = ci.interrogate(orig_image)
            print(prompt_clip)

            prefix_clip = prompt_clip.split(',')[0]
            print(prefix_clip)
            prefix.append(prefix_clip)

    if learned_prompt is None:
        logger.info(f'Soft prompt for target {args.orig_path}')
        learned_prompt = optimize_prompt(model, preprocess, args, device, 
                            target_images=orig_images, prefix=[i + connect for i in prefix])
        for i in range(len(image_paths)):
            logger.info(f'{image_paths[i]} - {prefix[i] + connect + learned_prompt}')


    #####################
    #   image editing   #
    #####################
    args.device = device
    args.my_token = ''
    args.low_resource = False
    args.num_ddim_steps = 50
    args.guidance_scale = 7.5
    args.max_num_words = 77
    logger.info(f'Image editing for target {args.orig_path}')

    # load stable diffusion model
    pipe, tokenizer = load_stable_diffusion(args)

    # target dummy prompt
    # replace_prompt = ""
    # for i in range(len(learned_prompt.split(' ')) - 1):
    #     replace_prompt += "<end_of_text> "

    for i in range(len(image_paths)):

        prompt = prefix[i] + connect + learned_prompt

        # start inversion
        null_inversion = NullInversion(args, pipe)
        (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_paths[i], prompt, offsets=offsets, verbose=True)

        if args.log_name == 'test':
            # save inversion
            prompts = [prompt]
            controller = AttentionStore(args)
            image_inv, x_t = run_and_display(args, pipe, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False)
            tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255. for img in \
                    [image_gt, image_inv[0]]]
            grid = make_grid(tensors, nrow=2)
            save_image(grid, f'{args.save_path}/null_inversion_{i}.png')


        if args.wm_prompt is None:
            prompt_target = prefix[i]
        else:
            prompt_target = prompt + connect + args.wm_prompt
        # image editing
        prompts = [
            prompt,         # original prompt
            prompt_target   # target prompt
        ]
        cross_replace_steps = {'default_': args.cross_replace_steps, }
        self_replace_steps = args.self_replace_steps
        # blend = 'man' if 'man' in prompt else 'woman' if 'woman' in prompt else 'person'
        blend_word = None # (((blend,), (blend,)))
        eq_params = None
        controller = make_controller(args, pipe.tokenizer, prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)

        images, _ = run_and_display(args, pipe, prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings)
        
        if args.log_name == 'test':
            # save comparison
            tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255. for img in \
                    [images[0], images[1]]]
            grid = make_grid(tensors, nrow=2)
            save_image(grid, f'{args.save_path}/combined_image_{i}.png')

        # save edited images
        save_image(torch.from_numpy(images[1]).permute(2, 0, 1).float() / 255.,
                   save_paths[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_len', type=int, default=4)
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--prompt_bs', type=int, default=1)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--print_step', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--clip_model', type=str, default='ViT-H-14')
    parser.add_argument('--clip_pretrain', type=str, default='laion2b_s32b_b79k')

    # customized params
    parser.add_argument('--orig_path', type=str, default="./data/n000050/set_A/")
    parser.add_argument('--save_path', type=str, default="./data/n000050/set_A_edited/")
    parser.add_argument('--log_name', type=str, default='test')
    parser.add_argument('--resume_name', type=str, default=None)
    parser.add_argument('--cross_replace_steps', type=float, default=0.8)
    parser.add_argument('--self_replace_steps', type=float, default=0.5)
    parser.add_argument('--wm_prompt', type=str, default=None)

    args = parser.parse_args()

    main(args)