#%%
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch as th
# th.cuda.set_device(1)
import torch.distributed as dist

import sys
sys.path.append('..')

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

os.environ['OPENAI_LOGDIR'] = '.'

#%%
def main(model_path):
    # args = create_argparser().parse_args()
    args_dict = {
        'model_path' : model_path,
        'use_fp16' : True,
        'batch_size' : 16,
        'num_samples' : 16,
        'class_cond' : False,
        'use_ddim' : False,
        'image_size' : 256,
        'clip_denoised' : True,
    }
    args = argparse.Namespace(**args_dict)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model_config = {'image_size': 256,
                    'num_channels': 128,
                    'num_res_blocks': 2,
                    'num_heads': 4,
                    'num_heads_upsample': -1,
                    'num_head_channels': -1,
                    'attention_resolutions': '16,8', #'32, 16, 8',
                    'channel_mult': '',
                    'dropout': 0.0,
                    'class_cond': False,
                    'use_checkpoint': False,
                    'use_scale_shift_norm': True,
                    'resblock_updown': True,
                    'use_fp16': True,
                    'use_new_attention_order': False,
                    'learn_sigma': True,
                    'diffusion_steps': 1000,
                    'noise_schedule': 'linear',
                    'timestep_respacing': "200",
                    'use_kl': False,
                    'predict_xstart': False,
                    'rescale_timesteps': True,
                    'rescale_learned_sigmas': False}
    model, diffusion = create_model_and_diffusion(
        # **args_to_dict(args, model_and_diffusion_defaults().keys())
        **model_config
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"{args.model_path[-14:]}_samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            fig, ax = plt.subplots(4, 4, figsize=(8,8))
            for i, img in enumerate(arr):
                ax[i//4, i%4].imshow(img[:, :, 0], cmap='gray')
            fig.suptitle(f"{args.model_path[-14:]}")
            plt.savefig(out_path[:-4]+'.jpg')

    dist.barrier()
    logger.log("sampling complete")
    
    


# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         num_samples=16,
#         batch_size=20,
#         use_ddim=False,
#         model_path="",
#     )
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser

#%%
if __name__ == "__main__":
    # models_dir = '/mnt/ssd8/wonjun/code/inpaint/diffusion_ckpts/NIH_normals_nofp16_lr3e-4_1e-5at020000_wd0.05'
    # models = os.listdir(models_dir)
    # models = sorted([f for f in models if 'model' in f])
    # for model in models:
    #     model_path = os.path.join(models_dir, model)
    #     main(model_path)
    
    model_path = 'training_checkpoints/model575000.pt'
    main(model_path)

# %%