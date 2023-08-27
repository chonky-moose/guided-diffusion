#%%
"""
With the guided-diffusion library, setting CUDA_VISIBLE_DEVICES
does not work. Instead, we have to use torch.cuda.set_device(1)
to use cuda:1. I don't know why this is ¯\_(ツ)_/¯
"""
import os
os.environ['OPENAI_LOGDIR'] = r"training_checkpoints"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
torch.cuda.set_device(1)

# import argparse

from guided_diffusion import logger, dist_util
# from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop

from torch.utils.data import DataLoader
# import torch.distributed as dist

# from cxr_Datasets_normals import NIH_Dataset
from pnp_dataloaders import (create_2by2_for_mimic,
                              combine_into_labelled_df,
                              train_valid_split,
                              make_dataloaders)
#%%
def main():
    batch_size = 4
    use_fp16 = False
    
    # args = create_argparser().parse_args("")
    
    # model_config used by blended_diffusion
    model_config = {'image_size': 256,
                    'num_channels': 128,
                    'num_res_blocks': 2,
                    'num_heads': 4,
                    'num_heads_upsample': -1,
                    'num_head_channels': 64,
                    'attention_resolutions': '16, 8', #'32, 16, 8',
                    'channel_mult': '',
                    'dropout': 0.0,
                    'class_cond': False,
                    'use_checkpoint': False,
                    'use_scale_shift_norm': True,
                    'resblock_updown': True,
                    'use_fp16': use_fp16,
                    'use_new_attention_order': False,
                    'learn_sigma': True,
                    'diffusion_steps': 1000,
                    'noise_schedule': 'linear',
                    'timestep_respacing': "200",
                    'use_kl': False,
                    'predict_xstart': False,
                    'rescale_timesteps': True,
                    'rescale_learned_sigmas': False}
    
    dist_util.setup_dist()    
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        # **args_to_dict(args, model_and_diffusion_defaults().keys())
        **model_config
    )
    model.to('cuda')
    # model.to(dist_util.dev())
    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    logger.log("creating data loader...")
    
    # # Original data loader
    # data = load_data(
    #     data_dir = r'/media/wonjun/HDD2TB/rsna-pneumonia-detection-jpgs/All_sample',
    #     batch_size=1, #args.batch_size,
    #     image_size=256, #args.image_size,
    #     class_cond=False, #args.class_cond,
    # )
    
    # # Dataloader using cxr_Datasets
    # ds = NIH_Dataset('/mnt/ssd8/wonjun/data/nih',
    #                 'train', # use the train split of NIH CXR data
    #                 ratio=0, # Use only normal images
    #                 transforms=None)
    # dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
    #                 num_workers=os.cpu_count())
    # def load_data(dataloader):
    #     while True:
    #         yield from dataloader
    # data = load_data(dl)
    
    
    # # Dataloader for pnp
    mimic_path = '/mnt/ssd8/wonjun/data/mimic'
    labels_path = '../mimic-cxr-cardiac-device-labels3.csv'

    both_present, edema_only, device_only, both_absent = create_2by2_for_mimic(
            metadata_path = f'{mimic_path}/mimic-cxr-2.0.0-metadata.csv',
            labels_path = labels_path
        )
    positive_label_df, negative_label_df = combine_into_labelled_df(
            both_present, edema_only, device_only, both_absent,
            n_both_present=1171, n_target_only=1171,
            n_confounder_only=1171, n_both_absent=1171
        )
    train_df, valid_df = train_valid_split(positive_label_df,
                                        negative_label_df)
    dl, _ = make_dataloaders(
            train_df, valid_df,
            mimic_path=f"{mimic_path}/files",
            batch_size=batch_size
        )

    def load_data(dataloader):
        while True:
            yield from dataloader
    data = load_data(dl)
    
    
    logger.log("training...")
    trainloop = TrainLoop(
                    model=model,
                    diffusion=diffusion,
                    data=data,
                    batch_size=batch_size, #args.batch_size,
                    microbatch=-1, #args.microbatch,
                    lr=1e-7, #3e-4, #args.lr,
                    ema_rate=0.9999, #args.ema_rate,
                    log_interval=100, #args.log_interval,
                    save_interval=5000, #args.save_interval,
                    resume_checkpoint="training_checkpoints/model005000.pt", #args.resume_checkpoint,
                    use_fp16=use_fp16, #args.use_fp16,
                    fp16_scale_growth=0.001, #args.fp16_scale_growth,
                    schedule_sampler=schedule_sampler,
                    weight_decay=0.05, #args.weight_decay,
                    lr_anneal_steps=0, #args.lr_anneal_steps,
                )
    trainloop.run_loop()


# def create_argparser():
#     defaults = dict(
#         data_dir=r'Z:',
#         schedule_sampler="uniform",
#         lr=3e-5,
#         weight_decay=0.0,
#         lr_anneal_steps=0,
#         batch_size=1,
#         microbatch=-1,  # -1 disables microbatches
#         ema_rate="0.9999",  # comma-separated list of EMA values
#         log_interval=10,
#         save_interval=10000,
#         resume_checkpoint=r"C:\Users\lab402\Projects\guided-diffusion\LOGDIR_for_rsna\model200000.pt",
#         use_fp16=False,
#         fp16_scale_growth=1e-3,
#     )
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser
#%%
# args = create_argparser().parse_args("")
# model_config = args_to_dict(args, model_and_diffusion_defaults().keys())
# print(model_config)

#%%
if __name__ == "__main__":
    # __spec__ = None # for pdb
    main()
# %%
