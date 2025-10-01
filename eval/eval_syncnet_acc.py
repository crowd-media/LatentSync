# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import os
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from einops import rearrange
from latentsync.models.stable_syncnet import StableSyncNet
from latentsync.data.syncnet_dataset import SyncNetDataset
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from accelerate.utils import set_seed


def main(config):
    set_seed(config.run.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.data.latent_space:
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-inpainting", subfolder="vae", revision="fp16", torch_dtype=torch.float16
        )
        vae.requires_grad_(False)
        vae.to(device)

    # Dataset and Dataloader setup
    dataset = SyncNetDataset(config.data.val_data_dir, config.data.val_fileslist, config)

    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        drop_last=False,
        worker_init_fn=dataset.worker_init_fn,
    )

    # Model
    syncnet = StableSyncNet(OmegaConf.to_container(config.model)).to(device)

    print(f"Load checkpoint from: {config.ckpt.inference_ckpt_path}")
    checkpoint = torch.load(config.ckpt.inference_ckpt_path, map_location=device, weights_only=True)

    syncnet.load_state_dict(checkpoint["state_dict"])
    syncnet.to(dtype=torch.float16)
    syncnet.requires_grad_(False)
    syncnet.eval()

    global_step = 0
    num_val_batches = len(test_dataloader) // config.data.batch_size
    progress_bar = tqdm(range(0, num_val_batches), initial=0, desc="Testing accuracy")

    num_correct_preds = 0
    num_total_preds = 0
    save_folder = "/data/users/carla/evaluations/latentsync/syncnet_debug/validate_frames/latentsync_input"
    os.makedirs(save_folder, exist_ok=True)

    save_folder_frames = os.path.join(save_folder, "frames")
    save_folder_audio = os.path.join(save_folder, "audio_samples")
    save_folder_label = os.path.join(save_folder, "labels")
    save_folder_sims = os.path.join(save_folder, "losses")

    os.makedirs(save_folder_frames, exist_ok=True)
    os.makedirs(save_folder_audio, exist_ok=True)
    os.makedirs(save_folder_label, exist_ok=True)
    os.makedirs(save_folder_sims, exist_ok=True)

    # while True:
    print(">>>>>>>",len(test_dataloader))
    for batch in test_dataloader:
        ### >>>> Test >>>> ###

        frames = batch["frames"].to(device, dtype=torch.float16)
        audio_samples = batch["audio_samples"].to(device, dtype=torch.float16)
        y = batch["y"].to(device, dtype=torch.float16).squeeze(1)

        if config.data.latent_space:
            frames = rearrange(frames, "b f c h w -> (b f) c h w")

            with torch.no_grad():
                frames = vae.encode(frames).latent_dist.sample() * 0.18215

            frames = rearrange(frames, "(b f) c h w -> b (f c) h w", f=config.data.num_frames)
        else:
            frames = rearrange(frames, "b f c h w -> b (f c) h w")

        if config.data.lower_half:
            height = frames.shape[2]
            frames = frames[:, :, height // 2 :, :]

        with torch.no_grad():
            torch.save(frames, os.path.join(save_folder_frames, f"frames_{global_step:d04}.pt"))
            torch.save(audio_samples, os.path.join(save_folder_audio, f"mel_chunk_{global_step:d04}.pt"))
            torch.save(y, os.path.join(save_folder_label, f"label_{global_step:d04}.pt"))
            vision_embeds, audio_embeds = syncnet(frames, audio_samples)

        sims = nn.functional.cosine_similarity(vision_embeds, audio_embeds)
        torch.save(sims, os.path.join(save_folder_sims, f"sims_{global_step:d04}.pt"))

        preds = (sims > 0.5).to(dtype=torch.float16)
        num_correct_preds += (preds == y).sum().item()
        num_total_preds += len(sims)

        progress_bar.update(1)
        global_step += 1
        # if global_step >= num_val_batches:

    progress_bar.close()
    accuracy = (num_correct_preds / num_total_preds)*100
    print(f"SyncNet Accuracy: {accuracy:.2f}%")
    np.save(os.path.join(save_folder, "accuracy.npy"), accuracy)
    torch.save(preds, os.path.join(save_folder, "preds.pt"))
    np.save(os.path.join(save_folder, "window_list.npy"), test_dataloader.dataset.window_indices)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code to test the accuracy of SyncNet")

    parser.add_argument("--config_path", type=str, default="configs/syncnet/syncnet_16_pixel_attn.yaml")
    args = parser.parse_args()

    # Load a configuration file
    config = OmegaConf.load(args.config_path)

    main(config)
