#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <video_path> <audio_path> <video_out_path>"
    exit 1
fi

VIDEO_PATH=$1
AUDIO_PATH=$2
VIDEO_OUT_PATH=$3

python -m scripts.inference \
    --unet_config_path "configs/unet/stage2_512.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --enable_deepcache \
    --video_path "assets/demo1_video.mp4" \
    --audio_path "assets/demo1_audio.wav" \
    --video_out_path "video_out.mp4"
