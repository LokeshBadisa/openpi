#!/bin/bash
uv run examples/d3il/convert_d3il_data_to_lerobot.py
CUDA_VISIBLE_DEVICES=3 uv run scripts/compute_norm_stats.py --config-name pi05_d3il_lora_pytorch --skip_video_decoding