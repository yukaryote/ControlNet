#!/bin/bash

if [ $(whoami) == jpmcd ]; then
  GROUP_DIR="fastai"
else
  GROUP_DIR="fastai_shared"
fi

# Make Hugging Face cache folder on drive
HF_LOCAL_DIR="/state/partition1/user/$(whoami)/cache/huggingface"
mkdir -p $HF_LOCAL_DIR
# HF folder in shared file system
HF_USER_DIR="/home/gridsan/$(whoami)/.cache/huggingface"
rsync -a --ignore-existing $HF_USER_DIR/ ${HF_LOCAL_DIR}
export HF_HOME="${HF_LOCAL_DIR}"

python tool_add_control_sd21.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_ini.ckpt
