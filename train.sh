#!/bin/bash

source /etc/profile
module load anaconda/2023a
source activate control

python tutorial_train_sd21.py
