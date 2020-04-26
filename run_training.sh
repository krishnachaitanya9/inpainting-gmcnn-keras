#!/bin/bash
# Usage: ./run_training.sh /FULL/PATH/TO/INPAINTING/DATASET
# The inpainting dataset folder structure should be something like below
#.
#├── mask
#│   └── mask
#├── test
#│   └── celeb
#└── train
#    └── celeb
# Inside the mask there should all masks in jpg/png formats
# Inside celeb folders there should all be images in jpg/png formats
python runner.py --train_path "$1/train" --test_path "$1/test" --mask_path "$1/mask" --experiment_name dl_project -warm_up_generator
python runner.py --train_path "$1/train" --test_path "$1/test" --mask_path "$1/mask" --experiment_name dl_project -from_weights
