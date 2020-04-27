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
warm_up=true
inpainting=$1
if [ ! $warm_up ]
then
  echo "Running Warm up generator"
  if ! python runner.py --train_path "$inpainting/train" --test_path "$inpainting/test" --mask_path "$inpainting/mask" --experiment_name dl_project -warm_up_generator ; then
    echo "Warmup done"
  fi
fi
for i in {1..5}
do
  if python runner.py --train_path "$inpainting/train" --test_path "$inpainting/test" --mask_path "$inpainting/mask" --experiment_name dl_project -from_weights ; then
    echo "Training for $i th time"
  fi
done
