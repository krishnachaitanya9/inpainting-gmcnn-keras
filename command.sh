# Use this command to run continuous 8 hrs training for models
python runner.py --train_path "$inpainting/train" --test_path "$inpainting/test" --mask_path "$inpainting/mask" --experiment_name dl_project -warm_up_generator && python runner.py --train_path "$inpainting/train" --test_path "$inpainting/test" --mask_path "$inpainting/mask" --experiment_name dl_project -from_weights
