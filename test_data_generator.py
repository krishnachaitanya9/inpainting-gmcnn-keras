import argparse
import os
import random
import shutil

if __name__ == "__main__":
    inpainting_dataset_path = "/home/shivababa/Documents/datasets/inpainting_dataset"
    test_path = f"{inpainting_dataset_path}/test/celeb"
    mask_path = f"{inpainting_dataset_path}/mask/mask"
    list_of_test_files = [test_path + '/' + f for f in os.listdir(test_path) if
                          os.path.isfile(os.path.join(test_path, f))]
    list_of_masks = [mask_path + '/' + f for f in os.listdir(mask_path) if
                     os.path.isfile(os.path.join(mask_path, f))]
    get_random_mask = lambda: random.choice(list_of_masks)
    i = 1
    for each_test in list_of_test_files:
        shutil.copyfile(each_test, "demo_tests/" + f"{i}.jpg")
        shutil.copyfile(get_random_mask(), "demo_tests/" + f"{i}.png")
        i += 1
        if i > 200:
            break
