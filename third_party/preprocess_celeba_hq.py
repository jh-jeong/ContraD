# This file is based on:
#   https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_partition.py

import os
import shutil
import pathlib

import pandas as pd
from shutil import copyfile
import PIL.Image


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


DATA_PATH = os.environ.get('DATA_DIR', 'data/')
DATA_PATH = pathlib.Path(f"{DATA_PATH}/CelebAMask-HQ/")

MAPPING = DATA_PATH / "CelebA-HQ-to-CelebA-mapping.txt"
IMAGES = DATA_PATH / "CelebA-HQ-img"
COPY_PATH = DATA_PATH / "CelebA-128-split"

# Destination paths
d_train_img = COPY_PATH / 'train/images'
d_test_img = COPY_PATH / 'test/images'

# Make folder
make_folder(d_train_img)
make_folder(d_test_img)

# Count num. images in each of destinations
train_count = 0
test_count = 0
val_count = 0

image_list = pd.read_csv(MAPPING, delim_whitespace=True, header=0)

for idx, x in enumerate(image_list.loc[:, 'orig_idx']):
    x = int(x)
    print(idx, x)
    src_img = PIL.Image.open(IMAGES / f'{idx}.jpg')
    dst_img = src_img.resize((128, 128), PIL.Image.ANTIALIAS)
    if x >= 182638:
        dst_img.save(d_test_img / f'{test_count}.jpg')
        test_count += 1
    elif x >= 162771 and x < 182638:
        if val_count < 176:
            dst_img.save(d_test_img / f'{test_count}.jpg')
            test_count += 1
        else:
            dst_img.save(d_train_img / f'{train_count}.jpg')
            train_count += 1
        val_count += 1
    else:
        dst_img.save(d_train_img / f'{train_count}.jpg')
        train_count += 1
    src_img.close()

print("Total: %d + %d = %d" % (train_count, test_count, train_count + test_count))