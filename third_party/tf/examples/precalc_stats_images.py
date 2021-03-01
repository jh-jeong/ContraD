#!/usr/bin/env python3

import os
import glob
import numpy as np
from imageio import imread
import tensorflow as tf

import fid

########
# PATHS
########
DATA_DIR = os.environ.get('DATA_DIR', 'data/')

data_path = f"{DATA_DIR}/afhq/dog/train/images" # set path to training set images
output_path = 'resources/fid_stats_afhq_dog.npz'  # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = 'resources/'
print("Check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("OK")

# loads all images into memory (this might require a lot of RAM!)
print("Load images..", end=" " , flush=True)
image_list = glob.glob(os.path.join(data_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
print("%d images found and loaded" % len(images))

print("Create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("OK")

print("Calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("Finished")
