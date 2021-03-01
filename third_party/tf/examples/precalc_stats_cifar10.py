#!/usr/bin/env python3

import numpy as np

import fid
import tensorflow as tf

########
# PATHS
########
out_path_train = 'resources/fid_stats_cifar10_train.npz'
out_path_test = 'resources/fid_stats_cifar10_test.npz'
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
(x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

print("%d/%d images found and loaded" % (len(x_train), len(x_test)))

print("Create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("OK")

print("Calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_train, sigma_train = fid.calculate_activation_statistics(x_train, sess, batch_size=100)
    mu_test, sigma_test = fid.calculate_activation_statistics(x_test, sess, batch_size=100)
    np.savez_compressed(out_path_train, mu=mu_train, sigma=sigma_train)
    np.savez_compressed(out_path_test, mu=mu_test, sigma=sigma_test)
print("Finished")
