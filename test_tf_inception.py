import os
import argparse
import pathlib

import tensorflow as tf
import numpy as np
from imageio import imread

import third_party.tf.fid as fid
import third_party.tf.inception_score as inc


parser = argparse.ArgumentParser(description='Testing script: FID / IS with TensorFlow backend')
parser.add_argument('images', type=str, help='Path to the directory of generated images')
parser.add_argument('stats', type=str, help='Path to precomputed .npz statistics')

parser.add_argument('--n_imgs', type=int, default=10000,
                    help='Number of images used to calculate the distances (default: 10000)')
parser.add_argument('--batch_size', type=int, default=500,
                    help='Batch size (default: 500)')
parser.add_argument('--gpu', type=str, default='',
                    help='GPU index to use (leave blank for CPU only)')
parser.add_argument('--inception_dir', type=str, default=None,
                    help='Directory to inception network')
parser.add_argument('--verbose', action='store_true',
                    help='Report status of program in console')

args = parser.parse_args()


def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def fwrite(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()


#-------------------------------------------------------------------------------
# Check parameters

if args.inception_dir is None:
    args.inception_dir = 'third_party/tf/resources'
PATH_INC = fid.check_or_download_inception(args.inception_dir)

PATH_DATA = args.images
if not os.path.exists(PATH_DATA):
    raise RuntimeError("Invalid path: %s" % PATH_DATA)
PATH_DATA = pathlib.Path(PATH_DATA)
data = list(PATH_DATA.glob('*.jpg')) + list(PATH_DATA.glob('*.png'))

if args.verbose:
    print("# DEBUG:::PATH_INC = " + str(PATH_INC))
    print("# DEBUG:::PATH_DATA = " + str(PATH_DATA))

if args.verbose and args.gpu != "":
    print("# Setting CUDA_VISIBLE_DEVICES to: " + str(args.gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

OUT_FN = str(PATH_DATA / f'test_inception_{str(np.random.randint(10000))}.csv')
init_logfile(OUT_FN, "FID,IS_MEAN,IS_STD")

#-------------------------------------------------------------------------------
if args.verbose:
    print("# Reading %d images..." % args.n_imgs, end="", flush=True)

# Read stats
f = np.load(args.stats)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

X = np.array([imread(str(data[i])).astype(np.float32) for i in range(args.n_imgs)])

if args.verbose:
    print("done")
    print("# image values in range [%.2f, %.2f]" % (X.min(), X.max()))

#-------------------------------------------------------------------------------
# Load inference model

fid.create_inception_graph(PATH_INC)
softmax = None

#-------------------------------------------------------------------------------

init = tf.global_variables_initializer()
sess = tf.Session()
with sess.as_default():
    sess.run(init)
    query_tensor = fid._get_inception_layer(sess)

    if softmax is None:
        softmax = inc.get_softmax(sess, query_tensor)

    # Calculate FID
    if args.verbose:
        print("#  -- Calculating FID...", flush=True)
    mu_gen, sigma_gen = fid.calculate_activation_statistics(X, sess, batch_size=args.batch_size, verbose=args.verbose)
    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    if args.verbose:
        print("#  -- FID = %.5f" % fid_value)

    # Calculate Inception score
    if args.verbose:
        print("#  -- Calculating Inception score...", flush=True)
    inc_mean, inc_std = inc.get_inception_score(X, softmax, sess, splits=10,
                                                batch_size=args.batch_size, verbose=args.verbose)
    if args.verbose:
        print("#  -- INC = %.5f +- %.5f" % (inc_mean, inc_std))

    fwrite(OUT_FN, f'{fid_value:.4f},{inc_mean:.4f},{inc_std:.4f}')