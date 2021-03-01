# derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py

import math
import tensorflow as tf
import numpy as np


def get_inception_score(images, softmax, sess, splits=10, batch_size=50, verbose=False):
    """Calculate inception score."""
    inps = images
    bs = batch_size
    preds = []
    n_batches = int(math.ceil(float(inps.shape[0]) / float(bs)))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        inp = inps[(i * bs):min((i + 1) * bs, inps.shape[0])]
        pred = sess.run(softmax, {'FID_Inception_Net/ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    if verbose:
        print(" done")
    return np.mean(scores), np.std(scores)


def get_softmax(sess, pool3):
    """Get softmax output."""
    w = sess.graph.get_operation_by_name("FID_Inception_Net/softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.reshape(tf.squeeze(pool3), (-1, w.shape[0])), w)
    softmax = tf.nn.softmax(logits)
    return softmax