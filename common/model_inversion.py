# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import sys
from PIL import Image
from sys import stdout
import scipy
import scipy.misc
from pylearn2.datasets.preprocessing import ZCA
from pylearn2.expr.preprocessing import global_contrast_normalize

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def mnist_imshow(img):
    plt.imshow(img.reshape([28, 28]), cmap="gray")
    plt.axis('off')


def face_imshow(img):
    plt.imshow(img.reshape([112, 92]), cmap="gray")
    plt.axis('off')


def one_hot_preds(preds):
    t = np.argmax(preds, axis=1)
    r = np.zeros(preds.shape)
    for i in range(t.shape[0]):
        r[i, t[i]] = 1
    return r


def one_hot_class(a):
    b = np.zeros((len(a), np.max(a).astype(int) + 1), np.float32)
    b[np.arange(len(a)), a.astype(int)] = 1
    return b


# def unpack_facedataset(path='/home/yash/Documents/Attacks/Guillaume-Freisz-project/orl_faces', sz=None):
def unpack_facedataset(path='D:\www\graduate_expriment\common\datasets\orl_faces', sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as err:
                    print("I/O error({0}): {1}".format(err))
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c = c + 1
    X = (np.array(X).astype(np.float32) / 255).reshape(len(X), 92 * 112)
    y = np.array(y).astype(np.float32)
    X = X.reshape(40, 10, 112 * 92)
    y = y.reshape(40, 10)

    train_x, test_x = X[:, 0:7, :], X[:, 7:10, :]
    train_y, test_y = y[:, 0:7], y[:, 7:10]
    train_x, test_x, train_y, test_y = train_x.reshape(40 * 7, 112 * 92), test_x.reshape(40 * 3,
                                                                                         112 * 92), train_y.reshape(
        40 * 7), test_y.reshape(40 * 3)

    return train_x, test_x, one_hot_class(train_y), one_hot_class(test_y)


def normalize(img, prep, img_shape):
    img = prep.inverse(img.reshape(1, -1))[0]
    img /= np.abs(img).max()
    img = np.clip(img, -1., 1.)
    img = (img + 1.) / 2.
    img = global_contrast_normalize(img.reshape(1, -1) * 255, scale=55.)
    img = prep._gpu_matrix_dot(img - prep.mean_, prep.P_)
    return img.reshape(img_shape)


def post_process(img, prep, img_shape):
    # normalize without contrast_normalize and mean_subtract
    img = prep.inverse(img.reshape(1, -1))[0]
    img /= np.abs(img).max()
    img = np.clip(img, -1., 1.)
    img = (img + 1.) / 2.
    return img.reshape(img_shape)


# %%

class Model:
    def __init__(self, x, y_):

        in_dim = int(x.get_shape()[1])  # 10304 for Face dataset
        out_dim = int(y_.get_shape()[1])  # 40 for Face dataset
        self.x = x
        # switiching to a simple 2-layer network with relu
        W = weight_variable([in_dim, out_dim])
        b = bias_variable([out_dim])
        self.y = tf.matmul(x, W) + b  # output layer
        self.probs = tf.nn.softmax(self.y)
        self.class_inds = tf.argmax(self.probs, 1)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=self.y))

        class_ind_correct = tf.argmax(y_, 1)
        self.class_prob = (self.probs[0, tf.cast(class_ind_correct[0], tf.int32)])
        self.loss = tf.subtract(tf.constant(1.0), self.class_prob)

        self.grads = tf.gradients(self.cross_entropy, x)

        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, train_x, train_y, sess, test_x, test_y, num_iters, disp_freq=50):
        for i in range(num_iters):
            feed_dict = {x: train_x, y_: train_y}
            sess.run(self.train_step, feed_dict)
            if (i % disp_freq == 0):
                train_acc = self.test(train_x, train_y, sess)
                test_acc = self.test(test_x, test_y, sess)
                stdout.write("\r Train Acc. : %f    Test Acc. : %f" % (train_acc, test_acc))
                stdout.flush()
        stdout.write("\n")

    def test(self, test_x, test_y, sess):
        return (sess.run(self.accuracy, feed_dict={x: test_x, y_: test_y}))

    def invert(self, sess, num_iters, lam, img, pre_process, pred_cutoff=0.99, disp_freq=1):

        probs = self.preds(img)
        class_ind = sess.run(self.class_inds, feed_dict={x: [img]})[0]
        current_X = np.zeros(list(img.shape)[0]).astype(np.float32)
        Y = (one_hot_preds(probs)).astype(np.float32)
        best_X = np.copy(current_X)
        best_loss = 100000.0
        prev_losses = [100000.0] * 100

        for i in range(num_iters):
            feed_dict = {x: [current_X], y_: Y}
            der, current_loss = sess.run([self.grads, self.loss], feed_dict)
            current_X = np.clip(current_X - lam * (der[0][0]), 0.0, 1.0)
            current_X = normalize(current_X, pre_process, current_X.shape)
            probs = self.preds(current_X)[0]

            if current_loss < best_loss:
                best_loss = current_loss
                best_X = current_X

            if current_loss > 2 * max(prev_losses):
                print("\n Breaking due to gradient chaos!!")
                break

            if pred_cutoff < probs[class_ind]:
                print("\n Above Probability Criteria!: {0}".format(probs[class_ind]))
                break

            if i % disp_freq == 0:
                #                 plt.close()
                #                 face_imshow(post_process(current_X, pre_process, current_X.shape))
                #                 plt.show()
                stdout.write("\r Acc: %f and Loss: %f and Best Loss: %f" % (probs[class_ind], current_loss, best_loss))
                stdout.flush()

        stdout.write("\n")
        print('Loop Escape.')

        current_preds = self.preds(current_X)
        best_preds = self.preds(best_X)
        current_X = post_process(current_X, pre_process, current_X.shape)
        best_X = post_process(best_X, pre_process, best_X.shape)
        return current_X, current_preds, best_X, best_preds

    def preds(self, img):
        return sess.run(self.probs, feed_dict={x: [img]})


def perform_inversion(pre_process, images):
    for img in images:
        face_imshow(img)
        plt.title('Image-Class used for inversion.')
        plt.show()
        print('Predictions: ' + str((model.preds(img))))

        inv_img_last, inv_img_last_p, inv_img_best, inv_img_best_p = model.invert(sess, 100, 0.1, img,
                                                                                  pre_process=pre_process)

        face_imshow(inv_img_best)
        plt.title('Best Image after inversion.')
        plt.show()
        print('Predictions: ' + str(inv_img_best_p))

        face_imshow(inv_img_last)
        plt.title('Last Iteration Image after inversion.')
        plt.show()
        print('Predictions: ' + str(inv_img_last_p))


# %%

train_x, test_x, train_y, test_y = unpack_facedataset()  # 7:3 ratio for train:test

# GCN and ZCA object!!
train_x_normalized = global_contrast_normalize(train_x * 255, scale=55.)
zca = ZCA()
zca.fit(train_x_normalized)

# %%

x = tf.placeholder(tf.float32, shape=[None, 112 * 92])
y_ = tf.placeholder(tf.float32, shape=[None, 40])
model = Model(x, y_)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
model.train(train_x, train_y, sess, test_x, test_y, 250)

# %%

perform_inversion(zca, test_x[0::3])

# %%

"""
train_x, test_x, train_y, test_y  = unpack_facedataset() # 7:3 ratio for train:test 
zca = ZCA()
zca.fit(train_x[:2])
img = train_x[10]
img = normalize(img)
face_imshow(img,zca,img.shape)
"""

"""
from sys import stdout
from time import sleep
import numpy as np

a = np.array([10.2,.2,24.5,3,4,5,6,7,8,9,10,11])
for i in range(1,10):
    stdout.write("\r Hye there %d hi deer %d " % (a,i))
    stdout.flush()
    sleep(.25)
stdout.write("\n") # move the cursor to the next line

"""