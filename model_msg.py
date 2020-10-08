# The following code is adopted and modified from from https://github.com/tensorlayer/dcgan/blob/master/data.py which is an implementation of DCGAN
# The model is modified so that the new features of MSG-GAN are incorporated into it.
# This function specifies the MSG-GAN model that is to be used for generating images.

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, Concat, MeanPool2d, Layer


# This function defines the MSG-GAN generator model.(Some layers are adopted from the
# # implementation above but the entire architecture is extended to cope with the requirements of MSG-GAN according to
# # the paper above.)
def get_generator(shape, gf_dim=16):  # Dimension of gen filters in first conv layer. [128]
    w_init = tf.random_normal_initializer(stddev=0.02)
    bias_init = tf.zeros_initializer()
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tf.nn.leaky_relu(x, 0.2)

    ni = Input(shape)
    nn = Dense(n_units=(gf_dim * 8 * 1 * 1), W_init=w_init, b_init=bias_init)(ni)
    nn = Reshape(shape=[-1, 1, 1, gf_dim * 8])(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 8, (4, 4), (1, 1), padding="VALID", W_init=w_init, b_init=bias_init)(nn)  # 4*4
    nn = Conv2d(gf_dim * 8, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)

    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 8, (5, 5), (2, 2), W_init=w_init, b_init=bias_init)(nn)  # 8*8
    nn = Conv2d(gf_dim * 8, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    nn = Conv2d(gf_dim * 8, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)

    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 8, (5, 5), (2, 2), W_init=w_init, b_init=bias_init)(nn)  # 16*16
    nn = Conv2d(gf_dim * 8, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    nn = Conv2d(gf_dim * 8, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)

    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 8, (5, 5), (2, 2), W_init=w_init, b_init=bias_init)(nn)  # 32*32
    nn = Conv2d(gf_dim * 8, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    nn = Conv2d(gf_dim * 8, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    model_1 = to_rgb(gf_dim * 8, nn)  # 32x32

    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 8, (5, 5), (2, 2), W_init=w_init, b_init=bias_init)(nn)  # 64*64
    nn = Conv2d(gf_dim * 4, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    nn = Conv2d(gf_dim * 4, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)

    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 4, (5, 5), (2, 2), W_init=w_init, b_init=bias_init)(nn)  # 128*128
    nn = Conv2d(gf_dim * 2, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    nn = Conv2d(gf_dim * 2, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    model_2 = to_rgb(gf_dim * 2, nn)  # 128*128

    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(gf_dim * 2, (5, 5), (2, 2), W_init=w_init, b_init=bias_init)(nn)  # 256*256
    nn = Conv2d(gf_dim, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    nn = Conv2d(gf_dim, (3, 3), (1, 1), act=lrelu, W_init=w_init)(nn)
    model_3 = to_rgb(gf_dim, nn)  # 256*256

    model = tl.models.Model(inputs=ni, outputs=[model_1, model_2, model_3])
    return model


# This method is for converting the intermediate activations into RGB scale for different scales of output images
# (This function is added by me)
def to_rgb(in_channels, nn):
    return Conv2d(3, (1, 1), act=lambda x: tf.nn.leaky_relu(x, 0.2), b_init=tf.zeros_initializer(),
                  in_channels=in_channels)(nn)
