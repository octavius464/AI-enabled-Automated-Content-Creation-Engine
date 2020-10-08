# The following code is adopted and modified from https://github.com/tensorlayer/dcgan/blob/master/data.py
# It specifies the model hyperparameters and other parameters when running the MSG-GAN model

import numpy as np
import tensorflow as tf
import tensorlayer as tl

## enable debug logging
tl.logging.set_verbosity(tl.logging.DEBUG)

# (This class is entirely adopted from the github implementation above with some minor change of learning rates,
# batch_size, z_dim and directory paths.)
class FLAGS(object):
    def __init__(self):
        self.n_epoch = 3000  # "Epoch to train [25]"
        self.z_dim = 128  # "Num of noise value]"
        self.lrG = 0.0003  # "Learning rate of for adam [0.0002]") # TTUR
        self.lrD = 0.0003
        self.beta1 = 0.5  # "Momentum term of adam [0.5]")
        self.batch_size = 16  # "The number of batch images [64]")
        self.output_size = 256  # "The size of the output images to produce [64]")
        self.sample_size = 64  # "The number of sample images [64]")
        self.c_dim = 3  # "Number of image channels. [3]")
        self.save_every_epoch = 1  # "The interval of saveing checkpoints.")
        self.checkpoint_dir = "checkpoint_msg_gan"  # "Directory name to save the checkpoints [checkpoint]")
        self.sample_dir = "samples_msg_gan"  # "Directory name to save the image samples [samples]")
        assert np.sqrt(self.sample_size) % 1 == 0., 'Flag `sample_size` needs to be a perfect square'


flags = FLAGS()

tl.files.exists_or_mkdir(flags.checkpoint_dir)  # save model
tl.files.exists_or_mkdir(flags.sample_dir)  # save generated image
