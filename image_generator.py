# This function is responsible for the image generator module by utilizing the trained MSG-GAN models for generating
# the context-specific images. It can generate a number of images and saved it into a folder for further manipulations.

import tensorlayer as tl
import numpy as np
from data_msg import flags
from model_msg import get_generator
import random

def generate_images(no_of_images):
    # As the training of MSG-GAN was not perfect, three different generators are loaded to allow for more sample diversity.
    checkpoint = "good_msg_networks_256"
    G_1 = get_generator([None, flags.z_dim])
    tl.files.load_and_assign_npz(name='{}/G_msg_gan_2664.npz'.format(checkpoint), network=G_1)

    G_2 = get_generator([None, flags.z_dim])
    tl.files.load_and_assign_npz(name='{}/G_msg_gan_2959.npz'.format(checkpoint), network=G_2)

    G_3 = get_generator([None, flags.z_dim])
    tl.files.load_and_assign_npz(name='{}/G_msg_gan_2980.npz'.format(checkpoint), network=G_3)

    generators = [G_1, G_2, G_3]

    # Randomly select one generator at each iteration
    for i in range(no_of_images):
        G = random.sample(generators, 1)[0]
        z = np.random.normal(loc=0.0, scale=1.0, size=[1, flags.z_dim]).astype(np.float32)
        G.eval()
        result = G(z)[2]
        tl.visualize.save_images(result.numpy(), [1, 1], '{}/background{}.png'.format('generated_image', i))

# tl.visualize.save_image(result.numpy(),'{}/train.png'.format('generated_image'))
