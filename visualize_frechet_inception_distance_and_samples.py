# -*- coding:utf-8 -*-  

""" 
@time: 3/2/18 9:50 AM 
@author: Chen He 
@site:  
@file: visualize_inception_score_and_samples.py
@description:  
"""

import cifar100
import pickle
import os

from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from wgan.model_32x32 import GAN

import tensorflow as tf


RESULT_BASE_FOLDER = 'result_wgan/cifar-100/0.0001/10000'
# RESULT_SUB_FOLDER = 'from_scratch'
RESULT_SUB_FOLDER = 'finetune_cifar-10_0.0001_200000_all_classes'
OUTPUT_BASE_FOLDER = 'result_frechet_inception_distance'

# read CIFAR-100
train_images, train_labels, train_one_hot_labels, \
    test_images, test_labels, test_one_hot_labels, \
    raw_images_train, raw_images_test, pixel_mean = cifar100.load_data(mean_subtraction=True)

output_folder = os.path.join(OUTPUT_BASE_FOLDER, RESULT_SUB_FOLDER)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a session and load the GAN model
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True
graph_gen = tf.Graph()
sess_wgan = tf.Session(config=run_config, graph=graph_gen)

wgan_obj = GAN(sess_wgan, graph_gen,
               dataset_name='cifar-100',
               mode='wgan-gp',
               batch_size=64,
               dim=128,
               output_dim=3072,
               lambda_param=10,
               critic_iters=5,
               iters=10000,
               result_dir='result_wgan',
               checkpoint_interval=500,
               adam_lr=1e-4,
               adam_beta1=0.5,
               adam_beta2=0.9,
               finetune=False,
               finetune_from=-1,
               pretrained_model_base_dir='result_wgan_all_classes',
               pretrained_model_sub_dir='cifar-10/0.0001/200000/all_classes')

for class_idx in range(0, 100):

    class_output_folder = os.path.join(output_folder, '%d' % (class_idx + 1))
    if not os.path.exists(class_output_folder):
        os.makedirs(class_output_folder)

    reals_output_folder = os.path.join(class_output_folder, 'reals')
    if not os.path.exists(reals_output_folder):
        os.makedirs(reals_output_folder)

    gens_output_folder = os.path.join(class_output_folder, 'gens')
    if not os.path.exists(gens_output_folder):
        os.makedirs(gens_output_folder)

    # Generate fake images
    wgan_obj.load(class_idx)
    gen_samples_int, _, _ = wgan_obj.test(500)

    # Save fake images
    for gen_sample_idx in range(len(gen_samples_int)):
        gen_sample_int = gen_samples_int[gen_sample_idx]

        image = Image.fromarray(np.uint8(gen_sample_int.reshape((3, 32, 32)).transpose((1, 2, 0))))
        image.save(os.path.join(gens_output_folder, '%d.jpg' % (gen_sample_idx + 1)))

    # Load images of the current class
    real_samples_int = raw_images_train[train_labels == class_idx]

    # Save real images
    for real_sample_idx in range(len(real_samples_int)):
        real_sample_int = real_samples_int[real_sample_idx]

        image = Image.fromarray(np.uint8(real_sample_int.reshape((3, 32, 32)).transpose((1, 2, 0))))
        image.save(os.path.join(reals_output_folder, '%d.jpg' % (real_sample_idx + 1)))

    print('Class %d' % (class_idx + 1))