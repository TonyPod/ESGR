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


RESULT_BASE_FOLDER = 'result_wgan/cifar-100/0.001/10000/from_scratch'
RESULT_SUB_FOLDER = ''
OUTPUT_SUB_FOLDER = 'cifar-100_0.001_10000'
OUTPUT_BASE_FOLDER = 'result_inception_score'

# read CIFAR-100
train_images, train_labels, train_one_hot_labels, \
    test_images, test_labels, test_one_hot_labels, \
    raw_images_train, raw_images_test, pixel_mean = cifar100.load_data(mean_subtraction=True)

class_names = cifar100.load_class_names()

output_folder = os.path.join(OUTPUT_BASE_FOLDER, OUTPUT_SUB_FOLDER)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for class_idx in range(0, 100):
    class_folder = os.path.join(RESULT_BASE_FOLDER, RESULT_SUB_FOLDER, 'class_%d' % (class_idx + 1))

    log_file = os.path.join(class_folder, 'log', 'log.pkl')
    last_sample_file = os.path.join(class_folder, 'samples', 'samples_10000.jpg')

    # load log file
    with open(log_file, 'rb') as f:
        log = pickle.load(f)
    inception_score = log['inception score']

    # real images of the current class
    raw_images_cur_class = raw_images_train[train_labels == class_idx]

    # load the generated image
    X = raw_images_cur_class[:64].reshape((64, 3, 32, 32))

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, n_samples/rows
    X = X.transpose(0, 2, 3, 1)
    h, w = X[0].shape[:2]
    img_real = np.zeros((h * nh, w * nw, 3))
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img_real[j*h:j*h+h, i*w:i*w+w] = x
    img_real = Image.fromarray(np.uint8(img_real))

    img_fake = Image.open(last_sample_file)
    img_fake = img_fake.crop((0, 0, 256, 256))

    img_merge = Image.new('RGB', (512, 256), 255)
    img_merge.paste(img_real, (0, 0))
    img_merge.paste(img_fake, (256, 0))

    img_merge.save(os.path.join(output_folder, 'class_%03d_%s_%f.jpg' % (class_idx+1, class_names[class_idx],
                                                                         inception_score[9999])))

    print('Class %d: %s' % (class_idx+1, class_names[class_idx]))