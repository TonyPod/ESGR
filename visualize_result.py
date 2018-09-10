# -*- coding:utf-8 -*-  

""" 
@time: 1/25/18 9:48 AM 
@author: Chen He 
@site:  
@file: visualize_result.py
@description:  
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pylab import *
import re
import numpy as np
import os

import pickle


# Draw the accuracy curve of certain method
def vis(np_file, dataset='CIFAR-100'):

    if np_file.endswith('.npz'):

        np_data = np.load(np_file)

        aver_acc_over_time = np_data['aver_acc_over_time']

        aver_acc_over_time = np.insert(aver_acc_over_time, 0, 1)

        x = [i + 1 for i in range(aver_acc_over_time.shape[0])]
        x_names = [str(i) for i in x]
        y = range(0, 110, 10)
        y_names = [str(i)+'%' for i in range(0, 110, 10)]

        plt.figure(figsize=(18, 9), dpi=150)

        if dataset == 'ImageNetDogs':
            plt.xlim(0, 120)
        elif dataset == 'CIFAR-100':
            plt.xlim(0, 100)
        plt.ylim(0, 100)

        plt.plot(x, aver_acc_over_time * 100.0, marker='o', mec='r', mfc='w')

        plt.legend()
        plt.xticks(x, x_names, rotation=45)
        plt.yticks(y, y_names)
        plt.margins(0)
        # plt.subplots_adjust(bottom=0.15)

        plt.xlabel("Number of classes")
        plt.ylabel("Accuracy")
        plt.title(dataset)

        # Horizontal reference lines
        for i in range(10, 100, 10):
            plt.hlines(i, 0, 100, colors = "lightgray", linestyles = "dashed")

        # plt.show()

        output_name = os.path.splitext(np_file)[0] + '.jpg'
        plt.savefig(output_name)
        plt.close()

    elif np_file.endswith('.pkl'):

        with open(np_file, 'rb') as file:
            np_data = pickle.load(file)

        aver_acc_over_time_dict = np_data['aver_acc_over_time']
        aver_acc_over_time_x = sorted(aver_acc_over_time_dict.keys())

        x = [i + 1 for i in aver_acc_over_time_x]
        x_names = [str(i) for i in x]
        y = range(0, 110, 10)
        y_names = [str(i) + '%' for i in range(0, 110, 10)]

        aver_acc_over_time_y = np.array([aver_acc_over_time_dict[i] for i in aver_acc_over_time_x])

        plt.figure(figsize=(18, 9), dpi=150)

        if dataset == 'ImageNetDogs':
            plt.xlim(0, 120)
        elif dataset == 'CIFAR-100':
            plt.xlim(0, 100)
        plt.ylim(0, 100)

        plt.plot(x, aver_acc_over_time_y * 100.0, marker='o', mec='r', mfc='w')

        plt.legend()
        plt.xticks(x, x_names, rotation=45)
        plt.yticks(y, y_names)
        plt.margins(0)
        # plt.subplots_adjust(bottom=0.15)

        plt.xlabel("Number of classes")
        plt.ylabel("Accuracy")
        plt.title("CIFAR-100")

        # Horizontal reference lines
        for i in range(10, 100, 10):
            plt.hlines(i, 0, 100, colors="lightgray", linestyles="dashed")

        # plt.show()

        output_name = os.path.splitext(np_file)[0] + '.jpg'
        plt.savefig(output_name)
        plt.close()