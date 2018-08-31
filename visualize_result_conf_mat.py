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

def GetFileNameAndExt(filename):
    (filepath, tempfilename) = os.path.split(filename)
    (shotname, ext_name) = os.path.splitext(tempfilename)
    return shotname, ext_name

def calc_mean_std(result):
    mean_result = np.mean(result, axis=0)
    std_result = np.std(result, axis=0)
    return mean_result, std_result


def vis_multiple(np_file_dict, interval, keys=None, dataset_name='CIFAR-100', output_name='comparison.jpg'):

    assert dataset_name == 'CIFAR-100'

    if keys is not None:
        keys_order = keys
    else:
        keys_order = np_file_dict.keys()

    folder_name, _ = GetFileNameAndExt(output_name)
    parent_folder_name = 'comparisons'
    folder_name = os.path.join(parent_folder_name, folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for method_name in keys_order:

        np_file = np_file_dict[method_name]
        if np_file.endswith('acc_over_time.pkl'):
            with open(np_file, 'rb') as file_in:
                pkl_data = pickle.load(file_in)

            conf_mat = pkl_data['conf_mat_over_time'][99]
            forget_score, learn_score, last_score, forget_rate, auc = calc_forget_adapt_score(pkl_data['conf_mat_over_time'], interval, True)
            visualize_conf_mat(conf_mat, folder_name, method_name)
        elif np_file.endswith('conf_mat_icarl.pkl') or np_file.endswith('conf_mat_hb1.pkl') :
            with open(np_file, 'rb') as file_in:
                conf_mat_over_time = pickle.load(file_in)

            # convert keys from [0, 1, ..., 9] to [9, 19, 29, ..., 99]
            if len(conf_mat_over_time) == 10 and max(conf_mat_over_time.keys()) == 9:
                conf_mat_over_time_tmp = dict()
                for i in range(10):
                    conf_mat_over_time_tmp[i*10+9] = conf_mat_over_time[i]
                conf_mat_over_time = conf_mat_over_time_tmp

            conf_mat = conf_mat_over_time[99]
            forget_score, learn_score, last_score, forget_rate, auc = calc_forget_adapt_score(conf_mat_over_time, interval, True)
            visualize_conf_mat(conf_mat, folder_name, method_name)
        else:
            raise Exception()

        print('%s: forget: %f, learn: %f, last: %f, forget rate: %f, auc: %f' %
              (method_name, forget_score, learn_score, last_score, forget_rate, auc))

def visualize_conf_mat(conf_mat, folder_name, method_name):

    conf_mat = conf_mat / 50.0

    fontsize = 14
    fig = plt.figure(figsize=(6, 6), dpi=220)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(conf_mat), cmap=plt.cm.jet,
                    vmin=0, vmax=1.0)

    width, height = conf_mat.shape

    # for x in range(0, width, 20):
    #     for y in range(0, height, 20):
    #         ax.annotate(str(conf_mat[x][y]), xy=(y, x),
    #                     horizontalalignment='center',
    #                     verticalalignment='center')

    cb = fig.colorbar(res)
    plt.title(method_name)
    plt.xticks(range(9, 109, 10), range(10, 110, 10), fontsize=fontsize)
    plt.yticks(range(9, 109, 10), range(10, 110, 10), fontsize=fontsize)
    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.savefig(os.path.join(folder_name, method_name + '.svg'), format='svg')
    plt.savefig(os.path.join(folder_name, method_name + '.pdf'), format='pdf')


def calc_forget_adapt_score(acc_cls_dict, interval, is_conf_mat=False):
    if interval == 1 and not acc_cls_dict.has_key(0):
        acc_cls_dict[0] = np.array([100])

    if is_conf_mat:
        last_acc = np.diag(acc_cls_dict[99]) / 100.
    else:
        last_acc = acc_cls_dict[99] / 100.
    best_acc = []
    for i in range(0, 100, interval):
        cls_idx = i + interval
        if is_conf_mat:
            best_acc[i:cls_idx] = np.diag(acc_cls_dict[cls_idx-1])[i:cls_idx] / 100.
        else:
            best_acc[i:cls_idx] = acc_cls_dict[cls_idx-1][i:cls_idx] / 100.
    best_acc = np.array(best_acc)

    forget_score = np.mean(best_acc - last_acc)
    learn_score = np.mean(best_acc)
    last_score = np.mean(last_acc)

    auc = np.mean([np.mean(np.diag(acc)) for acc in acc_cls_dict.values()]) / 100.
    forget_rate = forget_score / learn_score
    return forget_score, learn_score, last_score, forget_rate, auc


if __name__ == '__main__':

    # CIFAR-100 nb_cl=10
    np_files_dict = {
        'ESGR-gens(balanced)': './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_new_class_gens/acc_over_time.pkl',
        # 'iCaRL': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_1/LeNet/70/0.2/20/conf_mat_icarl.pkl',
        # 'ESGR-mix': './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
        'ESGR-mix(balanced)': './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
        # 'ESGR-gens': './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
        # 'ESGR-reals': './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
        # 'Joint Training': './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
        # 'LwF(sigmoid)': './result/cifar-100_order_1/nb_cl_10/truncated_seperated/lenet_sigmoid_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl',
        # 'LwF': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_1/LeNet/70/0.2/20_run-7/conf_mat_hb1.pkl',
        # 'DGR': './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.02/adam_lr_0.0001/wgan_gp_deep_generative_replay_ratio_0.8/acc_over_time.pkl',
    }
    # keys = ['Joint Training', 'ESGR-mix', 'ESGR-gens', 'ESGR-reals', 'iCaRL', 'LwF', 'DGR']
    # keys = ['DGR']
    # keys = ['Gens(balanced)']
    keys = ['ESGR-gens(balanced)']
    output_name = 'cifar-100_nb_cl_10_1e-3_LeNet_multi_runs'
    vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100')

    # # CIFAR-100 nb_cl=1
    # np_files_dict = {
    #     'iCaRL': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/order_1/LeNet/70/0.2/20/conf_mat_icarl.pkl',
    #     'Joint Training': './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
    #     'Mix': './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #     'Gens': './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #     'Reals': './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
    #     'Upperbound': './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
    #     # 'LwF(sigmoid)': './result/cifar-100_order_1/nb_cl_10/truncated_seperated/lenet_sigmoid_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl',
    #     'LwF(sigmoid)': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/order_1/LeNet/70/0.2/20_run-7/conf_mat_hb1.pkl',
    # }
    # keys = ['Upperbound', 'Joint Training', 'Mix', 'Gens', 'Reals', 'iCaRL', 'LwF(sigmoid)']
    # output_name = 'cifar-100_nb_cl_1_1e-3_LeNet_multi_runs'
    # vis_multiple(np_files_dict, 1, keys=keys, output_name=output_name, dataset_name='CIFAR-100')