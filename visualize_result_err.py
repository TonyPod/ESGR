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


def vis_multiple(np_files_dict, interval, keys=None, dataset_name='CIFAR-100', output_name='comparison', ylim=(0, 100)):

    fontsize = 14

    if dataset_name == 'CIFAR-100':
        num_class = 100
    elif dataset_name == 'ImageNet Dogs':
        num_class = 120

    x = [i + interval for i in range(0, num_class, interval)]
    x_names = [str(i) for i in x]
    y = range(0, 110, 10)
    y_names = [str(i) + '%' for i in range(0, 110, 10)]

    if interval == 1:
        plt.figure(figsize=(12, 7), dpi=220)
    else:
        plt.figure(figsize=(8, 7), dpi=220)

    plt.gca().set_autoscale_on(False)

    plt.xlim(0, num_class)  # 限定横轴的范围
    assert len(ylim) == 2
    plt.ylim(ylim)  # 限定纵轴的范围

    if keys is not None:
        keys_order = keys
    else:
        keys_order = np_files_dict.keys()

    parent_folder_name = 'comparisons'
    # folder_name, _ = GetFileNameAndExt(output_name)
    # folder_name = os.path.join(parent_folder_name, folder_name)
    # if not os.path.exists(folder_name):
    #     os.makedirs(folder_name)

    color_map = {'Joint Training': '#1f77b4',
                 'iCaRL': '#ff7f0e',
                 'LwF': '#2ca02c',
                 'DGR': '#9467bd',
                 'ESGR-mix': '#d62728',
                 'ESGR-gens': '#8c564b',
                 'ESGR-reals': '#e377c2',
                 'ESGR-mix(imbalanced)': '#d62728',
                 'ESGR-mix(balanced)': '#8c564b',
                 'ESGR-mix(balanced v2)': '#e377c2',
                 'ESGR-gens(imbalanced)': '#9467bd',
                 'ESGR-gens(balanced)': '#ff7f0e',
                 }

    for method_name in keys_order:

        np_files = np_files_dict[method_name]

        aver_acc_over_time_mul = []

        for np_file in np_files:
            # if np_file.endswith('acc_over_time.npz'):
            #     np_data = np.load(np_file)
            #     aver_acc_over_time = np_data['aver_acc_over_time']
            #     aver_acc_over_time = np.insert(aver_acc_over_time, 0, 1)
            #     aver_acc_over_time = aver_acc_over_time * 100.0
            if np_file.endswith('top1_acc_list_cumul_icarl_cl%d.npy' % interval):
                np_data = np.load(np_file)
                if method_name == 'iCaRL':
                    aver_acc_over_time = np_data[:, 0, 0]
                elif method_name == 'LwF(sigmoid)':
                    aver_acc_over_time = np_data[:, 1, 0]

            # elif np_file.endswith('results_top1_acc_cl10.npz'):
            #     np_data = np.load(np_file)
            #     aver_acc_over_time = np_data['acc_list'][:, 0]
            # elif np_file.endswith('results_top1_acc_cl10.pkl'):
            #     with open(np_file, 'rb') as file_in:
            #         pkl_data = pickle.load(file_in)
            #         aver_acc_over_time = pkl_data['acc_list'][:, 0]
            #         # forget_score, learn_score, last_score = calc_forget_adapt_score(pkl_data['conf_mats_icarl'], interval, True)
            #         # visualize_conf_mat(pkl_data['conf_mats_icarl'][99], folder_name, method_name)
            elif np_file.endswith('acc_over_time.pkl'):
                with open(np_file, 'rb') as file_in:
                    pkl_data = pickle.load(file_in)
                    aver_acc_over_time_dict = pkl_data['aver_acc_over_time']
                    if len(aver_acc_over_time_dict) == (num_class / interval - 1):
                        aver_acc_over_time_dict[0] = 1.0
                    elif len(aver_acc_over_time_dict) == (num_class / interval):
                        pass
                    else:
                        print(np_file)
                        raise Exception()
                    aver_acc_over_time_x = sorted(aver_acc_over_time_dict.keys())
                    aver_acc_over_time = np.array([aver_acc_over_time_dict[i] for i in aver_acc_over_time_x])
                    # forget_score, learn_score, last_score = calc_forget_adapt_score(pkl_data['aver_acc_per_class_over_time'], interval, False)
                    # visualize_conf_mat(pkl_data['conf_mat_over_time'][99], folder_name, method_name)
            else:
                raise Exception()
            # print('%s: forget: %f, learn: %f, last: %f' % (method_name, forget_score, learn_score, last_score))

            if max(aver_acc_over_time) <= 1.0:
                aver_acc_over_time = aver_acc_over_time * 100.0

            aver_acc_over_time_mul.append(aver_acc_over_time)

        y_mean, y_std = calc_mean_std(np.array(aver_acc_over_time_mul))
        plt.errorbar(x[:len(y_mean)], y_mean, yerr=y_std, marker='.', label=method_name, color=color_map[method_name])
        # plt.plot(x[:len(aver_acc_over_time_mul)], aver_acc_over_time_mul, marker='.', label=method_name)

    plt.legend(fontsize=fontsize)  # 让图例生效
    if interval == 1:
        plt.xticks(x[9::10], x_names[9::10], rotation=45, fontsize=fontsize) # too crowded in x-axis for one-class adding
    else:
        plt.xticks(x, x_names, rotation=45, fontsize=fontsize)

    plt.yticks(y, y_names, fontsize=fontsize)
    plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)

    plt.xlabel("Number of classes", fontsize=fontsize)  # X轴标签
    plt.ylabel("Accuracy", fontsize=fontsize)  # Y轴标签
    plt.title(dataset_name)  # 标题

    # 水平参考线
    for i in range(10, 100, 10):
        plt.hlines(i, 0, num_class, colors = "lightgray", linestyles = "dashed")

    # plt.show()
    plt.savefig(os.path.join(parent_folder_name, output_name + '.png'))
    plt.savefig(os.path.join(parent_folder_name, output_name + '.svg'))
    plt.savefig(os.path.join(parent_folder_name, output_name + '.pdf'))


def visualize_conf_mat(conf_mat, folder_name, method_name):

    conf_mat = conf_mat / 50.0

    fig = plt.figure(figsize=(4, 4), dpi=220)
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(conf_mat), cmap=plt.cm.jet,
                    interpolation='Nearest', vmin=0, vmax=1.0)

    width, height = conf_mat.shape

    # for x in range(0, width, 20):
    #     for y in range(0, height, 20):
    #         ax.annotate(str(conf_mat[x][y]), xy=(y, x),
    #                     horizontalalignment='center',
    #                     verticalalignment='center')

    # cb = fig.colorbar(res)
    plt.xticks(range(9, 109, 10), range(10, 110, 10))
    plt.yticks(range(9, 109, 10), range(10, 110, 10))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(folder_name, method_name + '.png'), format='png')
    plt.savefig(os.path.join(folder_name, method_name + '.svg'), format='svg')


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

    return forget_score, learn_score, last_score

# if __name__ == '__main__':
#
#     # # Figure 1: CIFAR-100 nb_cl=1 LeNet
#     # np_files_dict = {
#     #     'iCaRL': [
#     #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/order_1/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl1.npy',
#     #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/order_2/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl1.npy',
#     #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/order_3/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl1.npy'],
#     #     'Joint Training': [
#     #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
#     #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
#     #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl'],
#     #     'Gens+reals': [
#     #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#     #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#     #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#     #     'Gens': [
#     #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
#     #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
#     #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl'],
#     #     'Reals': [
#     #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
#     #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
#     #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl'],
#     #     'Upperbound': [
#     #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
#     #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
#     #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl'],
#     #
#     # }
#     # keys = ['Upperbound', 'Joint Training', 'Gens+reals', 'Gens', 'Reals', 'iCaRL']
#     # output_name = 'cifar-100_nb_cl_1_1e-3_LeNet_multi_runs'
#     # vis_multiple(np_files_dict, 1, keys=keys, output_name=output_name, dataset_name='CIFAR-100')
#
#     # Figure 2: CIFAR-100 nb_cl=10 LeNet
#     np_files_dict = {
#         'iCaRL': [
#             '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_1/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
#             '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_2/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
#             '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_3/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy'],
#         'Joint Training': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl'],
#         'Gens+reals': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#         'Gens': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl'],
#         'Reals': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl'],
#         'Upperbound': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl'],
#         'LwF(sigmoid)': [
#             '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_1/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
#             '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_2/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
#             '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_3/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy'],
#         # My reproduction(lower than hybrid1)
#         # 'LwF(sigmoid)': [
#         #     './result/cifar-100_order_1/nb_cl_10/truncated_seperated/lenet_sigmoid_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl',
#         #     './result/cifar-100_order_2/nb_cl_10/truncated_seperated/lenet_sigmoid_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl',
#         #     './result/cifar-100_order_3/nb_cl_10/truncated_seperated/lenet_sigmoid_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl'],
#     }
#     keys = ['Upperbound', 'Joint Training', 'Gens+reals', 'Gens', 'Reals', 'iCaRL', 'LwF(sigmoid)']
#     output_name = 'cifar-100_nb_cl_10_1e-3_LeNet_multi_runs'
#     vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100', ylim=(10, 90))
#
#     # Figure 3: ImageNet Dogs nb_cl=10 ResNet
#     np_files_dict = {
#         # 'iCaRL': [
#         #     '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_1/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
#         # ],
#         'Joint Training': [
#             './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/joint_training/acc_over_time.pkl',
#         ],
#         'Gens+reals': [
#             './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp_proto_high_1.0-1.0_icarl_2400_smoothing_1.0/acc_over_time.pkl',
#         ],
#         'Gens': [
#             './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp/acc_over_time.pkl',
#         ],
#         'Reals': [
#             './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp_proto_high_1.0-1.0_icarl_2400_smoothing_1.0_ablation_epoch_based/acc_over_time.pkl',
#         ],
#         'Upperbound': [
#             './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_all/weight_decay_1e-05/base_lr_0.2/joint_training/acc_over_time.pkl',
#         ],
#         'LwF(sigmoid)': [
#             './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated_seperated/resnet_sigmoid_init_no/weight_decay_1e-05/base_lr_2.0/lwf/acc_over_time.pkl',
#         ]
#     }
#     keys = ['Upperbound', 'Joint Training', 'Gens+reals', 'Gens', 'Reals', 'LwF(sigmoid)']
#     output_name = 'imagenet_dogs_nb_cl_10_ResNet'
#     vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='ImageNet Dogs', ylim=(0, 60))
#
#     # Figure 4: CIFAR-100 nb_cl=10 LeNet comparisons on good and bad generators
#     np_files_dict = {
#         'Gens+reals(Good)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#         'Gens(Good)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl'],
#         'Gens+reals(Bad)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#         'Gens(Bad)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp/acc_over_time.pkl'],
#     }
#     keys = ['Gens+reals(Good)', 'Gens(Good)', 'Gens+reals(Bad)', 'Gens(Bad)']
#     output_name = 'cifar-100_nb_cl_10_LeNet_good_bad_gen_multi_runs'
#     vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100', ylim=(10, 90))
#
#     # Figure 5: CIFAR-100 nb_cl=10 LeNet comparisons on exemplars selection strategies
#     np_files_dict = {
#         'Mix(high, Good)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#         'Mix(low, Good)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#         'Mix(random, Good)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#         'Mix(high, Bad)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#         'Mix(low, Bad)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#         'Mix(random, Bad)': [
#             './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl',
#             './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl'],
#     }
#     keys = ['Mix(high, Good)', 'Mix(low, Good)', 'Mix(random, Good)', 'Mix(high, Bad)', 'Mix(low, Bad)', 'Mix(random, Bad)']
#     output_name = 'cifar-100_nb_cl_10_LeNet_exemplars_selection_multi_runs'
#     vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100', ylim=(10, 90))


if __name__ == '__main__':

    # # Figure 1: CIFAR-100 nb_cl=1 LeNet
    # np_files_dict = {
    #     'iCaRL': [
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/order_1/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl1.npy',
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/order_2/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl1.npy',
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/order_3/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl1.npy'],
    #     'ESGR-mix': [
    #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-gens': [
    #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl'],
    #     'ESGR-reals': [
    #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl'],
    #     'Joint Training': [
    #         './result/cifar-100_order_1/nb_cl_1/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_1/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_1/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl'],
    #
    # }
    # keys = ['Joint Training', 'iCaRL', 'ESGR-mix', 'ESGR-gens', 'ESGR-reals']
    # output_name = 'cifar-100_nb_cl_1_1e-3_LeNet_multi_runs'
    # vis_multiple(np_files_dict, 1, keys=keys, output_name=output_name, dataset_name='CIFAR-100')
    # #
    # # Figure 2: CIFAR-100 nb_cl=10 LeNet
    # np_files_dict = {
    #     'iCaRL': [
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_1/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_2/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_3/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy'],
    #     'ESGR-mix': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-gens': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl'],
    #     'ESGR-reals': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl'],
    #     'Joint Training': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_all/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl'],
    #     'LwF': [
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_1/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_2/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy',
    #         '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_10/order_3/LeNet/70/0.2/20/top1_acc_list_cumul_icarl_cl10.npy'],
    #     'DGR': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.02/adam_lr_0.0001/wgan_gp_deep_generative_replay_ratio_0.8/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.02/adam_lr_0.0001/wgan_gp_deep_generative_replay_ratio_0.8/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.02/adam_lr_0.0001/wgan_gp_deep_generative_replay_ratio_0.8/acc_over_time.pkl',
    #     ]
    #     # My reproduction(lower than hybrid1)
    #     # 'LwF(sigmoid)': [
    #     #     './result/cifar-100_order_1/nb_cl_10/truncated_seperated/lenet_sigmoid_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl',
    #     #     './result/cifar-100_order_2/nb_cl_10/truncated_seperated/lenet_sigmoid_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl',
    #     #     './result/cifar-100_order_3/nb_cl_10/truncated_seperated/lenet_sigmoid_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl'],
    # }
    # keys = ['Joint Training', 'iCaRL', 'LwF', 'DGR', 'ESGR-mix', 'ESGR-gens', 'ESGR-reals']
    # output_name = 'cifar-100_nb_cl_10_1e-3_LeNet_multi_runs'
    # vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100', ylim=(10, 90))
    #
    # Figure 3: ImageNet Dogs nb_cl=10 ResNet
    np_files_dict = {
        'ESGR-mix': [
            './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp_proto_high_1.0-1.0_icarl_2400_smoothing_1.0/acc_over_time.pkl',
        ],
        'ESGR-gens': [
            './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp/acc_over_time.pkl',
        ],
        'ESGR-reals': [
            './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp_proto_high_1.0-1.0_icarl_2400_smoothing_1.0_ablation_epoch_based/acc_over_time.pkl',
        ],
        'Joint Training': [
            './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_all/weight_decay_1e-05/base_lr_0.2/joint_training/acc_over_time.pkl',
        ],
        'LwF': [
            './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated_seperated/resnet_sigmoid_init_no/weight_decay_1e-05/base_lr_2.0/lwf/acc_over_time.pkl',
        ],
        'DGR': [
            './result/imagenet_64x64_dogs_order_1/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/wgan_gp_deep_generative_replay/acc_over_time.pkl',
        ]
    }
    keys = ['Joint Training', 'LwF', 'DGR', 'ESGR-mix', 'ESGR-gens', 'ESGR-reals']
    output_name = 'imagenet_dogs_nb_cl_10_ResNet'
    vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='ImageNet Dogs', ylim=(0, 60))

    # # Figure 4: CIFAR-100 nb_cl=10 LeNet comparisons on good and bad generators
    # np_files_dict = {
    #     'ESGR-mix(fair)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-gens(fair)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl'],
    #     'ESGR-mix(poor)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-gens(poor)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp/acc_over_time.pkl'],
    # }
    # keys = ['ESGR-mix(fair)', 'ESGR-mix(poor)', 'ESGR-gens(fair)', 'ESGR-gens(poor)']
    # output_name = 'cifar-100_nb_cl_10_LeNet_good_bad_gen_multi_runs'
    # vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100', ylim=(10, 90))
    #
    # # Figure 5: CIFAR-100 nb_cl=10 LeNet comparisons on exemplars selection strategies
    # np_files_dict = {
    #     'ESGR-mix(high, fair)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-mix(low, fair)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-mix(random, fair)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-mix(high, poor)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-mix(low, poor)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-mix(random, poor)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    # }
    # keys = ['ESGR-mix(high, fair)', 'ESGR-mix(low, fair)', 'ESGR-mix(random, fair)', 'ESGR-mix(high, poor)', 'ESGR-mix(low, poor)', 'ESGR-mix(random, poor)']
    # output_name = 'cifar-100_nb_cl_10_LeNet_exemplars_selection_multi_runs'
    # vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100', ylim=(10, 90))
    #
    # Figure 6: CIFAR-100 nb_cl=10 LeNet comparisons on balanced and unbalanced data of ESGR-gens and ESGR-mix
    # np_files_dict = {
    #     'ESGR-gens(imbalanced)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.pkl'],
    #     'ESGR-gens(balanced)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_new_class_gens/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_new_class_gens/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_new_class_gens/acc_over_time.pkl'],
    #     'ESGR-mix(imbalanced)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-mix(balanced)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    # }
    # keys = ['ESGR-mix(balanced)', 'ESGR-mix(imbalanced)', 'ESGR-gens(balanced)', 'ESGR-gens(imbalanced)']
    # output_name = 'cifar-100_nb_cl_10_LeNet_balanced_multi_runs'
    # vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100', ylim=(10, 90))

    # # Figure 7: CIFAR-100 nb_cl=10 LeNet comparisons on balanced and unbalanced data of ESGR-mix
    # np_files_dict = {
    #     'ESGR-mix(unbalanced)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-mix(balanced)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_high_1.0-1.0_icarl_2000/acc_over_time.pkl'],
    #     'ESGR-mix(balanced v2)': [
    #         './result/cifar-100_order_1/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_v2_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_2/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_v2_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
    #         './result/cifar-100_order_3/nb_cl_10/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_balanced_v2_high_1.0-1.0_icarl_2000/acc_over_time.pkl']
    # }
    # keys = ['ESGR-mix(unbalanced)', 'ESGR-mix(balanced)', 'ESGR-mix(balanced v2)']
    # output_name = 'cifar-100_nb_cl_10_LeNet_mix_balanced_multi_runs'
    # vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name, dataset_name='CIFAR-100', ylim=(10, 90))