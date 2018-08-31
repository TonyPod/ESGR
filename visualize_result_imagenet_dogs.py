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

        plt.xlim(0, 120)  # 限定横轴的范围
        plt.ylim(0, 120)  # 限定纵轴的范围

        plt.plot(x, aver_acc_over_time * 100.0, marker='o', mec='r', mfc='w')

        plt.legend()  # 让图例生效
        plt.xticks(x, x_names, rotation=45)
        plt.yticks(y, y_names)
        plt.margins(0)
        # plt.subplots_adjust(bottom=0.15)

        plt.xlabel("Number of classes")  # X轴标签
        plt.ylabel("Accuracy")  # Y轴标签
        plt.title(dataset)  # 标题

        # 水平参考线
        for i in range(10, 100, 10):
            plt.hlines(i, 0, 120, colors = "lightgray", linestyles = "dashed")

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

        plt.xlim(0, 120)  # 限定横轴的范围
        plt.ylim(0, 100)  # 限定纵轴的范围

        plt.plot(x, aver_acc_over_time_y * 100.0, marker='o', mec='r', mfc='w')

        plt.legend()  # 让图例生效
        plt.xticks(x, x_names, rotation=45)
        plt.yticks(y, y_names)
        plt.margins(0)
        # plt.subplots_adjust(bottom=0.15)

        plt.xlabel("Number of classes")  # X轴标签
        plt.ylabel("Accuracy")  # Y轴标签
        plt.title("ImageNet-Dogs")  # 标题

        # 水平参考线
        for i in range(10, 100, 10):
            plt.hlines(i, 0, 120, colors="lightgray", linestyles="dashed")

        # plt.show()

        output_name = os.path.splitext(np_file)[0] + '.jpg'
        plt.savefig(output_name)
        plt.close()

def GetFileNameAndExt(filename):
    (filepath, tempfilename) = os.path.split(filename)
    (shotname, ext_name) = os.path.splitext(tempfilename)
    return shotname, ext_name

def vis_multiple(np_files_dict, interval, keys=None, output_name='comparison.jpg'):

    x = [i + interval for i in range(0, 120, interval)]
    x_names = [str(i) for i in x]
    y = range(0, 110, 10)
    y_names = [str(i) + '%' for i in range(0, 110, 10)]

    plt.figure(1, figsize=(18, 9), dpi=150)

    plt.xlim(0, 120)  # 限定横轴的范围
    plt.ylim(0, 100)  # 限定纵轴的范围

    if keys is not None:
        keys_order = keys
    else:
        keys_order = np_files_dict.keys()

    folder_name, _ = GetFileNameAndExt(output_name)
    parent_folder_name = 'comparisons'
    folder_name = os.path.join(parent_folder_name, folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for method_name in keys_order:

        np_file = np_files_dict[method_name]

        if np_file.endswith('acc_over_time.npz'):
            np_data = np.load(np_file)
            aver_acc_over_time = np_data['aver_acc_over_time']
            aver_acc_over_time = np.insert(aver_acc_over_time, 0, 1)
            aver_acc_over_time = aver_acc_over_time * 100.0
        elif np_file.endswith('top1_acc_list_cumul_icarl_cl1.npy'):
            np_data = np.load(np_file)
            aver_acc_over_time = np_data[:, 0, 0]
        elif np_file.endswith('results_top1_acc_cl10.npz'):
            np_data = np.load(np_file)
            aver_acc_over_time = np_data['acc_list'][:, 0]
        elif np_file.endswith('results_top1_acc_cl10.pkl'):
            with open(np_file, 'rb') as file_in:
                pkl_data = pickle.load(file_in)
                aver_acc_over_time = pkl_data['acc_list'][:, 0]
                forget_score, learn_score, last_score = calc_forget_adapt_score(pkl_data['conf_mats_icarl'], interval, True)
                visualize_conf_mat(pkl_data['conf_mats_icarl'][119], folder_name, method_name)
        elif np_file.endswith('acc_over_time.pkl'):
            with open(np_file, 'rb') as file_in:
                pkl_data = pickle.load(file_in)
                aver_acc_over_time_dict = pkl_data['aver_acc_over_time']
                aver_acc_over_time_x = sorted(aver_acc_over_time_dict.keys())
                aver_acc_over_time = np.array([aver_acc_over_time_dict[i] for i in aver_acc_over_time_x])
                forget_score, learn_score, last_score = calc_forget_adapt_score(pkl_data['aver_acc_per_class_over_time'], interval, False)
                visualize_conf_mat(pkl_data['conf_mat_over_time'][119], folder_name, method_name)
        else:
            raise Exception()
        print('%s: forget: %f, learn: %f, last: %f' % (method_name, forget_score, learn_score, last_score))


        if max(aver_acc_over_time) < 1.0:
            aver_acc_over_time = aver_acc_over_time * 100.0

        plt.figure(1)
        plt.plot(x[:len(aver_acc_over_time)], aver_acc_over_time, marker='.', label=method_name)

    plt.figure(1)
    plt.legend()  # 让图例生效
    plt.xticks(x, x_names, rotation=45)
    plt.yticks(y, y_names)
    plt.margins(0)
    # plt.subplots_adjust(bottom=0.15)

    plt.xlabel("Number of classes")  # X轴标签
    plt.ylabel("Accuracy")  # Y轴标签
    plt.title("CIFAR-100")  # 标题

    # 水平参考线
    for i in range(10, 100, 10):
        plt.hlines(i, 0, 120, colors = "lightgray", linestyles = "dashed")

    # plt.show()
    plt.savefig(os.path.join(parent_folder_name, output_name))


def visualize_conf_mat(conf_mat, folder_name, method_name):

    conf_mat = conf_mat / 50.0

    fig = plt.figure(figsize=(6, 6), dpi=120)
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
    plt.xticks(range(9, 129, 10), range(10, 130, 10))
    plt.yticks(range(9, 129, 10), range(10, 130, 10))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(folder_name, method_name + '.png'), format='png')


def calc_forget_adapt_score(acc_cls_dict, interval, is_conf_mat=False):
    if interval == 1 and not acc_cls_dict.has_key(0):
        acc_cls_dict[0] = np.array([120])

    if is_conf_mat:
        last_acc = np.diag(acc_cls_dict[119]) / 100.
    else:
        last_acc = acc_cls_dict[119] / 100.
    best_acc = []
    for i in range(0, 120, interval):
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

if __name__ == '__main__':
    # vis('result/cifar-100/lenet/0.02/acc_over_time.npz')
    np_files_dict = {
                     # 'Gens': './result/cifar-100/lenet/0.01/wgan_gp_rename/acc_over_time.npz',
                     # 'Gens(0.00001)': './result/cifar-100/lenet/0.01/wgan_gp_rename/acc_over_time.npz',
                     # 'Gens(0.0001)': './result/cifar-100/lenet/0.01/wgan_gp/acc_over_time.npz',
                     # 'Gens+reals(adaptive)': './result/cifar-100/lenet/0.01/wgan_gp_proto_auto-39.0-8.0_high_1.0-1.0_icarl_2000/acc_over_time.npz',
                     # 'Gens+reals(equal)': './result/cifar-100/lenet/0.01/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.npz',
                     # 'Reals': './result/cifar-100/lenet/0.01/wgan_gp_proto_ablation_iter_based/wgan_gp_proto_high_1.0-1.0_icarl_2000_run-3/acc_over_time.npz',
                     # 'Joint Training': './result/cifar-100/lenet/0.01/joint_training/acc_over_time.npz',
                     # 'iCaRL': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/unshuffled/LeNet/70/0.2/20_run-3/top1_acc_list_cumul_icarl_cl1.npy',
                     # 'iCaRL-50': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/LeNet/70/0.2/50/top1_acc_list_cumul_icarl_cl1.npy',
                     # 'iCaRL-100': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/LeNet/70/0.2/100/top1_acc_list_cumul_icarl_cl1.npy',
                     # 'iCaRL-200': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/LeNet/70/0.2/200/top1_acc_list_cumul_icarl_cl1.npy',
                     # 'iCaRL-500': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/LeNet/70/0.2/500/top1_acc_list_cumul_icarl_cl1.npy'
                     # 'Finetune CIFAR-100': './result/cifar-100/lenet/0.01/wgan_gp_finetune_cifar-100_0.0001_200000_all_classes/acc_over_time.npz',
                     # 'Finetune CIFAR-10': './result/cifar-100/lenet/0.01/wgan_gp_finetune_cifar-10_0.0001_200000_all_classes/acc_over_time.npz'
                     # 'High': './result/cifar-100/lenet/0.01/wgan_gp_proto_20_high_1.0-1.0/acc_over_time.npz',
                     # 'Low': './result/cifar-100/lenet/0.01/wgan_gp_proto_20_low_1.0-1.0/acc_over_time.npz',
                     # 'Random': './result/cifar-100/lenet/0.01/wgan_gp_proto_20_random_1.0-1.0/acc_over_time.npz',

                     # 'Gens': './result/cifar-100/lenet/0.01/adam_lr_0.001/wgan_gp/acc_over_time.npz',
                     # 'Gens+reals(equal, high)': './result/cifar-100/lenet/0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.npz',
                     # 'Gens+reals(equal, low)': './result/cifar-100/lenet/0.01/adam_lr_0.001/wgan_gp_proto_low_1.0-1.0_icarl_2000/acc_over_time.npz',
                     # 'Gens+reals(equal, random)': './result/cifar-100/lenet/0.01/adam_lr_0.001/wgan_gp_proto_random_1.0-1.0_icarl_2000/acc_over_time.npz',

                     # Adam_lr:
                     # 'Gens(0.001)': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.npz',
                     # 'Gens(0.0001)': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.0001/wgan_gp_rename/acc_over_time.npz',
                     # 'Gens+reals(0.001)': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.npz',
                     # 'Gens+reals(0.0001)': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.npz',

                     # LeNet
                     # 'iCaRL': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/unshuffled/LeNet/70/0.2/20_run-3/top1_acc_list_cumul_icarl_cl1.npy',
                     # 'Joint Training': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/joint_training/acc_over_time.npz',
                     # 'Gens': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.001/wgan_gp/acc_over_time.npz',
                     # 'Gens+reals': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.npz',

                     # NIN
                     # 'Joint Training': './result/cifar-100/nb_cl_1/nin/weight_decay_0.00001/base_lr_0.1/joint_training/acc_over_time.npz',
                     # 'Gens': './result/cifar-100/nb_cl_1/nin/weight_decay_0.00001/base_lr_0.1/adam_lr_0.001/wgan_gp/acc_over_time.npz',
                     # 'Gens+reals': './result/cifar-100/nb_cl_1/nin/weight_decay_0.00001/base_lr_0.1/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.npz',

                     # ResNet
                     # 'iCaRL': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/unshuffled/ResNet/70/2.0/20/top1_acc_list_cumul_icarl_cl1.npy',
                     # 'Joint Training': './result/cifar-100/nb_cl_1/resnet_5/weight_decay_0.002/base_lr_0.1/joint_training/acc_over_time.npz',
                     # 'Gens': './result/cifar-100/nb_cl_1/resnet_5/weight_decay_0.0001/base_lr_0.1/adam_lr_0.001/wgan_gp/acc_over_time.npz',
                     # 'Gens+reals': './result/cifar-100/nb_cl_1/resnet_5/weight_decay_0.0001/base_lr_0.1/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.npz',

                     # Ablation
                     # 'Reals': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_ablation_iter_based/wgan_gp_proto_high_1.0-1.0_icarl_2000_run-3/acc_over_time.npz',
                     # 'Gens': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.0001/wgan_gp_rename/acc_over_time.npz',
                     # 'Gens+reals': './result/cifar-100/nb_cl_1/lenet/weight_decay_0.00001/base_lr_0.01/adam_lr_0.0001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.npz',

                     # Currently used
                     # ImageNet 64x64
                     # 'iCaRL': '/home/hechen/iCaRL/iCaRL-Tensorflow/nb_cl_10/ResNet/60/2.0/20/results_top1_acc_cl10.pkl',
                     # 'Joint Training': './result/imagenet_64x64/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/joint_training/acc_over_time.pkl',
                     # 'Gens': './result/imagenet_64x64/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp/acc_over_time.pkl',
                     # 'Gens+reals': './result/imagenet_64x64/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
                     # 'Reals': './result/imagenet_64x64/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
                     # 'LwF(sigmoid)': './result/imagenet_64x64/nb_cl_10/truncated_seperated/resnet_sigmoid_init_no/weight_decay_1e-05/base_lr_2.0/lwf/acc_over_time.pkl',

                     # 'iCaRL': '/home/hechen/iCaRL/iCaRL-TheanoLasagne/result_nb_cl_1/unshuffled/LeNet/70/0.2/20_run-3/top1_acc_list_cumul_icarl_cl1.npy',
                     # 'Joint Training': './result/cifar-100/nb_cl_1/truncated/lenet_init_no/weight_decay_1e-05/base_lr_0.01/joint_training/acc_over_time.pkl',
                     # 'Gens': './result/cifar-100/nb_cl_1/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_run-4/acc_over_time.pkl',
                     # 'Gens+reals': './result/cifar-100/nb_cl_1/lenet_init_no/weight_decay_1e-05/base_lr_0.01/adam_lr_0.001/wgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',

                     # ImageNet 64x64
                     # 'iCaRL': '/home/hechen/iCaRL/iCaRL-Tensorflow/nb_cl_10/ResNet/60/2.0/20/results_top1_acc_cl10.pkl',
                     'Upper bound': './result/imagenet_64x64_dogs/nb_cl_10/truncated/resnet_init_all/weight_decay_1e-05/base_lr_0.2/joint_training/acc_over_time.pkl',
                     'Joint Training': './result/imagenet_64x64_dogs/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/joint_training/acc_over_time.pkl',
                     'Gens': './result/imagenet_64x64_dogs/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp/acc_over_time.pkl',
                     'Gens+reals': './result/imagenet_64x64_dogs/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp_proto_high_1.0-1.0_icarl_2000/acc_over_time.pkl',
                     'Reals': './result/imagenet_64x64_dogs/nb_cl_10/truncated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/adam_lr_0.0002/acwgan_gp_proto_high_1.0-1.0_icarl_2000_ablation_epoch_based/acc_over_time.pkl',
                     'LwF(sigmoid)': './result/imagenet_64x64_dogs/nb_cl_10/truncated_seperated/resnet_sigmoid_init_no/weight_decay_1e-05/base_lr_2.0/lwf/acc_over_time.pkl',
                     'LwF(softmax)': './result/imagenet_64x64_dogs/nb_cl_10/truncated_seperated/resnet_init_no/weight_decay_1e-05/base_lr_0.2/lwf/acc_over_time.pkl',

    }
    # keys = ['Without finetune', 'Joint Training', 'Finetune CIFAR-10', 'Finetune CIFAR-100']; output_name = 'comparison_finetune'
    # keys = ['High', 'Low', 'Random']; output_name = 'comparison_exemplar_selection.jpg'
    # keys = ['Gens', 'Reals', 'Gens+reals(adaptive)', 'Gens+reals(equal)']; output_name = 'comparison_ablation.jpg'
    # keys = ['Gens(0.001)', 'Gens(0.0001)']; output_name = 'comparison_adam_lr.jpg'
    # keys = ['Gens(0.00001)', 'Gens(0.0001)', 'Joint Training', 'iCaRL']; output_name = 'comparison_new'
    # keys = ['Gens+reals(adaptive)', 'Gens+reals(equal)']; output_name = 'comparison_adaptive'
    # keys = ['Joint Training', 'iCaRL', 'Gens', 'Gens+reals(equal, high)', 'Gens+reals(equal, low)', 'Gens+reals(equal, random)']; output_name = 'comparison_1e-3_lenet'
    # keys = ['Joint Training', 'iCaRL', 'Gens', 'Gens+reals']; output_name = 'comparison_cifar-100_1e-3_LeNet'
    # keys = ['Joint Training', 'Gens', 'Gens+reals']; output_name = 'comparison_1e-3_nin'
    # keys = ['Joint Training', 'iCaRL', 'Gens', 'Gens+reals']; output_name = 'comparison_1e-3_ResNet'

    # keys = ['Gens(0.001)', 'Gens(0.0001)', 'Gens+reals(0.001)', 'Gens+reals(0.0001)']; output_name = 'comparison_adam_lr.png'
    # keys = ['Reals', 'Gens', 'Gens+reals']; output_name = 'comparison_ablation_2(1e-4).png'
    # keys = ['iCaRL', 'Joint Training', 'LwF(sigmoid)', 'Gens', 'Gens+reals', 'Reals']; output_name = 'comparison_imagenet_64x64.png'
    keys = ['Upper bound', 'Joint Training', 'LwF(sigmoid)', 'LwF(softmax)', 'Gens', 'Gens+reals', 'Reals']; output_name = 'comparison_imagenet_dogs_64x64.png'

    vis_multiple(np_files_dict, 10, keys=keys, output_name=output_name)