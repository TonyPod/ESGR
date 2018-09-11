# Exemplar-Supported Generative Reproduction for Class Incremental Learning

This paper has been accepted as a poster at BMVC 2018

## Abstract

Incremental learning with deep neural networks often suffers from catastrophic forgetting, where newly learned patterns may completely erase the previous knowledge. A remedy is to review the old data (i.e. rehearsal) occasionally like humans to prevent forgetting. While recent approaches focus on storing historical data or the generator of old classes for rehearsal, we argue that they cannot fully and reliably represent old classes. In this paper, we propose a novel class incremental learning method called **Exemplar-Supported Generative Reproduction (ESGR)** that can better reconstruct memory of old classes and mitigate catastrophic forgetting. Specifically, we use Generative Adversarial Networks (GANs) to model the underlying distributions of old classes and select additional real exemplars as anchors to support the learned distribution. When learning from new class samples, synthesized data generated by GANs and real exemplars stored in the
memory for old classes can be jointly reviewed to mitigate catastrophic forgetting. By conducting experiments on CIFAR-100 and ImageNet-Dogs, we prove that our method has superior performance against state-of-the-arts.

## Requirements

Latest version of Tensorflow (implemented and tested under Tensorflow 1.3)

## Datasets 

For CIFAR-100, download the [python version of CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and extract it in a certain folder, let's say /home/user/cifar-100-python, then set data_path of cifar100.py to /home/user

For ImageNet-Dogs, download the [ImageNet64x64](https://pan.baidu.com/s/1k3tXWDTJ7tsoYZXStXu4pQ) (downsampled ImageNet according to https://patrykchrabaszcz.github.io/Imagenet32/; to facilitate the training process, I store pictures of a certain class in a seperate file) dataset first and change data_path to your folder path


## Citation

