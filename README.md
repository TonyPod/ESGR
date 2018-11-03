# Exemplar-Supported Generative Reproduction for Class Incremental Learning

This paper has been accepted as a poster at BMVC 2018 [[Paper]](http://bmvc2018.org/contents/papers/0325.pdf)[[Supplementary Material]](http://bmvc2018.org/contents/supplementary/pdf/0325_supp.pdf)

## Abstract

Incremental learning with deep neural networks often suffers from catastrophic forgetting, where newly learned patterns may completely erase the previous knowledge. A remedy is to review the old data (i.e. rehearsal) occasionally like humans to prevent forgetting. While recent approaches focus on storing historical data or the generator of old classes for rehearsal, we argue that they cannot fully and reliably represent old classes. In this paper, we propose a novel class incremental learning method called **Exemplar-Supported Generative Reproduction (ESGR)** that can better reconstruct memory of old classes and mitigate catastrophic forgetting. Specifically, we use Generative Adversarial Networks (GANs) to model the underlying distributions of old classes and select additional real exemplars as anchors to support the learned distribution. When learning from new class samples, synthesized data generated by GANs and real exemplars stored in the
memory for old classes can be jointly reviewed to mitigate catastrophic forgetting. By conducting experiments on CIFAR-100 and ImageNet-Dogs, we prove that our method has superior performance against state-of-the-arts.

## Requirements

Latest version of Tensorflow (implemented and tested under Tensorflow 1.3)

## Datasets 

For CIFAR-100, download the [python version of CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) and extract it in a certain folder, let's say /home/user/cifar-100-python, then set data_path of cifar100.py to /home/user

For ImageNet-Dogs, download the [ImageNet64x64](https://pan.baidu.com/s/1k3tXWDTJ7tsoYZXStXu4pQ) dataset first and change data_path in imagenet_64x64.py to your folder path. ImageNet64x64 is a downsampled ImageNet according to https://patrykchrabaszcz.github.io/Imagenet32/; to facilitate the training process, I store images of different classes in different pickle files


## Citation

If you use our codes, please cite it:

```bibtex
@inproceedings{he-bmvc2018,
  title = {Exemplar-Supported Generative Reproduction for Class Incremental Learning},
  author = {Chen He, Ruiping Wang, Shiguang Shan, and Xilin Chen},
  booktitle = {British Machine Vision Conference (BMVC)},
  year = {2018}
}
```

## Further

If you have any question, feel free to contact me. My email is chen.he@vipl.ict.ac.cn

P.S. The implementation of WGAN-GP is based on https://github.com/igul222/improved_wgan_training

Another thing just to remind you that my implementation of [Deep Generative Replay (DGR)](https://arxiv.org/abs/1705.08690) may be a little bit different with the original paper. In Page 4 Line 6, the author says "Here, the replayed target is past solver's response to replayed input". If I understand it correctly, the response here may be the softmax output using the old model, but the author doesn't mention that he uses "softmax" as the output layer throughout the paper, and if softmax is actually used, there are further problems, for example, what is the dimension of the past solver's response? (the dimension of the past solver's response may be different with that of the current solver's response) Since the author doesn't release the code, what I do is classify the generated samples over old classes, do zero-padding to make its dimension the same as the number of the seen classes (it looks similar to other ground-truth labels)
