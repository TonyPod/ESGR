# -*- coding:utf-8 -*-  

""" 
@time: 1/19/18 10:38 AM 
@author: Chen He 
@site:  
@file: utils_nin.py
@description:  
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim


def nin(images, num_classes=10, is_training=False,
        dropout_keep_prob=0.5,
        prediction_fn=slim.softmax,
        scope='NIN'):

    end_points = {}

    with tf.variable_scope(scope, 'NIN', [images, num_classes]):
        net = slim.conv2d(images, 192, [5, 5], scope='conv1', weights_regularizer=tf.nn.l2_loss)
        net = slim.conv2d(net, 160, [1, 1], scope='ccp1', weights_regularizer=tf.nn.l2_loss)
        net = slim.conv2d(net, 96, [1, 1], scope='ccp2', weights_regularizer=tf.nn.l2_loss)
        net = slim.max_pool2d(net, [3, 3], 2, padding='SAME', scope='pool1')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout3')

        net = slim.conv2d(net, 192, [5, 5], scope='conv2', weights_regularizer=tf.nn.l2_loss)
        net = slim.conv2d(net, 192, [1, 1], scope='ccp3', weights_regularizer=tf.nn.l2_loss)
        net = slim.conv2d(net, 192, [1, 1], scope='ccp4', weights_regularizer=tf.nn.l2_loss)
        net = slim.avg_pool2d(net, [3, 3], 2, padding='SAME', scope='pool2')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')

        net = slim.conv2d(net, 192, [3, 3], scope='conv3', weights_regularizer=tf.nn.l2_loss)
        net = slim.conv2d(net, 192, [1, 1], scope='ccp5', weights_regularizer=tf.nn.l2_loss)
        net = slim.conv2d(net, num_classes, [1, 1], scope='ccp6', weights_regularizer=tf.nn.l2_loss, activation_fn=None)
        net = slim.avg_pool2d(net, [8, 8], 1, scope='pool3')
        logits = slim.flatten(net)

    end_points['Logits'] = logits
    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points
