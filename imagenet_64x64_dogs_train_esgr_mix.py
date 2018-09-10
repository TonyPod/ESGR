# -*- coding:utf-8 -*-

'''
Implememtation of the proposed method ESGR-mix
'''

import tensorflow as tf
tf.set_random_seed(1993)

import utils_resnet_64x64

import numpy as np
np.random.seed(1993)

import os
import pprint
import visualize_result

from sklearn.metrics import confusion_matrix

import pickle

from wgan.model_64x64_conditional import WGAN64x64

import imagenet_64x64

flags = tf.app.flags

flags.DEFINE_string("dataset", "imagenet_64x64_dogs", "The name of dataset")

# Generator: W-GAN_GP
flags.DEFINE_integer("dim", 64, "This overfits substantially; you're probably better off with 64 [128]")
flags.DEFINE_integer("lambda_param", 10, "Gradient penalty lambda hyperparameter [10]")
flags.DEFINE_integer("critic_iters", 5, "How many critic iterations per generator iteration [5]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_integer("iters", 20000, "How many generator epochs to train for [64]")
flags.DEFINE_integer("output_dim", 64*64*3, "Number of pixels in CIFAR10 (3*32*32) [3072]")
flags.DEFINE_string("mode", 'wgan-gp', "Valid options are dcgan, wgan, or wgan-gp")
flags.DEFINE_string("result_dir_cwgan", 'result_acwgan', "")
flags.DEFINE_integer("gan_save_interval", 1000, 'interval to save a checkpoint(number of iters)')
flags.DEFINE_float("adam_lr", 2e-4, 'default: 1e-4')
flags.DEFINE_boolean("use_decay", True, "")
flags.DEFINE_boolean("conditional", True, "")
flags.DEFINE_boolean("acgan", True, "")
flags.DEFINE_float("acgan_scale", 1., '')
flags.DEFINE_float("acgan_scale_g", 0.1, '')
flags.DEFINE_boolean("normalization_g", True, "")
flags.DEFINE_boolean("normalization_d", False, "")
flags.DEFINE_integer("gen_bs_multiple", 2, "")
flags.DEFINE_integer("n_gpus", 1, "")

flags.DEFINE_boolean("only_gen_no_cls", False, "")

flags.DEFINE_boolean('use_momentum', True, 'Gradient descent or gradient descent with momentum')
flags.DEFINE_float('momentum', 0.9, '')

flags.DEFINE_integer('epochs_per_category', 60, 'number of epochs for each training session')
flags.DEFINE_integer('train_batch_size', 128, 'training batch size')
flags.DEFINE_integer('test_batch_size', 128, 'test batch size')

flags.DEFINE_float('base_lr', .2, '2. for sigmoid, .2 for softmax')
flags.DEFINE_float('weight_decay', 0.00001, '0.00001')
flags.DEFINE_float('lr_factor', 5., '')
flags.DEFINE_integer('display_interval', 20, '')
flags.DEFINE_integer('test_interval', 100, '')
lr_strat = [20, 30, 40, 50]

flags.DEFINE_string('result_dir', 'result/', '')

# Network architecture
flags.DEFINE_string('network_arch', 'resnet', 'resnet')
# flags.DEFINE_integer('num_resblocks', 5, 'number of resblocks when ResNet is used')
flags.DEFINE_boolean('use_softmax', True, 'True: softmax; False: sigmoid')
flags.DEFINE_boolean('no_truncate', False, '')

# Add how many classes every time
flags.DEFINE_integer('nb_cl', 10, '')

# DEBUG
flags.DEFINE_integer('from_class_idx', 0, 'starting category_idx')
flags.DEFINE_integer('to_class_idx', 119, 'ending category_idx')

# Init params when new nodes added
flags.DEFINE_string('init_strategy', 'no', 'no | last | all')

# Order file
flags.DEFINE_string('order_file', 'order_1', '')

# Data aug
flags.DEFINE_boolean('flip', False, '')

# PROTO
flags.DEFINE_integer('num_exemplars_per_class', 20, 'enable only when memory_constrained is False')
flags.DEFINE_string('exemplar_select_criterion', 'high', 'high | low | random')
flags.DEFINE_float('proto_weight', 1., 'the weight of the proto')
flags.DEFINE_float('gen_weight', 1., 'the weight of the generated samples')
flags.DEFINE_float('label_smoothing', 1., 'the smoothed label for generated samples')
flags.DEFINE_boolean('reorder_exemplars', False, '')
flags.DEFINE_boolean('visualize_exemplars', True, 'save exemplars into pngs files')

# PROTO AUTO
flags.DEFINE_boolean('auto_choose_num_exemplars', False, 'choose the samples with the highest prob as exemplars or not')
flags.DEFINE_boolean('memory_constrained', True, '')
flags.DEFINE_integer('memory_upperbound', 2400, '2000 for the first 100 classes of ImageNet')
flags.DEFINE_float('auto_param1', 39, '')
flags.DEFINE_float('auto_param2', 8, '')

# Generate more and select distinctive ones
flags.DEFINE_boolean('gen_more_and_select', False, '')
flags.DEFINE_integer('gen_how_many', 2000, '')

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):

    pp.pprint(flags.FLAGS.__flags)

    order = []
    with open('imagenet_64x64_dogs_%s.txt' % FLAGS.order_file) as file_in:
        for line in file_in.readlines():
            order.append(int(line))
    order = np.array(order)

    assert FLAGS.mode == 'wgan-gp'

    NUM_CLASSES = 120
    NUM_TEST_SAMPLES_PER_CLASS = 50
    NUM_TRAIN_SAMPLES_PER_CLASS = 1300 # around 1300

    if not FLAGS.only_gen_no_cls:

        def build_cnn(inputs, is_training):
            train_or_test = {True: 'train', False: 'test'}
            if FLAGS.network_arch == 'resnet':
                logits, end_points = utils_resnet_64x64.ResNet(inputs, train_or_test[is_training],
                                                               num_outputs=NUM_CLASSES,
                                                               alpha=0.0,
                                                               scope=('ResNet-' + train_or_test[is_training]))
            else:
                raise Exception()
            return logits, end_points

        # save all intermediate result in the result_folder
        method_name = '_'.join(os.path.basename(__file__).split('.')[0].split('_')[4:])
        method_name += '_gen_%d_and_select' % FLAGS.gen_how_many if FLAGS.gen_more_and_select else ''
        method_name += '_auto-%.1f-%.1f' % (FLAGS.auto_param1, FLAGS.auto_param2) \
            if FLAGS.auto_choose_num_exemplars else ('_%d' % FLAGS.num_exemplars_per_class if not FLAGS.memory_constrained else '')
        method_name += '_%s' % FLAGS.exemplar_select_criterion
        method_name += '_%.1f-%.1f' % (FLAGS.proto_weight, FLAGS.gen_weight)
        method_name += '_icarl_%d' % FLAGS.memory_upperbound if FLAGS.memory_constrained else ''
        method_name += '_reorder' if FLAGS.reorder_exemplars else ''
        method_name += '_smoothing_%.1f' % FLAGS.label_smoothing

        cls_func = '' if FLAGS.use_softmax else '_sigmoid'
        result_folder = os.path.join(FLAGS.result_dir, FLAGS.dataset + ('_flip' if FLAGS.flip else '') + '_' + FLAGS.order_file,
                                     'nb_cl_' + str(FLAGS.nb_cl),
                                     'non_truncated' if FLAGS.no_truncate else 'truncated',
                                     FLAGS.network_arch + cls_func + '_init_' + FLAGS.init_strategy,
                                     'weight_decay_' + str(FLAGS.weight_decay),
                                     'base_lr_' + str(FLAGS.base_lr),
                                     'adam_lr_' + str(FLAGS.adam_lr),
                                     method_name)

        if os.path.exists(result_folder):
            temp_i = 2
            while True:
                result_folder_mod = result_folder + '_run-' + str(temp_i)
                if not os.path.exists(result_folder_mod):
                    result_folder = result_folder_mod
                    break
                temp_i += 1
        os.makedirs(result_folder)
        print('Result folder: %s' % result_folder)

        graph_cls = tf.Graph()
        with graph_cls.as_default():
            '''
            Define variables
            '''
            batch_images = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
            batch = tf.Variable(0, trainable=False)
            learning_rate = tf.placeholder(tf.float32, shape=[])

            '''
            Network output mask
            '''
            mask_output = tf.placeholder(tf.bool, shape=[NUM_CLASSES])

            '''
            Old and new ground truth
            '''
            one_hot_labels_truncated = tf.placeholder(tf.float32, shape=[None, None])

            '''
            Define the training network
            '''
            train_logits, _ = build_cnn(batch_images, True)
            train_masked_logits = tf.gather(train_logits, tf.squeeze(tf.where(mask_output)),
                                            axis=1)  # masking operation
            train_masked_logits = tf.cond(tf.equal(tf.rank(train_masked_logits), 1),
                                          lambda: tf.expand_dims(train_masked_logits, 1),
                                          lambda: train_masked_logits)  # convert to (N, 1) if the shape is (N,), otherwise softmax would output wrong values
            # Train accuracy(since there is only one class excluding the old recorded responses, this accuracy is not very meaningful)
            train_pred = tf.argmax(train_masked_logits, 1)
            train_ground_truth = tf.argmax(one_hot_labels_truncated, 1)
            correct_prediction = tf.equal(train_pred, train_ground_truth)
            train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            train_batch_weights = tf.placeholder(tf.float32, shape=[None])

            reg_weights = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = FLAGS.weight_decay * tf.add_n(reg_weights)

            '''
            More Settings
            '''
            if FLAGS.use_softmax:
                empirical_loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels_truncated,
                                                                 logits=train_masked_logits,
                                                                 weights=train_batch_weights)
            else:
                empirical_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=one_hot_labels_truncated,
                                                                 logits=train_masked_logits,
                                                                 weights=train_batch_weights)

            loss = empirical_loss + regularization_loss
            if FLAGS.use_momentum:
                opt = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum).minimize(loss, global_step=batch)
            else:
                opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch)

            '''
            Define the testing network
            '''
            test_logits, _ = build_cnn(batch_images, False)
            test_masked_logits = tf.gather(test_logits, tf.squeeze(tf.where(mask_output)), axis=1)
            test_masked_logits = tf.cond(tf.equal(tf.rank(test_masked_logits), 1),
                                         lambda: tf.expand_dims(test_masked_logits, 1),
                                         lambda: test_masked_logits)
            test_masked_prob = tf.nn.softmax(test_masked_logits)
            test_pred = tf.argmax(test_masked_logits, 1)
            test_accuracy = tf.placeholder(tf.float32)

            '''
            Copy network (define the copying op)
            '''
            if FLAGS.network_arch == 'resnet':
                all_variables = tf.get_collection(tf.GraphKeys.WEIGHTS)
            else:
                raise Exception('Invalid network architecture')
            copy_ops = [all_variables[ix + len(all_variables) // 2].assign(var.value()) for ix, var in
                        enumerate(all_variables[0:len(all_variables) // 2])]

            '''
            Init certain layers when new classes added
            '''
            init_ops = tf.no_op()
            if FLAGS.init_strategy == 'all':
                init_ops = tf.global_variables_initializer()
            elif FLAGS.init_strategy == 'last':
                if FLAGS.network_arch == 'resnet':
                    init_vars = [var for var in tf.global_variables() if 'fc' in var.name and 'train' in var.name]
                init_ops = tf.initialize_variables(init_vars)

            '''
            Create session
            '''
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config, graph=graph_cls)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()

        '''
        Summary
        '''
        train_loss_summary = tf.summary.scalar('train_loss', loss)
        train_acc_summary = tf.summary.scalar('train_accuracy', train_accuracy)
        test_acc_summary = tf.summary.scalar('test_accuracy', test_accuracy)

        summary_dir = os.path.join(result_folder, 'summary')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        train_summary_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train'), sess.graph)
        test_summary_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'test'))

        iteration = 0

        '''
        Declaration of other vars
        '''
        # Average accuracy on seen classes
        aver_acc_over_time = dict()
        aver_acc_per_class_over_time = dict()
        conf_mat_over_time = dict()

        # Network mask
        mask_output_val = np.zeros([NUM_CLASSES], dtype=bool)
        mask_output_test = np.zeros([NUM_CLASSES], dtype=bool)

        '''
        Exemplars(for ablation study and other purposes)
        '''
        exemplars_dir = os.path.join(result_folder, 'exemplars')
        if not os.path.exists(exemplars_dir):
            os.makedirs(exemplars_dir)

    '''
    Train generative model(DC-GAN)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    run_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    run_config.gpu_options.allow_growth = True
    graph_gen = tf.Graph()
    sess_wgan = tf.Session(config=run_config, graph=graph_gen)

    acwgan_obj = WGAN64x64(sess_wgan, graph_gen,
                     dataset_name=FLAGS.dataset + '_' + FLAGS.order_file,
                     mode=FLAGS.mode,
                     batch_size=FLAGS.batch_size,
                     dim=FLAGS.dim,
                     output_dim=FLAGS.output_dim,
                     lambda_param=FLAGS.lambda_param,
                     critic_iters=FLAGS.critic_iters,
                     iters=FLAGS.iters,
                     result_dir=FLAGS.result_dir_cwgan,
                     checkpoint_interval=FLAGS.gan_save_interval,
                     adam_lr=FLAGS.adam_lr,
                     use_decay=FLAGS.use_decay,
                     conditional=FLAGS.conditional,
                     acgan=FLAGS.acgan,
                     acgan_scale=FLAGS.acgan_scale,
                     acgan_scale_g=FLAGS.acgan_scale_g,
                     normalization_g=FLAGS.normalization_g,
                     normalization_d=FLAGS.normalization_d,
                     gen_bs_multiple=FLAGS.gen_bs_multiple,
                     nb_cl=FLAGS.nb_cl,
                     n_gpus=FLAGS.n_gpus)

    exemplars = []

    test_images, test_labels, test_one_hot_labels, raw_images_test = imagenet_64x64.load_test_data()

    if not FLAGS.only_gen_no_cls:
        # train and test data of seen classes
        test_x = np.zeros([0, 64, 64, 3], dtype=np.float32)
        test_y = np.zeros([0], dtype=np.float32)

    '''
    Class Incremental Learning
    '''
    print('Starting from category ' + str(FLAGS.from_class_idx + 1) + ' to ' + str(FLAGS.to_class_idx + 1))
    print('Adding %d categories every time' % FLAGS.nb_cl)
    assert(FLAGS.from_class_idx % FLAGS.nb_cl == 0)
    for category_idx in range(FLAGS.from_class_idx, FLAGS.to_class_idx + 1, FLAGS.nb_cl):

        to_category_idx = category_idx + FLAGS.nb_cl - 1
        if FLAGS.nb_cl == 1:
            print('Adding Category ' + str(category_idx + 1))
        else:
            print('Adding Category %d-%d' % (category_idx + 1, to_category_idx + 1))

        train_x_gan = np.zeros([0, FLAGS.output_dim], dtype=np.uint8)
        train_y_gan = np.zeros([0], dtype=float)
        test_x_gan = np.zeros([0, FLAGS.output_dim], dtype=np.uint8)
        test_y_gan = np.zeros([0], dtype=float)

        if not FLAGS.only_gen_no_cls:
            # train and test data of seen classes
            train_y_one_hot = np.zeros([0, NUM_CLASSES], dtype=np.float32)

        for category_idx_in_group in range(category_idx, to_category_idx + 1):
            real_category_idx = order[category_idx_in_group]
            real_images_train_cur_cls, raw_images_train_cur_cls = imagenet_64x64.load_train_data(real_category_idx, flip=FLAGS.flip)

            # GAN
            train_x_gan = np.concatenate((train_x_gan, raw_images_train_cur_cls))
            train_y_gan_cur_cls = np.ones([len(raw_images_train_cur_cls)]) * (category_idx_in_group % FLAGS.nb_cl)
            train_y_gan = np.concatenate((train_y_gan, train_y_gan_cur_cls))

            test_indices_cur_cls = [idx for idx in range(len(test_labels)) if
                                    test_labels[idx] == real_category_idx]
            test_x_gan_cur_cls = raw_images_test[test_indices_cur_cls, :]
            test_y_gan_cur_cls = np.ones([len(test_indices_cur_cls)]) * (category_idx_in_group % FLAGS.nb_cl)
            test_x_gan = np.concatenate((test_x_gan, test_x_gan_cur_cls))
            test_y_gan = np.concatenate((test_y_gan, test_y_gan_cur_cls))

            # Classification network
            if not FLAGS.only_gen_no_cls:
                train_y_one_hot_cur_cls = np.zeros([len(raw_images_train_cur_cls), NUM_CLASSES])
                train_y_one_hot_cur_cls[:, category_idx_in_group] = np.ones(len(raw_images_train_cur_cls))

                test_indices_cur_cls = [idx for idx in range(len(test_labels)) if
                                        test_labels[idx] == real_category_idx]
                test_x_cur_cls = test_images[test_indices_cur_cls, :]
                test_y_cur_cls = np.ones([len(test_indices_cur_cls)]) * category_idx_in_group

                test_x = np.concatenate((test_x, test_x_cur_cls))
                test_y = np.concatenate((test_y, test_y_cur_cls))
                train_y_one_hot = np.concatenate((train_y_one_hot, train_y_one_hot_cur_cls))


        '''
        Train classification model
        '''
        # No need to train the classifier if there is only one class
        if (to_category_idx > 0 and not FLAGS.only_gen_no_cls) or not FLAGS.use_softmax:

            # init certain layers
            sess.run(init_ops)

            if FLAGS.no_truncate:
                mask_output_val[:] = True
            else:
                mask_output_val[:to_category_idx + 1] = True

            # Test on all seen classes
            mask_output_test[:to_category_idx + 1] = True

            '''
            Generate samples of old classes
            '''
            train_x = np.copy(train_x_gan)
            if FLAGS.no_truncate:
                train_y_truncated = train_y_one_hot[:, :]
            else:
                train_y_truncated = train_y_one_hot[:, :to_category_idx + 1]
            train_weights_val = np.ones(len(train_x))

            for old_category_idx in range(0, category_idx):
                if old_category_idx % FLAGS.nb_cl == 0:
                    # Load old class model
                    if not acwgan_obj.load((old_category_idx / FLAGS.nb_cl + 1) * FLAGS.nb_cl - 1)[0]:
                        raise Exception("[!] Train a model first, then run test mode")

                num_gen_samples_x_needed = NUM_TRAIN_SAMPLES_PER_CLASS - len(exemplars[old_category_idx])
                if num_gen_samples_x_needed > 0:
                    if FLAGS.gen_more_and_select:
                        gen_samples_x_more, _, _ = acwgan_obj.test(FLAGS.gen_how_many, old_category_idx % FLAGS.nb_cl)
                        gen_samples_x_more_real = imagenet_64x64.convert_images(gen_samples_x_more)
                        gen_samples_prob = sess.run(test_masked_prob, feed_dict={batch_images: gen_samples_x_more_real,
                                                                                 mask_output: mask_output_val})
                        gen_samples_scores_cur_cls = gen_samples_prob[:, old_category_idx]
                        top_k_indices = np.argsort(-gen_samples_scores_cur_cls)[:num_gen_samples_x_needed]
                        gen_samples_x = gen_samples_x_more[top_k_indices]
                    else:
                        gen_samples_x, _, _ = acwgan_obj.test(num_gen_samples_x_needed, old_category_idx % FLAGS.nb_cl)
                    # import wgan.tflib.save_images
                    # wgan.tflib.save_images.save_images(gen_samples_x[:128].reshape((128, 3, 64, 64)),
                    #                                    'test.jpg')
                    train_x = np.concatenate((train_x, gen_samples_x, exemplars[old_category_idx]))
                    train_weights_val = np.concatenate((train_weights_val,
                                                        np.ones(len(gen_samples_x)) * FLAGS.gen_weight,
                                                        np.ones(len(exemplars[old_category_idx])) * FLAGS.proto_weight))

                    gen_samples_y = np.ones((len(gen_samples_x), to_category_idx + 1)) * (
                            (1 - FLAGS.label_smoothing) / to_category_idx)
                    gen_samples_y[:, old_category_idx] = np.ones((len(gen_samples_x))) * FLAGS.label_smoothing

                    exemplars_y = np.zeros((len(exemplars[old_category_idx]), to_category_idx + 1))
                    exemplars_y[:, old_category_idx] = np.ones((len(exemplars[old_category_idx])))

                    train_y_truncated = np.concatenate((train_y_truncated, gen_samples_y, exemplars_y))

                elif num_gen_samples_x_needed == 0:
                    train_x = np.concatenate((train_x, exemplars[old_category_idx]))
                    train_weights_val = np.concatenate((train_weights_val,
                                                        np.ones(len(exemplars[old_category_idx])) * FLAGS.proto_weight))

                    exemplars_y = np.zeros((len(exemplars[old_category_idx]), to_category_idx + 1))
                    exemplars_y[:, old_category_idx] = np.ones((len(exemplars[old_category_idx])))

                    train_y_truncated = np.concatenate((train_y_truncated, exemplars_y))


            # # DEBUG:
            # train_indices = [idx for idx in range(NUM_SAMPLES_TOTAL) if train_labels[idx] <= category_idx]
            # train_x = raw_images_train[train_indices, :]
            # # Record the response of the new data using the old model(category_idx is consistent with the number of True in mask_output_val_prev)
            # train_y_truncated = train_one_hot_labels[train_indices, :category_idx + 1]

            # Training set
            # Convert the raw images from the data-files to floating-points.
            train_x = imagenet_64x64.convert_images(train_x)

            # Shuffle the indices and create mini-batch
            batch_indices_perm = []

            epoch_idx = 0
            lr = FLAGS.base_lr

            '''
            Training with mixed data
            '''
            while True:
                # Generate mini-batch
                if len(batch_indices_perm) == 0:
                    if epoch_idx >= FLAGS.epochs_per_category:
                        break
                    if epoch_idx in lr_strat:
                        lr /= FLAGS.lr_factor
                        print("NEW LEARNING RATE: %f" % lr)
                    epoch_idx = epoch_idx + 1

                    shuffled_indices = range(train_x.shape[0])
                    np.random.shuffle(shuffled_indices)
                    for i in range(0, len(shuffled_indices), FLAGS.train_batch_size):
                        batch_indices_perm.append(shuffled_indices[i:i + FLAGS.train_batch_size])
                    batch_indices_perm.reverse()

                popped_batch_idx = batch_indices_perm.pop()

                # Use the random index to select random images and labels.
                train_weights_batch_val = train_weights_val[popped_batch_idx]
                train_x_batch = train_x[popped_batch_idx, :, :, :]
                train_y_batch = [train_y_truncated[k] for k in popped_batch_idx]

                # Train
                train_loss_summary_str, train_acc_summary_str, train_accuracy_val, \
                train_loss_val, train_empirical_loss_val, train_reg_loss_val, _ = sess.run(
                    [train_loss_summary, train_acc_summary, train_accuracy, loss, empirical_loss,
                     regularization_loss, opt], feed_dict={batch_images: train_x_batch,
                                                           one_hot_labels_truncated: train_y_batch,
                                                           mask_output: mask_output_val,
                                                           learning_rate: lr,
                                                           train_batch_weights: train_weights_batch_val})

                # Test
                if iteration % FLAGS.test_interval == 0:
                    sess.run(copy_ops)

                    # Divide and conquer: to avoid allocating too much GPU memory
                    test_pred_val = []
                    for i in range(0, len(test_x), FLAGS.test_batch_size):
                        test_x_batch = test_x[i:i + FLAGS.test_batch_size]
                        test_pred_val_batch = sess.run(test_pred, feed_dict={batch_images: test_x_batch,
                                                                             mask_output: mask_output_test})
                        test_pred_val.extend(test_pred_val_batch)

                    test_accuracy_val = 1. * np.sum(np.equal(test_pred_val, test_y)) / (len(test_pred_val))
                    test_per_class_accuracy_val = np.diag(confusion_matrix(test_y, test_pred_val)) * 2
                    # I simply multiply the correct predictions by 2 to calculate the accuracy since there are 50 samples per class in the test set

                    test_acc_summary_str = sess.run(test_acc_summary, feed_dict={test_accuracy: test_accuracy_val})

                    test_summary_writer.add_summary(test_acc_summary_str, iteration)

                    print("TEST: step %d, lr %.4f, accuracy %g" % (iteration, lr, test_accuracy_val))
                    print("PER CLASS ACCURACY: " + " | ".join(str(o) + '%' for o in test_per_class_accuracy_val))

                # Print the training logs
                if iteration % FLAGS.display_interval == 0:
                    train_summary_writer.add_summary(train_loss_summary_str, iteration)
                    train_summary_writer.add_summary(train_acc_summary_str, iteration)
                    print("TRAIN: epoch %d, step %d, lr %.4f, accuracy %g, loss %g, empirical %g, reg %g" % (
                        epoch_idx, iteration, lr, train_accuracy_val, train_loss_val,
                        train_empirical_loss_val, train_reg_loss_val))

                iteration = iteration + 1

            '''
            Final test(before the next class is added)
            '''
            sess.run(copy_ops)
            # Divide and conquer: to avoid allocating too much GPU memory
            test_pred_val = []
            for i in range(0, len(test_x), FLAGS.test_batch_size):
                test_x_batch = test_x[i:i + FLAGS.test_batch_size]
                test_pred_val_batch = sess.run(test_pred, feed_dict={batch_images: test_x_batch,
                                                                     mask_output: mask_output_test})
                test_pred_val.extend(test_pred_val_batch)

            test_accuracy_val = 1. * np.sum(np.equal(test_pred_val, test_y)) / (len(test_pred_val))
            conf_mat = confusion_matrix(test_y, test_pred_val)
            test_per_class_accuracy_val = np.diag(conf_mat)

            # Record and save the cumulative accuracy
            aver_acc_over_time[to_category_idx] = test_accuracy_val
            aver_acc_per_class_over_time[to_category_idx] = test_per_class_accuracy_val
            conf_mat_over_time[to_category_idx] = conf_mat

            dump_obj = dict()
            dump_obj['flags'] = flags.FLAGS.__flags
            dump_obj['aver_acc_over_time'] = aver_acc_over_time
            dump_obj['aver_acc_per_class_over_time'] = aver_acc_per_class_over_time
            dump_obj['conf_mat_over_time'] = conf_mat_over_time

            np_file_result = os.path.join(result_folder, 'acc_over_time.pkl')
            with open(np_file_result, 'wb') as file:
                pickle.dump(dump_obj, file)

            visualize_result.vis(np_file_result, 'ImageNetDogs')

        # reorder the exemplars
        if FLAGS.reorder_exemplars:
            for old_category_idx in range(category_idx):

                sess.run(copy_ops)
                # Divide and conquer: to avoid allocating too much GPU memory
                train_prob_cur_cls_exemplars_val = sess.run(test_masked_prob,
                                                            feed_dict={batch_images: imagenet_64x64.convert_images(
                                                                       exemplars[old_category_idx]),
                                                                       mask_output: mask_output_val})
                train_prob_cur_cls_exemplars_val = train_prob_cur_cls_exemplars_val[:, old_category_idx]
                reorder_indices = np.argsort(-train_prob_cur_cls_exemplars_val)
                exemplars[old_category_idx] = exemplars[old_category_idx][reorder_indices]

        # select the exemplars
        for category_idx_in_group in range(category_idx, to_category_idx + 1):
            train_indices_cur_cls = [idx for idx in range(len(train_y_gan)) if train_y_gan[idx] == category_idx_in_group % FLAGS.nb_cl]
            train_x_cur_cls = train_x_gan[train_indices_cur_cls]
            train_x_cur_cls_normalized = imagenet_64x64.convert_images(train_x_cur_cls)
            sess.run(copy_ops)
            # Divide and conquer: to avoid allocating too much GPU memory
            train_prob_cur_cls_val = sess.run(test_masked_prob, feed_dict={batch_images: train_x_cur_cls_normalized,
                                                                           mask_output: mask_output_val})
            train_prob_cur_cls_val = train_prob_cur_cls_val[:, category_idx_in_group]

            # use iCaRL-like memory mechanism to save exemplars or not
            if FLAGS.memory_constrained:

                if FLAGS.auto_choose_num_exemplars:  # auto or fixed number of exemplars
                    # check if we can save all new samples as exemplars
                    if NUM_TRAIN_SAMPLES_PER_CLASS > FLAGS.memory_upperbound - sum([len(exemplars[i]) for i in range(len(exemplars))]):
                        # load inception scores of all classes
                        save_exemplars_ratios = []
                        for i in range(category_idx_in_group + 1):
                            inception_score = acwgan_obj.load_inception_score(i)
                            save_exemplars_ratio = FLAGS.auto_param1 - FLAGS.auto_param2 * inception_score
                            save_exemplars_ratios.append(save_exemplars_ratio)

                        save_exemplars_ratios = np.array(save_exemplars_ratios)
                        keep_exemplars_num = np.floor(save_exemplars_ratios * FLAGS.memory_upperbound
                                                         / sum(save_exemplars_ratios)).astype(int)
                        for old_category_idx in range(category_idx_in_group):
                            exemplars[old_category_idx] = exemplars[old_category_idx][:keep_exemplars_num[old_category_idx]]
                        num_exemplars_cur_cls = keep_exemplars_num[-1]
                    else:
                        num_exemplars_cur_cls = NUM_TRAIN_SAMPLES_PER_CLASS

                else:
                    num_exemplars_per_cls = int(FLAGS.memory_upperbound // (category_idx_in_group + 1))
                    num_exemplars_per_cls = min(num_exemplars_per_cls, NUM_TRAIN_SAMPLES_PER_CLASS)
                    # remove redundant elements in the memory for previous classes
                    if category_idx_in_group > 0 and len(exemplars[0]) > num_exemplars_per_cls:
                        for old_category_idx in range(category_idx_in_group):
                            exemplars[old_category_idx] = exemplars[old_category_idx][:num_exemplars_per_cls]


                    # add how many new elements in the memory for the current class
                    num_exemplars_cur_cls = num_exemplars_per_cls
                    print(' [*] Store %d exemplars for each class' % num_exemplars_cur_cls)

            else:
                if FLAGS.auto_choose_num_exemplars: # auto or fixed number of exemplars
                    inception_score = acwgan_obj.load_inception_score(category_idx_in_group)
                    num_exemplars_cur_cls = int(np.floor(FLAGS.auto_param1 - FLAGS.auto_param2 * inception_score))
                    print(' [*] Inception score %f, store %d exemplars' % (inception_score, num_exemplars_cur_cls))
                else:
                    num_exemplars_cur_cls = FLAGS.num_exemplars_per_class

            selected_indices = np.array(range(len(train_prob_cur_cls_val)))
            if FLAGS.exemplar_select_criterion == 'high':
                selected_indices = train_prob_cur_cls_val.argsort()[:-(num_exemplars_cur_cls+1):-1]    # select the last 20
            elif FLAGS.exemplar_select_criterion == 'low':
                selected_indices = train_prob_cur_cls_val.argsort()[:num_exemplars_cur_cls]    # select the last 20
            elif FLAGS.exemplar_select_criterion == 'random':
                random_idx = range(len(train_prob_cur_cls_val))
                np.random.shuffle(random_idx)
                selected_indices = random_idx[:num_exemplars_cur_cls]

            exemplars.append(train_x_cur_cls[selected_indices])

            np_file_exemplars = os.path.join(exemplars_dir, 'exemplars_%d' % (category_idx_in_group+1))
            np.save(np_file_exemplars, exemplars)

        '''
        Train generative model(W-GAN)
        '''
        if acwgan_obj.check_model(to_category_idx):
            print(" [*] Model of Class %d-%d exists. Skip the training process" % (
                category_idx + 1, to_category_idx + 1))
        else:
            print(" [*] Model of Class %d-%d does not exist. Start the training process" % (
                category_idx + 1, to_category_idx + 1))
            acwgan_obj.train(train_x_gan, train_y_gan, test_x_gan, test_y_gan, to_category_idx)

    # Save the final model
    if not FLAGS.only_gen_no_cls:
        checkpoint_dir = os.path.join(result_folder, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'))
        sess.close()


if __name__ == '__main__':
    tf.app.run()
