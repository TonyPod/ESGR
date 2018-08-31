import os, sys
import time
import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.cond_batchnorm
import tflib.inception_score
import tflib.save_images
import tflib.ops.layernorm
import tflib.plot

import pickle

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


class WGAN64x64(object):
    def __init__(self, sess, graph, dataset_name, mode, batch_size, dim, output_dim,
                 lambda_param, critic_iters, iters, result_dir, checkpoint_interval,
                 adam_lr, use_decay, conditional, acgan, acgan_scale, acgan_scale_g,
                 normalization_g, normalization_d, gen_bs_multiple, nb_cl, n_gpus):

        self.sess = sess
        self.graph = graph
        self.dataset_name = dataset_name
        self.mode = mode
        self.batch_size = batch_size
        self.dim = dim
        self.output_dim = output_dim
        self.lambda_param = lambda_param

        self.critic_iters = critic_iters
        self.iters = iters
        self.result_dir = result_dir
        self.save_interval = checkpoint_interval

        self.adam_lr = adam_lr

        # acgan
        self.use_decay = use_decay
        self.conditional = conditional
        self.acgan = acgan
        self.acgan_scale = acgan_scale
        self.acgan_scale_g = acgan_scale_g

        self.normalization_g = normalization_g
        self.normalization_d = normalization_d
        self.gen_bs_multiple = gen_bs_multiple

        self.nb_cl = nb_cl

        self.n_gpus = n_gpus
        self.DEVICES = ['/gpu:{}'.format(i) for i in range(n_gpus)]
        if len(self.DEVICES) == 1:  # Hack because the code assumes 2 GPUs
            self.DEVICES = [self.DEVICES[0], self.DEVICES[0]]

        self.build_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def build_model(self):

        with tf.variable_scope("wgan") as scope, self.graph.as_default():
            self._iteration = tf.placeholder(tf.int32, shape=None)
            self.all_real_data_int = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])
            self.all_real_labels = tf.placeholder(tf.int32, shape=[self.batch_size])

            labels_splits = tf.split(self.all_real_labels, len(self.DEVICES), axis=0)

            fake_data_splits = []
            for i, device in enumerate(self.DEVICES):
                with tf.device(device):
                    fake_data_splits.append(self.generator(self.batch_size / len(self.DEVICES), labels_splits[i]))

            all_real_data = tf.reshape(2 * ((tf.cast(self.all_real_data_int, tf.float32) / 256.) - .5),
                                       [self.batch_size, self.output_dim])
            all_real_data += tf.random_uniform(shape=[self.batch_size, self.output_dim], minval=0., maxval=1. / 128)  # dequantize
            all_real_data_splits = tf.split(all_real_data, len(self.DEVICES), axis=0)

            self.DEVICES_B = self.DEVICES[:len(self.DEVICES) / 2]
            self.DEVICES_A = self.DEVICES[len(self.DEVICES) / 2:]

            disc_costs = []
            disc_acgan_costs = []
            disc_acgan_accs = []
            disc_acgan_fake_accs = []
            for i, device in enumerate(self.DEVICES_A):
                with tf.device(device):
                    real_and_fake_data = tf.concat([
                        all_real_data_splits[i],
                        all_real_data_splits[len(self.DEVICES_A) + i],
                        fake_data_splits[i],
                        fake_data_splits[len(self.DEVICES_A) + i]
                    ], axis=0)
                    real_and_fake_labels = tf.concat([
                        labels_splits[i],
                        labels_splits[len(self.DEVICES_A) + i],
                        labels_splits[i],
                        labels_splits[len(self.DEVICES_A) + i]
                    ], axis=0)
                    disc_all, disc_all_acgan = self.discriminator(real_and_fake_data, real_and_fake_labels, reuse=False)
                    disc_real = disc_all[:self.batch_size / len(self.DEVICES_A)]
                    disc_fake = disc_all[self.batch_size / len(self.DEVICES_A):]
                    disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
                    if self.conditional and self.acgan:
                        disc_acgan_costs.append(tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(
                                logits=disc_all_acgan[:self.batch_size / len(self.DEVICES_A)],
                                labels=real_and_fake_labels[:self.batch_size / len(self.DEVICES_A)])
                        ))
                        disc_acgan_accs.append(tf.reduce_mean(
                            tf.cast(
                                tf.equal(
                                    tf.to_int32(tf.argmax(disc_all_acgan[:self.batch_size / len(self.DEVICES_A)], dimension=1)),
                                    real_and_fake_labels[:self.batch_size / len(self.DEVICES_A)]
                                ),
                                tf.float32
                            )
                        ))
                        disc_acgan_fake_accs.append(tf.reduce_mean(
                            tf.cast(
                                tf.equal(
                                    tf.to_int32(tf.argmax(disc_all_acgan[self.batch_size / len(self.DEVICES_A):], dimension=1)),
                                    real_and_fake_labels[self.batch_size / len(self.DEVICES_A):]
                                ),
                                tf.float32
                            )
                        ))

            for i, device in enumerate(self.DEVICES_B):
                with tf.device(device):
                    real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(self.DEVICES_A) + i]], axis=0)
                    fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(self.DEVICES_A) + i]], axis=0)
                    labels = tf.concat([
                        labels_splits[i],
                        labels_splits[len(self.DEVICES_A) + i],
                    ], axis=0)
                    alpha = tf.random_uniform(
                        shape=[self.batch_size / len(self.DEVICES_A), 1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data
                    interpolates = real_data + (alpha * differences)
                    gradients = tf.gradients(self.discriminator(interpolates, labels, reuse=True)[0], [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
                    disc_costs.append(gradient_penalty)

            self.disc_wgan = tf.add_n(disc_costs) / len(self.DEVICES_A)
            if self.conditional and self.acgan:
                self.disc_acgan = tf.add_n(disc_acgan_costs) / len(self.DEVICES_A)
                self.disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(self.DEVICES_A)
                self.disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(self.DEVICES_A)
                self.disc_cost = self.disc_wgan + (self.acgan_scale * self.disc_acgan)
            else:
                self.disc_acgan = tf.constant(0.)
                self.disc_acgan_acc = tf.constant(0.)
                self.disc_acgan_fake_acc = tf.constant(0.)
                self.disc_cost = self.disc_wgan

            disc_params = lib.params_with_name('Discriminator.')

            if self.use_decay:
                decay = tf.maximum(0., 1. - (tf.cast(self._iteration, tf.float32) / self.iters))
            else:
                decay = 1.

            gen_costs = []
            gen_acgan_costs = []
            for device in self.DEVICES:
                with tf.device(device):
                    n_samples = self.gen_bs_multiple * self.batch_size / len(self.DEVICES)
                    fake_labels = tf.cast(tf.random_uniform([n_samples]) * self.nb_cl, tf.int32)
                    if self.conditional and self.acgan:
                        disc_fake, disc_fake_acgan = self.discriminator(self.generator(n_samples, fake_labels), fake_labels, reuse=True)
                        gen_costs.append(-tf.reduce_mean(disc_fake))
                        gen_acgan_costs.append(tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                        ))
                    else:
                        gen_costs.append(
                            -tf.reduce_mean(self.discriminator(self.generator(n_samples, fake_labels), fake_labels, reuse=True)[0]))
            self.gen_cost = (tf.add_n(gen_costs) / len(self.DEVICES))
            if self.conditional and self.acgan:
                self.gen_cost += (self.acgan_scale_g * (tf.add_n(gen_acgan_costs) / len(self.DEVICES)))

            gen_opt = tf.train.AdamOptimizer(learning_rate=self.adam_lr * decay, beta1=0., beta2=0.9)
            disc_opt = tf.train.AdamOptimizer(learning_rate=self.adam_lr * decay, beta1=0., beta2=0.9)
            gen_gv = gen_opt.compute_gradients(self.gen_cost, var_list=lib.params_with_name('Generator'))
            disc_gv = disc_opt.compute_gradients(self.disc_cost, var_list=disc_params)
            self.gen_train_op = gen_opt.apply_gradients(gen_gv)
            self.disc_train_op = disc_opt.apply_gradients(disc_gv)

            # Function for generating samples
            if self.nb_cl <= 10:
                nb_images = self.nb_cl * self.nb_cl
                self.fixed_labels_all = tf.constant(np.array(range(self.nb_cl) * self.nb_cl, dtype='int32'))
            else:
                nb_images = self.nb_cl * 1
                self.fixed_labels_all = tf.constant(np.array(range(self.nb_cl), dtype='int32'))
            fixed_noise = tf.constant(np.random.normal(size=(nb_images, 128)).astype('float32'))
            self.fixed_labels = tf.placeholder(tf.int32, shape=[nb_images])
            self.fixed_noise_samples = self.generator(nb_images, self.fixed_labels, noise=fixed_noise)

            self.fixed_noise_samples_all = self.generator(nb_images, self.fixed_labels_all, noise=fixed_noise)

            # self.fixed_labels_1 = tf.constant(np.array([1] * 100, dtype='int32'))
            # self.fixed_noise_samples_1 = self.generator(100, self.fixed_labels_1, noise=fixed_noise)
            #
            # self.fixed_labels_2 = tf.placeholder(tf.int32, shape=[100])
            # self.fixed_noise_samples_2 = self.generator(100, self.fixed_labels_2, noise=fixed_noise)

            # Function for calculating inception score
            fake_labels_100 = tf.cast(tf.random_uniform([100]) * self.nb_cl, tf.int32)
            self.samples_100 = self.generator(100, fake_labels_100)

            var_list = tf.trainable_variables()

            bn_moving_vars = [var for var in tf.global_variables() if 'moving_mean' in var.name]
            bn_moving_vars += [var for var in tf.global_variables() if 'moving_variance' in var.name]
            var_list += bn_moving_vars

            self.saver = tf.train.Saver(var_list=var_list)  # var_list doesn't contain Adam params

    def Normalize(self, name, inputs, labels=None):
        if not self.conditional:
            labels = None
        if self.conditional and self.acgan and ('Discriminator' in name):
            labels = None

        if ('Discriminator' in name) and self.normalization_d:
            return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs, labels=labels, n_labels=self.nb_cl)
        elif ('Generator' in name) and self.normalization_g:
            if labels is not None:
                return lib.ops.cond_batchnorm.Batchnorm(name, [0, 2, 3], inputs, labels=labels, n_labels=self.nb_cl)
            else:
                return lib.ops.batchnorm.Batchnorm(name, [0, 2, 3], inputs, fused=True)
        else:
            return inputs

    def ResidualBlock(self, name, input_dim, output_dim, filter_size, inputs, resample=None, labels=None):
        """
        resample: None, 'down', or 'up'
        """
        if resample == 'down':
            conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
            conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = ConvMeanPool
        elif resample == 'up':
            conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = UpsampleConv
            conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
        elif resample == None:
            conv_shortcut = lib.ops.conv2d.Conv2D
            conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
            conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
        else:
            raise Exception('invalid resample value')

        if output_dim == input_dim and resample == None:
            shortcut = inputs  # Identity skip-connection
        else:
            shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                     he_init=False, biases=True, inputs=inputs)

        output = inputs
        output = self.Normalize(name + '.N1', output, labels=labels)
        output = tf.nn.relu(output)
        output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output)
        output = self.Normalize(name + '.N2', output, labels=labels)
        output = tf.nn.relu(output)
        output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output)

        return shortcut + output

    def gen_labels(self, nb_samples, condition=None):
        labels = np.zeros([nb_samples], dtype=np.int32)
        for i in range(nb_samples):
            if condition is not None:   # random or not
                label_item = condition
            else:
                label_item = np.random.randint(0, self.nb_cl)
            labels[i] = label_item
        return labels

    def generate_image(self, frame, train_log_dir_for_cur_class):

        samples = self.sess.run(self.fixed_noise_samples_all)
        samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
        samples_folder = os.path.join(train_log_dir_for_cur_class, 'samples', 'all')
        if not os.path.exists(samples_folder):
            os.makedirs(samples_folder)
        lib.save_images.save_images(samples.reshape((-1, 3, 64, 64)),
                                    os.path.join(samples_folder,
                                                 'samples_{}.jpg'.format(frame)))

        # different y, same x
        for category_idx in range(self.nb_cl):

            y = np.ones(self.nb_cl * self.nb_cl) * category_idx
            samples = self.sess.run(self.fixed_noise_samples, feed_dict={self.fixed_labels: y})
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            samples_folder = os.path.join(train_log_dir_for_cur_class, 'samples', 'class_%d' % (category_idx + 1))
            if not os.path.exists(samples_folder):
                os.makedirs(samples_folder)
            lib.save_images.save_images(samples.reshape((-1, 3, 64, 64)),
                                        os.path.join(samples_folder,
                                                     'samples_{}.jpg'.format(frame)))

    def get_inception_score(self, n):
        all_samples = []
        for i in range(n/100):
            all_samples.append(self.sess.run(self.samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 64, 64)).transpose((0,2,3,1))
        return lib.inception_score.get_inception_score(list(all_samples))

    def test(self, n_samples, label, mean=0, stddev=1):
        assert n_samples > 0
        with self.graph.as_default():
            noise = tf.random_normal([n_samples, 128], mean, stddev)
            if label is not None:
                labels = np.ones(n_samples, dtype=np.int32) * label
            else:
                labels = None

            output = self.sampler(n_samples, labels, noise=noise)
            samples, z = self.sess.run([output, noise])

        samples_int = ((samples + 1.) * (255. / 2)).astype('int32')
        return samples_int, samples, z

    def generator(self, n_samples, labels, noise=None):

        with tf.variable_scope('Generator') as scope:

            if noise is None:
                noise = tf.random_normal([n_samples, 128])

            output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*self.dim, noise)
            output = tf.reshape(output, [-1, 8*self.dim, 4, 4])

            output = self.ResidualBlock('Generator.Res1', 8*self.dim, 8*self.dim, 3, output, resample='up', labels=labels)
            output = self.ResidualBlock('Generator.Res2', 8*self.dim, 4*self.dim, 3, output, resample='up', labels=labels)
            output = self.ResidualBlock('Generator.Res3', 4*self.dim, 2*self.dim, 3, output, resample='up', labels=labels)
            output = self.ResidualBlock('Generator.Res4', 2*self.dim, 1*self.dim, 3, output, resample='up', labels=labels)

            output = self.Normalize('Generator.OutputN', output)
            output = tf.nn.relu(output)
            output = lib.ops.conv2d.Conv2D('Generator.Output', 1*self.dim, 3, 3, output)
            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim])

    def sampler(self, n_samples, labels, noise=None):

        with tf.variable_scope('Generator') as scope:
            scope.reuse_variables()

            if noise is None:
                noise = tf.random_normal([n_samples, 128])

            output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * 8 * self.dim, noise)
            output = tf.reshape(output, [-1, 8 * self.dim, 4, 4])

            output = self.ResidualBlock('Generator.Res1', 8 * self.dim, 8 * self.dim, 3, output, resample='up', labels=labels)
            output = self.ResidualBlock('Generator.Res2', 8 * self.dim, 4 * self.dim, 3, output, resample='up', labels=labels)
            output = self.ResidualBlock('Generator.Res3', 4 * self.dim, 2 * self.dim, 3, output, resample='up', labels=labels)
            output = self.ResidualBlock('Generator.Res4', 2 * self.dim, 1 * self.dim, 3, output, resample='up', labels=labels)

            output = self.Normalize('Generator.OutputN', output)
            output = tf.nn.relu(output)
            output = lib.ops.conv2d.Conv2D('Generator.Output', 1 * self.dim, 3, 3, output)
            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim])

    def discriminator(self, inputs, labels, reuse):

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 3, 64, 64])
            output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, self.dim, 3, output, he_init=False)

            output = self.ResidualBlock('Discriminator.Res1', self.dim, 2*self.dim, 3, output, resample='down', labels=labels)
            output = self.ResidualBlock('Discriminator.Res2', 2*self.dim, 4*self.dim, 3, output, resample='down', labels=labels)
            output = self.ResidualBlock('Discriminator.Res3', 4*self.dim, 8*self.dim, 3, output, resample='down', labels=labels)
            output = self.ResidualBlock('Discriminator.Res4', 8*self.dim, 8*self.dim, 3, output, resample='down', labels=labels)

            output = tf.reshape(output, [-1, 4*4*8*self.dim])
            output_wgan = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*self.dim, 1, output)
            output_wgan = tf.reshape(output_wgan, [-1])

            if self.conditional and self.acgan:
                output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', 4*4*8*self.dim, self.nb_cl, output)
                return output_wgan, output_acgan
            else:
                return output_wgan, None

    def train(self, data_X, data_y, test_X, test_y, category_idx):

        train_log_dir_for_cur_class = self.model_dir_for_class(category_idx)

        if not os.path.exists(train_log_dir_for_cur_class):
            os.makedirs(train_log_dir_for_cur_class)

        def get_test_epoch(test_X, test_y):
            reorder = np.array(range(len(test_X)))
            np.random.shuffle(reorder)
            test_X = test_X[reorder]
            test_y = test_y[reorder]

            for j in range(len(test_X) / self.batch_size):
                yield (test_X[j * self.batch_size:(j + 1) * self.batch_size],
                       test_y[j * self.batch_size:(j + 1) * self.batch_size])

        def get_train_inf(data_X, data_y):
            while True:
                batch_idxs = len(data_X) // self.batch_size
                reorder = np.array(range(len(data_X)))
                np.random.shuffle(reorder)
                data_X = data_X[reorder]
                data_y = data_y[reorder]
                for idx in range(0, batch_idxs):
                    _data_X = data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                    _data_y = data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                    yield (_data_X, _data_y)


        with self.graph.as_default():
            self.sess.run(tf.initialize_all_variables())

        # Train loop
        gen = get_train_inf(data_X, data_y)

        # reset the cache of the plot
        lib.plot.reset()

        for iteration in range(self.iters):
            start_time = time.time()

            if iteration > 0:
                _ = self.sess.run([self.gen_train_op], feed_dict={self._iteration: iteration})

            for i in range(self.critic_iters):
                _data, _labels = gen.next()
                if self.conditional and self.acgan:
                    _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = self.sess.run(
                        [self.disc_cost, self.disc_wgan, self.disc_acgan, self.disc_acgan_acc, self.disc_acgan_fake_acc, self.disc_train_op],
                        feed_dict={self.all_real_data_int: _data, self.all_real_labels: _labels, self._iteration: iteration})
                else:
                    _disc_cost, _ = self.sess.run([self.disc_cost, self.disc_train_op],
                                                feed_dict={self.all_real_data_int: _data, self.all_real_labels: _labels,
                                                           self._iteration: iteration})

            lib.plot.plot('cost', _disc_cost)
            if self.conditional and self.acgan:
                lib.plot.plot('wgan', _disc_wgan)
                lib.plot.plot('acgan', _disc_acgan)
                lib.plot.plot('acc_real', _disc_acgan_acc)
                lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
            lib.plot.plot('time', time.time() - start_time)
            print("iter {}: disc cost: {}\ttime: {}".format(iteration+1, _disc_cost, time.time() - start_time))

            # Save checkpoint
            if (iteration + 1) % self.save_interval == 0:
                inception_score = self.get_inception_score(50000)
                lib.plot.plot('inception_50k', inception_score[0])
                lib.plot.plot('inception_50k_std', inception_score[1])
                self.save(iteration + 1, category_idx)

            # Calculate dev loss and generate samples every 100 iters
            if (iteration+1) % 100 == 0:
                dev_disc_costs = []
                for images, _labels in get_test_epoch(test_X, test_y):
                    _dev_disc_cost = self.sess.run([self.disc_cost], feed_dict={self.all_real_data_int: images,
                                                                         self.all_real_labels: _labels})
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

                self.generate_image(iteration, train_log_dir_for_cur_class)
                lib.plot.flush(train_log_dir_for_cur_class)

            lib.plot.tick()

        # Save checkpoint
        self.save(iteration + 1, category_idx, final=True)

    @property
    def model_dir(self):
        finetune_str = 'from_scratch'
        return os.path.join(self.result_dir, self.dataset_name, 'nb_cl_%d' % self.nb_cl, str(self.adam_lr),
                            str(self.iters), finetune_str)

    def model_dir_for_class(self, category_idx):
        if category_idx is None:
            return os.path.join(self.model_dir, 'all_classes')
        else:
            return os.path.join(self.model_dir, 'class_%d-%d' % (category_idx + 2 - self.nb_cl, category_idx + 1))

    def save(self, step, category_idx, final=False):
        model_name = self.mode + ".model"
        if final:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", str(step))

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # To make sure that the checkpoints of old classes are no longer recorded
        self.saver.set_last_checkpoints_with_time([])
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load_inception_score(self, category_idx, step=-1):
        log_file = os.path.join(self.model_dir_for_class(category_idx), "log", "log.pkl")
        with open(log_file, 'rb') as file:
            log_data = pickle.load(file)

        if step == -1:
            inception_score_max_idx = max(log_data["inception score"].keys())
            inception_score = log_data["inception score"][inception_score_max_idx]
        else:
            inception_score = log_data["inception score"][step-1]

        return inception_score

    def load(self, category_idx, step=-1):
        import re
        print(" [*] Reading checkpoints...")
        if step == -1:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", str(step))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def check_model(self, category_idx, step=-1):
        """
        Check whether the old models(which<category_idx) exist
        :param category_idx:
        :return: True or false
        """
        print(" [*] Checking checkpoints for class %d" % (category_idx + 1))
        if step == -1:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(self.model_dir_for_class(category_idx), "checkpoints", str(step))

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            return True
        else:
            return False

