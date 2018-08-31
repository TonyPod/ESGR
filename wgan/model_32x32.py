import os, sys
import time
import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.save_images
import tflib.inception_score
import tflib.plot
import tflib.ops
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.deconv2d

import pickle

def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


class GAN(object):

    def __init__(self, sess, graph, dataset_name, mode, batch_size, dim, output_dim,
                 lambda_param, critic_iters, iters, result_dir, checkpoint_interval,
                 adam_lr, adam_beta1, adam_beta2, finetune, finetune_from,
                 pretrained_model_base_dir, pretrained_model_sub_dir):

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
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

        self.finetune = finetune
        self.finetune_from = finetune_from
        self.pretrained_model_base_dir = pretrained_model_base_dir
        self.pretrained_model_sub_dir = pretrained_model_sub_dir
        self.pretrained_model_dir = os.path.join(self.pretrained_model_base_dir, self.pretrained_model_sub_dir)

        self.build_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def build_model(self):
        with tf.variable_scope("gan") as scope, self.graph.as_default():
            # placeholder for CIFAR samples
            self.real_data_int = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])

            real_data = 2 * ((tf.cast(self.real_data_int, tf.float32) / 255.) - .5)
            fake_data = self.generator(self.batch_size)

            disc_real = self.discriminator(real_data, reuse=False)
            disc_fake = self.discriminator(fake_data, reuse=True)

            self.summary_fake_data = tf.summary.image('fake_data',
                                                      tf.transpose(tf.reshape(fake_data, [-1, 3, 32, 32]),
                                                                   [0, 2, 3, 1]))  # transpose: NCWH -> NWHC
            self.summary_disc_real = tf.summary.histogram('disc_real', disc_real)
            self.summary_disc_fake = tf.summary.histogram('disc_fake', disc_fake)

            gen_params = [var for var in tf.global_variables() if 'Generator' in var.name]
            disc_params = [var for var in tf.global_variables() if 'Discriminator' in var.name]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.mode == 'wgan':
                    gen_cost = -tf.reduce_mean(disc_fake)
                    self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                    self.gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
                    self.disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(self.disc_cost, var_list=disc_params)

                    clip_ops = []
                    for var in disc_params:
                        clip_bounds = [-.01, .01]
                        clip_ops.append(
                            tf.assign(
                                var,
                                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                            )
                        )

                    self.clip_disc_weights = tf.group(*clip_ops)

                elif self.mode == 'wgan-gp':
                    # Standard WGAN loss
                    gen_cost = -tf.reduce_mean(disc_fake)
                    self.disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                    # Gradient penalty
                    alpha = tf.random_uniform(
                        shape=[self.batch_size, 1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data
                    interpolates = real_data + (alpha * differences)
                    gradients = tf.gradients(self.discriminator(interpolates, reuse=True), [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                    self.disc_cost += self.lambda_param * gradient_penalty

                    # self.gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                    #                                                                                          var_list=gen_params)
                    # self.disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(self.disc_cost,
                    #                                                                                           var_list=disc_params)
                    self.gen_train_op = \
                        tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                               beta1=self.adam_beta1, beta2=self.adam_beta2)\
                            .minimize(gen_cost, var_list=gen_params)
                    self.disc_train_op = \
                        tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                               beta1=self.adam_beta1, beta2=self.adam_beta2)\
                            .minimize(self.disc_cost, var_list=disc_params)

                elif self.mode == 'dcgan':
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                    self.disc_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                    self.disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
                    self.disc_cost /= 2.

                    self.gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                                  var_list=gen_params)
                    self.disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.disc_cost,
                                                                                                   var_list=disc_params)

            self.summary_gen_cost = tf.summary.scalar('gen_cost', gen_cost)
            self.summary_disc_cost = tf.summary.scalar('disc_cost', self.disc_cost)

            # NOTICE: I think it is correct now
            self.summaries_gen = tf.summary.merge([self.summary_gen_cost, self.summary_disc_fake,
                                                   self.summary_fake_data])
            self.summaries_disc = tf.summary.merge([self.summary_disc_cost, self.summary_disc_fake,
                                                    self.summary_disc_real])

            # For generating samples
            fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
            self.fixed_noise_samples_128 = self.sampler(128, noise=fixed_noise_128)

            # For calculating inception score
            self.samples_100 = self.sampler(100)

            var_list = tf.trainable_variables()

            bn_moving_vars = [var for var in tf.global_variables() if 'moving_mean' in var.name]
            bn_moving_vars += [var for var in tf.global_variables() if 'moving_variance' in var.name]
            var_list += bn_moving_vars

            self.saver = tf.train.Saver(var_list=var_list)  # var_list doesn't contain Adam params

    def generate_image(self, frame, train_log_dir_for_cur_class):
        samples = self.sess.run(self.fixed_noise_samples_128)
        samples = ((samples + 1.) * (255. / 2)).astype('int32')
        samples_folder = os.path.join(train_log_dir_for_cur_class, 'samples')
        if not os.path.exists(samples_folder):
            os.makedirs(samples_folder)
        tflib.save_images.save_images(samples.reshape((128, 3, 32, 32)),
                                           os.path.join(samples_folder,
                                                        'samples_{}.jpg'.format(frame)))

    def get_inception_score(self):
        all_samples = []
        for i in range(10):
            all_samples.append(self.sess.run(self.samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
        return tflib.inception_score.get_inception_score(list(all_samples))

    def test(self, n_samples, mean=0, stddev=1):
        assert n_samples > 0
        with self.graph.as_default():
            noise = tf.random_normal([n_samples, 128], mean, stddev)
            output = self.sampler(n_samples, noise=noise)
            samples, z = self.sess.run([output, noise])

        samples_int = ((samples + 1.) * (255. / 2)).astype('int32')
        return samples_int, samples, z

    def generator(self, n_samples, noise=None):

        with tf.variable_scope('Generator') as scope:

            if noise is None:
                noise = tf.random_normal([n_samples, 128])

            output = tflib.ops.linear.Linear('g_Input', 128, 4 * 4 * 4 * self.dim, noise)
            output = tf.layers.batch_normalization(output, training=True, axis=1, name='g_bn1')
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4 * self.dim, 4, 4])

            output = tflib.ops.deconv2d.Deconv2D('g_2', 4 * self.dim, 2 * self.dim, 5, output)
            output = tf.layers.batch_normalization(output, training=True, axis=1, name='g_bn2')
            output = tf.nn.relu(output)

            output = tflib.ops.deconv2d.Deconv2D('g_3', 2 * self.dim, self.dim, 5, output)
            output = tf.layers.batch_normalization(output, training=True, axis=1, name='g_bn3')
            output = tf.nn.relu(output)

            output = tflib.ops.deconv2d.Deconv2D('g_5', self.dim, 3, 5, output)

            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim])

    def sampler(self, n_samples, noise=None):

        with tf.variable_scope('Generator') as scope:
            scope.reuse_variables()

            if noise is None:
                noise = tf.random_normal([n_samples, 128])

            output = tflib.ops.linear.Linear('g_Input', 128, 4 * 4 * 4 * self.dim, noise)
            output = tf.layers.batch_normalization(output, training=False, axis=1, name='g_bn1')
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4 * self.dim, 4, 4])

            output = tflib.ops.deconv2d.Deconv2D('g_2', 4 * self.dim, 2 * self.dim, 5, output)
            output = tf.layers.batch_normalization(output, training=False, axis=1, name='g_bn2')
            output = tf.nn.relu(output)

            output = tflib.ops.deconv2d.Deconv2D('g_3', 2 * self.dim, self.dim, 5, output)
            output = tf.layers.batch_normalization(output, training=False, axis=1, name='g_bn3')
            output = tf.nn.relu(output)

            output = tflib.ops.deconv2d.Deconv2D('g_5', self.dim, 3, 5, output)

            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim])

    def discriminator(self, inputs, reuse):

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 3, 32, 32])

            output = tflib.ops.conv2d.Conv2D('d_1', 3, self.dim, 5, output, stride=2)
            output = leaky_relu(output)

            output = tflib.ops.conv2d.Conv2D('d_2', self.dim, 2 * self.dim, 5, output, stride=2)
            if self.mode != 'wgan-gp':
                output = tf.layers.batch_normalization(output, training=True, axis=1, name='d_bn2')
            output = leaky_relu(output)

            output = tflib.ops.conv2d.Conv2D('d_3', 2 * self.dim, 4 * self.dim, 5, output, stride=2)
            if self.mode != 'wgan-gp':
                output = tf.layers.batch_normalization(output, training=True, axis=1, name='d_bn3')
            output = leaky_relu(output)

            output = tf.reshape(output, [-1, 4*4*4*self.dim])
            output = tflib.ops.linear.Linear('d_Output', 4 * 4 * 4 * self.dim, 1, output)

            return tf.reshape(output, [-1])

    def train(self, data_X, test_X, category_idx=None):

        train_log_dir_for_cur_class = self.model_dir_for_class(category_idx)

        if not os.path.exists(train_log_dir_for_cur_class):
            os.makedirs(train_log_dir_for_cur_class)

        def get_test_epoch():
            np.random.shuffle(test_X)

            for j in range(len(test_X) / self.batch_size):
                yield (test_X[j * self.batch_size:(j + 1) * self.batch_size])

        def get_train_inf():
            while True:
                batch_idxs = len(data_X) // self.batch_size
                np.random.shuffle(data_X)
                for idx in range(0, batch_idxs):
                    _data = data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                    yield _data

        with self.graph.as_default():
            # Train loop
            self.sess.run(tf.initialize_all_variables())    # saver's var_list contains 28 variables, tf.all_variables() contains more(plus Adam params)
        if self.finetune:
            self.load_pretrained(step=self.finetune_from)

        gen = get_train_inf()

        # reset the cache of the plot
        tflib.plot.reset()

        # summary
        summary_dir = os.path.join(train_log_dir_for_cur_class, 'summary')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        summary_writer = tf.summary.FileWriter(summary_dir, self.graph)

        for iteration in range(self.iters):
            start_time = time.time()
            # Train generator
            if iteration > 0:
                _, summaries_gen_str = self.sess.run([self.gen_train_op, self.summaries_gen])
                summary_writer.add_summary(summaries_gen_str, iteration)
            # Train critic
            if self.mode == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = self.critic_iters
            for i in range(disc_iters):
                _data = gen.next()
                _disc_cost, _, summaries_disc_str = \
                    self.sess.run([self.disc_cost, self.disc_train_op, self.summaries_disc],
                                  feed_dict={self.real_data_int: _data})
                summary_writer.add_summary(summaries_disc_str, iteration)
                if self.mode == 'wgan':
                    _ = self.sess.run(self.clip_disc_weights)

            tflib.plot.plot('train disc cost', _disc_cost)
            tflib.plot.plot('time', time.time() - start_time)
            print("iter {}: disc cost: {}\ttime: {}".format(iteration+1, _disc_cost, time.time() - start_time))

            # Calculate inception score every 1K iters
            # NOTICE!!!
            # if iteration % 1000 == 999:
            #     inception_score = self.get_inception_score()
            #     tflib.plot.plot('inception score', inception_score[0])

            # Calculate dev loss and generate samples every 100 iters
            if (iteration+1) % 100 == 0:
                dev_disc_costs = []
                for images in get_test_epoch():
                    _dev_disc_cost = self.sess.run(self.disc_cost, feed_dict={self.real_data_int: images})
                    dev_disc_costs.append(_dev_disc_cost)
                tflib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                self.generate_image(iteration+1, train_log_dir_for_cur_class)

            # Save checkpoint
            if (iteration+1) % self.save_interval == 0:
                inception_score = self.get_inception_score()
                tflib.plot.plot('inception score', inception_score[0])
                self.save(iteration+1, category_idx)

            # Save logs every 100 iters
            if (iteration+1) % 100 == 0:
                tflib.plot.flush(train_log_dir_for_cur_class)

            tflib.plot.tick()


        # final save checkpoint
        self.save(iteration+1, category_idx, final=True)


    @property
    def model_dir(self):
        finetune_str = {True: ('finetune_' + self.pretrained_model_sub_dir.replace('/', '_')), False: 'from_scratch'}
        return os.path.join(self.result_dir, self.dataset_name, str(self.adam_lr), str(self.iters),
                            finetune_str[self.finetune])

    def model_dir_for_class(self, category_idx):
        if category_idx is None:
            return os.path.join(self.model_dir, 'all_classes')
        else:
            return os.path.join(self.model_dir, 'class_%d' % (category_idx + 1))

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

    def load_pretrained(self, step=-1):
        import re
        print(" [*] Reading checkpoints...")
        if step == -1:
            checkpoint_dir = os.path.join(self.pretrained_model_dir, "checkpoints", "final")
        else:
            checkpoint_dir = os.path.join(self.pretrained_model_dir, "checkpoints", str(step))

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
