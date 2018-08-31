import os, sys
import time
import functools

import numpy as np
import tensorflow as tf

import wgan.tflib as lib
import wgan.tflib.save_images
import wgan.tflib.inception_score
import wgan.tflib.plot
import wgan.tflib.ops
import wgan.tflib.ops.linear
import wgan.tflib.ops.conv2d
import wgan.tflib.ops.deconv2d

import pickle

def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def conv_cond_concat(x, y):
   x_shapes = x.get_shape()
   y_shapes = y.get_shape()
   return tf.concat([x, y*tf.ones([x_shapes[0], y_shapes[1], x_shapes[2], x_shapes[3]])], 1)


class ACGAN(object):

    def __init__(self, sess, graph, dataset_name, mode, batch_size, dim, output_dim,
                 lambda_param, critic_iters, iters, result_dir, checkpoint_interval,
                 adam_lr, adam_beta1, adam_beta2, finetune, finetune_from,
                 pretrained_model_base_dir, pretrained_model_sub_dir, nb_cl, pre_iters):

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

        self.nb_cl = nb_cl

        self.pre_iters = pre_iters

        self.build_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def build_model(self):
        with tf.variable_scope("gan") as scope, self.graph.as_default():
            # placeholder for CIFAR samples
            self.real_data_int = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])

            self.real_y = tf.placeholder(tf.float32, shape=[None, self.nb_cl])
            self.gen_y = tf.placeholder(tf.float32, shape=[None, self.nb_cl])
            self.sample_y = tf.placeholder(tf.float32, shape=[None, self.nb_cl])

            real_data = 2 * ((tf.cast(self.real_data_int, tf.float32) / 255.) - .5)
            fake_data, fake_labels = self.generator(self.batch_size, self.gen_y)

            # set_shape to facilitate concatenation of label and image
            fake_data.set_shape([self.batch_size, fake_data.shape[1].value])

            disc_real, disc_real_class = self.discriminator(real_data, reuse=False)
            disc_fake, disc_fake_class = self.discriminator(fake_data, reuse=True)

            self.real_accu = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(disc_real_class, 1), tf.argmax(self.real_y, 1)), dtype=tf.float32))
            self.fake_accu = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(disc_fake_class, 1), tf.argmax(fake_labels, 1)), dtype=tf.float32))

            self.real_class_cost = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_class, labels=self.real_y))
            self.gen_class_cost = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_class, labels=fake_labels))
            self.class_cost = self.real_class_cost + self.gen_class_cost

            self.summary_fake_data = tf.summary.image('fake_data',
                                                      tf.transpose(tf.reshape(fake_data, [-1, 3, 32, 32]),
                                                                   [0, 2, 3, 1]))  # transpose: NCWH -> NWHC
            self.summary_disc_real = tf.summary.histogram('disc_real', disc_real)
            self.summary_disc_fake = tf.summary.histogram('disc_fake', disc_fake)

            gen_params = [var for var in tf.trainable_variables() if 'Generator' in var.name]
            disc_params = [var for var in tf.trainable_variables() if 'Discriminator' in var.name and 'ClassOutput' not in var.name]
            class_params = [var for var in tf.trainable_variables()]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.mode == 'wgan-gp':
                    # Standard WGAN loss
                    self.gen_cost = -tf.reduce_mean(disc_fake)
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

            with tf.control_dependencies(update_ops):
                if self.mode == 'wgan-gp':
                    self.gen_train_op = \
                        tf.train.AdamOptimizer(learning_rate=self.adam_lr,
                                               beta1=self.adam_beta1, beta2=self.adam_beta2)\
                            .minimize(self.gen_cost, var_list=gen_params)
                    self.disc_train_op = \
                        tf.train.AdamOptimizer(learning_rate=self.adam_lr * 5,
                                               beta1=self.adam_beta1, beta2=self.adam_beta2)\
                            .minimize(self.disc_cost, var_list=disc_params)
                    self.class_train_op = \
                        tf.train.AdamOptimizer(learning_rate=self.adam_lr * 5,
                                               beta1=self.adam_beta1, beta2=self.adam_beta2) \
                            .minimize(self.class_cost, var_list=class_params)


            self.summary_gen_cost = tf.summary.scalar('gen_cost', self.gen_cost)
            self.summary_disc_cost = tf.summary.scalar('disc_cost', self.disc_cost)

            # NOTICE: I think it is correct now
            self.summaries_gen = tf.summary.merge([self.summary_gen_cost, self.summary_disc_fake,
                                                   self.summary_fake_data])
            self.summaries_disc = tf.summary.merge([self.summary_disc_cost, self.summary_disc_fake,
                                                    self.summary_disc_real])

            # For generating samples
            fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
            self.fixed_noise_samples_128 = self.sampler(128, self.sample_y, noise=fixed_noise_128)[0]

            # For calculating inception score
            self.samples_100 = self.sampler(100, self.sample_y)[0]

            var_list = tf.trainable_variables()

            bn_moving_vars = [var for var in tf.global_variables() if 'moving_mean' in var.name]
            bn_moving_vars += [var for var in tf.global_variables() if 'moving_variance' in var.name]
            var_list += bn_moving_vars

            self.saver = tf.train.Saver(var_list=var_list)  # var_list doesn't contain Adam params

    def gen_labels(self, nb_samples, condition=None):
        labels = np.zeros([nb_samples, self.nb_cl], dtype=np.float32)
        for i in range(nb_samples):
            if condition is not None:   # random or not
                label_item = condition
            else:
                label_item = np.random.randint(0, self.nb_cl)
            labels[i, label_item] = 1   # one hot label
        return labels

    def generate_image(self, frame, train_log_dir_for_cur_class):
        # different y, same x
        for category_idx in range(self.nb_cl):

            y = self.gen_labels(128, category_idx)
            samples = self.sess.run(self.fixed_noise_samples_128, feed_dict={self.sample_y: y})
            samples = ((samples + 1.) * (255. / 2)).astype('int32')
            samples_folder = os.path.join(train_log_dir_for_cur_class, 'samples', 'class_%d' % (category_idx + 1))
            if not os.path.exists(samples_folder):
                os.makedirs(samples_folder)
            wgan.tflib.save_images.save_images(samples.reshape((128, 3, 32, 32)),
                                               os.path.join(samples_folder,
                                                            'samples_{}.jpg'.format(frame)))

    def get_inception_score(self):
        all_samples = []
        for i in range(10):
            for category_idx in range(self.nb_cl):
                y = self.gen_labels(100, category_idx)
                all_samples.append(self.sess.run(self.samples_100, feed_dict={self.sample_y: y}))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
        return wgan.tflib.inception_score.get_inception_score(list(all_samples))

    def test(self, n_samples, mean=0, stddev=1):
        assert n_samples > 0
        with self.graph.as_default():
            noise = tf.random_normal([n_samples, 128], mean, stddev)
            output = self.sampler(n_samples, self.sample_y, noise=noise)[0]
            samples, z = self.sess.run([output, noise])

        samples_int = ((samples + 1.) * (255. / 2)).astype('int32')
        return samples_int, samples, z

    def generator(self, n_samples, y, noise=None):

        with tf.variable_scope('Generator') as scope:

            if noise is None:
                noise = tf.random_normal([n_samples, 128])

            if self.nb_cl > 1:
                z = tf.concat([noise, y], axis=1)
            else:
                z = noise

            output = wgan.tflib.ops.linear.Linear('g_Input', z.shape[1].value, 4 * 4 * 4 * self.dim, z)
            output = tf.layers.batch_normalization(output, training=True, axis=1, name='g_bn1')
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4 * self.dim, 4, 4])

            output = wgan.tflib.ops.deconv2d.Deconv2D('g_2', 4 * self.dim, 2 * self.dim, 5, output)
            output = tf.layers.batch_normalization(output, training=True, axis=1, name='g_bn2')
            output = tf.nn.relu(output)

            output = wgan.tflib.ops.deconv2d.Deconv2D('g_3', 2 * self.dim, self.dim, 5, output)
            output = tf.layers.batch_normalization(output, training=True, axis=1, name='g_bn3')
            output = tf.nn.relu(output)

            output = wgan.tflib.ops.deconv2d.Deconv2D('g_5', self.dim, 3, 5, output)

            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim]), y

    def sampler(self, n_samples, y, noise=None):

        with tf.variable_scope('Generator') as scope:
            scope.reuse_variables()

            if noise is None:
                noise = tf.random_normal([n_samples, 128])

            if self.nb_cl > 1:
                z = tf.concat([noise, y], axis=1)
            else:
                z = noise

            output = wgan.tflib.ops.linear.Linear('g_Input', z.shape[1].value, 4 * 4 * 4 * self.dim, z)
            output = tf.layers.batch_normalization(output, training=False, axis=1, name='g_bn1')
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, 4 * self.dim, 4, 4])

            output = wgan.tflib.ops.deconv2d.Deconv2D('g_2', 4 * self.dim, 2 * self.dim, 5, output)
            output = tf.layers.batch_normalization(output, training=False, axis=1, name='g_bn2')
            output = tf.nn.relu(output)

            output = wgan.tflib.ops.deconv2d.Deconv2D('g_3', 2 * self.dim, self.dim, 5, output)
            output = tf.layers.batch_normalization(output, training=False, axis=1, name='g_bn3')
            output = tf.nn.relu(output)

            output = wgan.tflib.ops.deconv2d.Deconv2D('g_5', self.dim, 3, 5, output)

            output = tf.tanh(output)

            return tf.reshape(output, [-1, self.output_dim]), y

    def discriminator(self, inputs, reuse):

        with tf.variable_scope('Discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            output = tf.reshape(inputs, [-1, 3, 32, 32])

            output = wgan.tflib.ops.conv2d.Conv2D('d_1', 3, self.dim, 5, output, stride=2)
            output = leaky_relu(output)

            output = wgan.tflib.ops.conv2d.Conv2D('d_2', self.dim, 2 * self.dim, 5, output, stride=2)
            if self.mode != 'wgan-gp':
                output = tf.layers.batch_normalization(output, training=True, axis=1, name='d_bn2')
            output = leaky_relu(output)

            output = wgan.tflib.ops.conv2d.Conv2D('d_3', 2 * self.dim, 4 * self.dim, 5, output, stride=2)
            if self.mode != 'wgan-gp':
                output = tf.layers.batch_normalization(output, training=True, axis=1, name='d_bn3')
            output = leaky_relu(output)

            output = tf.reshape(output, [-1, 4*4*4*self.dim])
            sourceOutput = wgan.tflib.ops.linear.Linear('d_SourceOutput', 4 * 4 * 4 * self.dim, 1, output)
            classOutput = wgan.tflib.ops.linear.Linear('d_ClassOutput', 4 * 4 * 4 * self.dim, self.nb_cl, output)

            return tf.reshape(sourceOutput, shape=[-1]), tf.reshape(classOutput, shape=[-1, self.nb_cl])

    def train(self, data_X, data_y, test_X, test_y, category_idx):

        train_log_dir_for_cur_class = self.model_dir_for_class(category_idx)

        if not os.path.exists(train_log_dir_for_cur_class):
            os.makedirs(train_log_dir_for_cur_class)

        '''
        def get_test_epoch(test_X, test_y):
            test_X_y = list(zip(test_X, test_y))
            np.random.shuffle(test_X_y)
            test_X[:], test_y[:] = zip(*test_X_y)

            for j in range(len(test_X) / self.batch_size):
                yield (test_X[j * self.batch_size:(j + 1) * self.batch_size],
                       test_y[j * self.batch_size:(j + 1) * self.batch_size])

        def get_train_inf(data_X, data_y):
            while True:
                batch_idxs = len(data_X) // self.batch_size
                data_X_y = list(zip(data_X, data_y))
                np.random.shuffle(data_X_y)
                data_X[:], data_y[:] = zip(*data_X_y)
                for idx in range(0, batch_idxs):
                    _data_X = data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                    _data_y = data_y[idx * self.batch_size:(idx + 1) * self.batch_size]
                    yield (_data_X, _data_y)
        '''
        
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
            # Train loop
            self.sess.run(tf.initialize_all_variables())    # saver's var_list contains 28 variables, tf.all_variables() contains more(plus Adam params)
        if self.finetune:
            self.load_pretrained(step=self.finetune_from)

        gen = get_train_inf(data_X, data_y)

        # reset the cache of the plot
        wgan.tflib.plot.reset()

        # summary
        summary_dir = os.path.join(train_log_dir_for_cur_class, 'summary')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        summary_writer = tf.summary.FileWriter(summary_dir, self.graph)

        # for iterp in range(self.pre_iters):
        #
        #     start_time = time.time()
        #     _data_X, _data_y = gen.next()
        #     _, accuracy = self.sess.run([self.disc_train_op, self.real_accu],
        #                                 feed_dict={self.real_data_int: _data_X, self.real_y: _data_y,
        #                                            self.gen_y: self.gen_labels(self.batch_size)})
        #     if iterp % 100 == 99:
        #         print('iter {}: pretraining accuracy: {}\ttime: {}'.format(iterp, accuracy,
        #                                                                    time.time() - start_time))

        for iteration in range(self.iters):
            start_time = time.time()
            # Train generator
            if iteration > 0:
                _, summaries_gen_str = self.sess.run([self.gen_train_op, self.summaries_gen],
                                                     feed_dict={self.gen_y: self.gen_labels(self.batch_size)})
                summary_writer.add_summary(summaries_gen_str, iteration)

            # Train critic
            if self.mode == 'dcgan':
                disc_iters = 1
            else:
                disc_iters = self.critic_iters

            for i in range(disc_iters):
                _data_X, _data_y = gen.next()
                _disc_cost, _gen_cost, _fake_accu, _real_accu, _gen_class_cost, _real_class_cost, _, _, summaries_disc_str = \
                    self.sess.run([self.disc_cost, self.gen_cost, self.fake_accu, self.real_accu, self.gen_class_cost,
                                   self.real_class_cost, self.disc_train_op, self.class_train_op, self.summaries_disc],
                                  feed_dict={self.real_data_int: _data_X,
                                             self.real_y: _data_y,
                                             self.gen_y: self.gen_labels(self.batch_size)})
                summary_writer.add_summary(summaries_disc_str, iteration)
                if self.mode == 'wgan':
                    _ = self.sess.run(self.clip_disc_weights)

            lib.plot.plot('time', time.time() - start_time)
            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('train gen cost', _gen_cost)

            lib.plot.plot('gen class cost', _gen_class_cost)
            lib.plot.plot('real class cost', _real_class_cost)
            lib.plot.plot('gen accuracy', _fake_accu)
            lib.plot.plot('real accuracy', _real_accu)

            print("iter {}: disc cost: {}\ttime: {}".format(iteration + 1, _disc_cost, time.time() - start_time))

            # Calculate inception score every 1K iters
            # NOTICE!!!
            # if iteration % 1000 == 999:
            #     inception_score = self.get_inception_score()
            #     tflib.plot.plot('inception score', inception_score[0])

            # Calculate dev loss and generate samples every 100 iters
            if (iteration+1) % 100 == 0:
                dev_disc_costs = []
                for images, labels in get_test_epoch(test_X, test_y):
                    _dev_disc_cost = self.sess.run(self.disc_cost, feed_dict={self.real_data_int: images,
                                                                              self.real_y: labels,
                                                                              self.gen_y: self.gen_labels(self.batch_size)})
                    dev_disc_costs.append(_dev_disc_cost)
                wgan.tflib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                self.generate_image(iteration+1, train_log_dir_for_cur_class)

            # Save checkpoint
            if (iteration+1) % self.save_interval == 0:
                inception_score = self.get_inception_score()
                wgan.tflib.plot.plot('inception score', inception_score[0])
                self.save(iteration+1, category_idx)

            # Save logs every 100 iters
            if (iteration+1) % 100 == 0:
                wgan.tflib.plot.flush(train_log_dir_for_cur_class)

            wgan.tflib.plot.tick()


        # final save checkpoint
        self.save(iteration+1, category_idx, final=True)


    @property
    def model_dir(self):
        finetune_str = {True: ('finetune_' + self.pretrained_model_sub_dir.replace('/', '_')), False: 'from_scratch'}
        return os.path.join(self.result_dir, self.dataset_name, 'nb_cl_%d' % self.nb_cl, str(self.adam_lr), str(self.iters),
                            str(self.pre_iters), finetune_str[self.finetune])

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
