import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from ViTCA.model import ViTCA
from Utils.trainer_utils import compute_metrics_relative, Trainer, merge_metrics, tensor_to_images, send_to_telegram


class TrainerViTCA(Trainer):
    def __init__(self, params):
        super().__init__(params)
        self.model_params = self.params['model_params']
        self.input_shape = self.model_params['input_shape']
        self.batch_size = self.input_shape[0]
        self.output_overflow_loss = self.params['output_overflow_loss']
        self.hidden_overflow_loss = self.params['hidden_overflow_loss']
        self.reconstruction_loss_factor = self.params['reconstruction_loss_factor']
        self.overflow_loss_factor = self.params['overflow_loss_factor']

        self.pool_length = self.params['pool_length']
        self.pool = []
        self.update_probability = self.params['update_probability']
        self.min_cell_updates = self.params['min_cell_updates']
        self.max_cell_updates = self.params['max_cell_updates']

        self.model = ViTCA(self.model_params)
        self.best_model = ViTCA(self.model_params)
        self.model([self.model.seed(tf.random.normal(self.input_shape, dtype=tf.float32)),
                    tf.constant(2 * self.max_cell_updates, dtype=tf.float32, shape=(1,)),
                    tf.constant(self.update_probability, dtype=tf.float32, shape=(1,))])
        self.best_model([self.best_model.seed(tf.random.normal(self.input_shape, dtype=tf.float32)),
                         tf.constant(2 * self.max_cell_updates, dtype=tf.float32, shape=(1,)),
                         tf.constant(self.update_probability, dtype=tf.float32, shape=(1,))])
        self.best_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(self.params['learning_rate'],
                                                                    self.epochs * len(self.train_gen)))
        self.train_gen.switch_mode(train_ca=True)
        self.val_gen.switch_mode(train_ca=True)
        self.test_gen.switch_mode(train_ca=True)

    def display_performance_image(self, path):
        self.test_gen.on_epoch_end()
        anchor, positive = self.test_gen.__getitem__(0)
        anchor, positive = anchor[: self.test_length], positive[: self.test_length]
        positive_pred = self.best_model.get_rgb_out(self.best_model([self.best_model.seed(positive),
                                                                     tf.constant(2 * self.max_cell_updates,
                                                                                 dtype=tf.float32, shape=(1,)),
                                                                     tf.constant(self.update_probability,
                                                                                 dtype=tf.float32, shape=(1,))],
                                                                    training=False))
        images = (tensor_to_images(anchor), tensor_to_images(positive_pred), tensor_to_images(positive))
        rows, columns = self.test_length, 3
        f, ax = plt.subplots(rows, columns, figsize=(self.test_length * 32, self.test_length * 16))
        for i in range(rows):
            for pos, block in enumerate(images):
                ax[i, pos].imshow(block[i])
        if path is not None:
            plt.savefig(os.path.join(path, f'epoch_{self.curr_epoch}_ca.png'))
        plt.close()

    def pool_sample(self):
        pool_y, pool_x = zip(*self.pool[:self.input_shape[0]])
        y_true = tf.concat([tf.expand_dims(t, axis=0) for t in pool_y], axis=0)
        x = tf.concat([tf.expand_dims(t, axis=0) for t in pool_x], axis=0)
        return x, y_true

    def update_pool(self, sampled, y_true, y_pred):
        if sampled:
            self.pool[:self.batch_size] = zip(y_true, y_pred)
        else:
            self.pool += zip(y_true, y_pred)
        np.random.shuffle(self.pool)
        self.pool = self.pool[:self.pool_length]

    @tf.function
    def compute_losses(self, y_true, y_pred):
        losses_dict = {}
        rgb_out = self.model.get_rgb_out(y_pred)
        losses_dict['rec_loss'] = tf.reduce_mean(tf.keras.losses.mae(y_true, rgb_out))
        total_loss = self.reconstruction_loss_factor * losses_dict['rec_loss']
        if self.output_overflow_loss:
            losses_dict['out_loss'] = tf.reduce_mean(tf.abs(rgb_out - tf.clip_by_value(rgb_out, 0., 1.)))
            total_loss += self.overflow_loss_factor * losses_dict['out_loss']
        if self.hidden_overflow_loss:
            hidden = self.model.get_hidden(y_pred)
            losses_dict['hid_loss'] = tf.reduce_mean(tf.abs(hidden - tf.clip_by_value(hidden, -1., 1.)))
            total_loss += self.overflow_loss_factor * losses_dict['hid_loss']
        return losses_dict, total_loss

    @tf.function
    def train_on_batch(self, inputs):
        y_true, x = inputs
        cell_updates = tf.cast(tf.random.uniform((1,), self.min_cell_updates, self.max_cell_updates + 1,
                                                 dtype=tf.int32), dtype=tf.float32)
        with tf.GradientTape() as tape:
            y_pred = self.model([x, cell_updates, tf.constant(self.update_probability, dtype=tf.float32, shape=(1,))],
                                training=True)
            losses, loss = self.compute_losses(y_true, y_pred)
        gradients = [g / (tf.norm(g) + 1e-8) if g is not None else g for g in
                     tape.gradient(loss, self.model.trainable_weights)]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        metrics = compute_metrics_relative(y_true, self.model.get_rgb_out(y_pred), self.model.get_rgb_in(x))
        return losses, metrics, y_pred

    @tf.function
    def validate_on_batch(self, inputs):
        y_true, x = inputs
        y_pred = self.model([self.model.seed(x), tf.constant(2 * self.max_cell_updates, dtype=tf.float32, shape=(1,)),
                             tf.constant(self.update_probability, dtype=tf.float32, shape=(1,))], training=False)
        losses = self.compute_losses(y_true, y_pred)[0]
        metrics = compute_metrics_relative(y_true, self.model.get_rgb_out(y_pred), x)
        return losses, metrics

    def final_test(self):
        self.test_gen.on_epoch_end()
        val_losses = []
        val_metrics = []
        for value in self.test_gen:
            y_true, x = value
            y_pred = self.best_model([self.best_model.seed(x),
                                      tf.constant(2 * self.max_cell_updates, dtype=tf.float32, shape=(1,)),
                                      tf.constant(self.update_probability, dtype=tf.float32, shape=(1,))],
                                     training=False)
            val_losses.append(self.compute_losses(y_true, y_pred)[0])
            val_metrics.append(compute_metrics_relative(y_true, self.model.get_rgb_out(y_pred), x))
        losses = merge_metrics(val_losses)
        metrics = merge_metrics(val_metrics)
        new_line = pd.DataFrame([{**i, **j} for i, j in zip(losses, metrics)])
        with open(self.logs_directory_test, "wb") as log_file:
            pickle.dump(new_line, log_file)
        if self.token is not None:
            send_to_telegram('Final Test\nModel   {}\n{}'.format(self.model_name, new_line.mean().to_string()),
                             self.token, self.chat_id)
        print(new_line.mean())

    def train(self, resume=False):
        if resume:
            self.resume()
        for ep in range(self.curr_epoch, self.epochs):
            self.curr_epoch += 1
            for pos, value in enumerate(self.train_gen):
                if len(self.pool) > self.batch_size:
                    x, y_true = self.pool_sample()
                    _, _, y_pred = self.train_on_batch((y_true, x))
                    self.update_pool(True, y_true, y_pred)
                y_true, x = value
                x = self.model.seed(x)
                losses, metrics, y_pred = self.train_on_batch((y_true, x))
                self.update_pool(False, y_true, y_pred)
                self.display_status(ep, pos, losses, metrics, False)
            val_losses = []
            val_metrics = []
            for value in self.val_gen:
                y_true, x = value
                losses, metrics = self.validate_on_batch((y_true, x))
                val_losses.append(losses)
                val_metrics.append(metrics)
            self.display_status(ep, ep, val_losses, val_metrics, validation=True)
            self.save_model(val_losses, val_metrics)
            if ep % self.display_frequency == 0:
                self.display_performance_image(path=self.images_dir)
        self.display_performance_image(path=self.images_dir)
        self.final_test()
        self.clean_folder()
        return self.best_model
