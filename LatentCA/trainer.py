import os
import pickle
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from LatentCA.model import AutoEncoderDown2, AutoEncoderDown3
from Utils.trainer_utils import Trainer, compose_input, tensor_to_images, init_total_variation_loss_filters, \
    ssim_loss, psnr_loss, mse_loss, cosine_loss, compute_metrics_relative, compute_metrics, merge_metrics, \
    send_to_telegram


class LatentTrainerAE(Trainer):
    def __init__(self, params):
        super().__init__(params)
        self.model_params = self.params['model_params']
        self.input_shape = self.model_params['input_shape']
        self.pool_shape = self.model_params['pool_shape']
        self.num_down_sampling = self.params['num_down_sampling']
        if self.num_down_sampling == 2:
            self.model = AutoEncoderDown2(self.model_params)
            self.best_model = AutoEncoderDown2(self.model_params)
        else:
            self.model = AutoEncoderDown3(self.model_params)
            self.best_model = AutoEncoderDown3(self.model_params)
        self.model(
            compose_input(tf.random.normal(self.input_shape, dtype=tf.float32), self.pool_shape, 64, 0.5, None, False,
                          True))
        self.best_model(
            compose_input(tf.random.normal(self.input_shape, dtype=tf.float32), self.pool_shape, 64, 0.5, None, False,
                          True))
        self.best_model.set_weights(self.model.get_weights())
        self.model.change_mod(latent_ca_trainable=False, autoencoder_trainable=True)
        self.best_model.change_mod(latent_ca_trainable=False, autoencoder_trainable=True)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(self.params['learning_rate'],
                                                                    self.epochs * len(self.train_gen)))
        self.train_gen.switch_mode(train_ca=False)
        self.val_gen.switch_mode(train_ca=False)
        self.test_gen.switch_mode(train_ca=False)
        self.zero_factor = tf.constant(0.)
        self.margin = self.params['margin']
        self.reconstruction_loss_ae = self.params['reconstruction_loss_ae']
        self.reconstruction_loss_ae_weight = self.params['reconstruction_loss_weight_ae']
        self.distance_loss = self.params['distance_loss']
        self.distance_loss_weight = self.params['distance_loss_weight']
        self.task_loss = self.params['task_loss']
        self.task_loss_weight = self.params['task_loss_weight']
        self.equivalent_loss = self.params['equivalent_loss']
        self.equivalent_loss_weight = self.params['equivalent_loss_weight']
        self.perturbation_intensity = self.params['perturbation_intensity']
        self.total_variation_loss_ae = self.params['total_variation_loss_ae']
        self.total_variation_loss_ae_weight = self.params['total_variation_loss_weight_ae']
        self.convX = None
        self.convY = None
        if self.total_variation_loss_ae:
            self.convX, self.convY = init_total_variation_loss_filters(input_shape=self.input_shape)
        self.perturbation = self.perturbation_intensity > 0

    def display_performance_image(self, path):
        self.test_gen.on_epoch_end()
        anchor, positive, negative = self.test_gen.__getitem__(0)
        anchor, positive = anchor[: self.test_length], positive[: self.test_length]
        anchor_pred, anchor_embed, _, anchor_x_1 = self.best_model(compose_input(anchor, self.pool_shape), training=False)
        positive_pred, positive_embed, _, positive_x_1 = self.best_model(compose_input(positive, self.pool_shape),
                                                                         training=False)
        anchor_conditioned = self.best_model.pop_out(anchor_embed, positive_x_1, training=False)
        positive_conditioned = self.best_model.pop_out(positive_embed, anchor_x_1, training=False)
        random_embed = anchor_embed + tf.random.normal(shape=positive_embed.shape, mean=0.0, stddev=0.1)
        random_conditioned = self.best_model.pop_out(random_embed, anchor_x_1, training=False)
        images = (tensor_to_images(anchor), tensor_to_images(anchor_pred), tensor_to_images(anchor_conditioned),
                  tensor_to_images(positive), tensor_to_images(positive_pred), tensor_to_images(positive_conditioned),
                  tensor_to_images(random_conditioned))
        rows, columns = self.test_length, 7
        f, ax = plt.subplots(rows, columns, figsize=(self.test_length * 32, self.test_length * 16))
        for i in range(rows):
            for pos, block in enumerate(images):
                ax[i, pos].imshow(block[i])
        if path is not None:
            plt.savefig(os.path.join(path, f'epoch_{self.curr_epoch}.png'))
        plt.close()

    @tf.function
    def compute_losses_part_1(self, input_tensor, anchor, positive, y_pred, positive_y_pred, anchor_embed,
                              positive_embed, negative_embed):
        losses_dict = {}
        total_loss = 0.
        if self.reconstruction_loss_ae is not None:
            if self.reconstruction_loss_ae == 'SSIM':
                rec_loss_total = ssim_loss(input_tensor, y_pred)
            elif self.reconstruction_loss_ae == 'PSRN':
                rec_loss_total = psnr_loss(input_tensor, y_pred)
            else:
                rec_loss_total = mse_loss(input_tensor, y_pred)
            losses_dict['rec_loss'] = rec_loss_total
            total_loss += self.reconstruction_loss_ae_weight * losses_dict['rec_loss']
        if self.distance_loss is not None:
            anchor_embed_flat = tf.reshape(anchor_embed, shape=(anchor_embed.shape[0], np.prod(anchor_embed.shape[1:])))
            positive_embed_flat = tf.reshape(positive_embed,
                                             shape=(positive_embed.shape[0], np.prod(positive_embed.shape[1:])))
            negative_embed_flat = tf.reshape(negative_embed,
                                             shape=(negative_embed.shape[0], np.prod(negative_embed.shape[1:])))
            if self.distance_loss == 'COS':
                dist_loss_total = cosine_loss(anchor_embed_flat, positive_embed_flat) - cosine_loss(anchor_embed_flat,
                                                                                                    negative_embed_flat)
            else:
                dist_loss_total = mse_loss(anchor_embed_flat, positive_embed_flat) - mse_loss(anchor_embed_flat,
                                                                                              negative_embed_flat)
            losses_dict['dist_loss'] = tf.maximum(self.zero_factor, dist_loss_total + self.margin)
            total_loss += self.distance_loss_weight * losses_dict['dist_loss']
        if self.task_loss is not None:
            mask = tf.cast(tf.cast(positive - anchor, dtype=tf.bool), dtype=tf.float32)
            if self.task_loss == 'SSIM':
                task_loss_total = ssim_loss(positive_y_pred * mask, positive * mask)
            elif self.task_loss == 'PSNR':
                task_loss_total = psnr_loss(positive_y_pred * mask, positive * mask)
            else:
                task_loss_total = mse_loss(positive_y_pred * mask, positive * mask)
            losses_dict['task_loss'] = task_loss_total
            total_loss += self.task_loss_weight * losses_dict['task_loss']
        if self.total_variation_loss_ae:
            x_grad = tf.reduce_sum(
                tf.exp(-tf.abs(self.convX(anchor))) * tf.abs(self.convX(y_pred[:self.input_shape[0]])), axis=[1, 2, 3])
            y_grad = tf.reduce_sum(
                tf.exp(-tf.abs(self.convY(anchor))) * tf.abs(self.convY(y_pred[:self.input_shape[0]])), axis=[1, 2, 3])
            losses_dict['tv_loss'] = tf.reduce_mean(x_grad + y_grad)
            total_loss += self.total_variation_loss_ae_weight * losses_dict['tv_loss']
        return losses_dict, total_loss

    @tf.function
    def compute_losses_part_2(self, anchor, positive, anchor_conv1, positive_conv1, conditioned_positive,
                              conditioned_anchor):
        losses_dict = {}
        total_loss = 0.
        mask = tf.cast(tf.cast(positive - anchor, dtype=tf.bool), dtype=tf.float32)
        if self.equivalent_loss == 'PureNoiseSSIM':
            equiv_loss_total = ssim_loss(conditioned_anchor * mask, positive * mask) + ssim_loss(
                conditioned_positive * mask, anchor * mask)
        elif self.equivalent_loss == 'PureNoisePSNR':
            equiv_loss_total = psnr_loss(conditioned_anchor * mask, positive * mask) + psnr_loss(
                conditioned_positive * mask, anchor * mask)
        elif self.equivalent_loss == 'PureNoiseMSE':
            equiv_loss_total = mse_loss(conditioned_anchor * mask, positive * mask) + mse_loss(
                conditioned_positive * mask, anchor * mask)
        elif self.equivalent_loss == 'NormalNoise':
            equiv_loss_total = mse_loss(anchor_conv1, positive_conv1) + mse_loss(conditioned_anchor * mask,
                                                                                 positive * mask) + \
                               mse_loss(conditioned_positive * mask, anchor * mask)
        elif self.equivalent_loss == 'PureSSIM':
            equiv_loss_total = ssim_loss(conditioned_anchor, positive) + ssim_loss(conditioned_positive, anchor)
        elif self.equivalent_loss == 'PurePSNR':
            equiv_loss_total = psnr_loss(conditioned_anchor, positive) + psnr_loss(conditioned_positive, anchor)
        elif self.equivalent_loss == 'PureMSE':
            equiv_loss_total = mse_loss(conditioned_anchor, positive) + mse_loss(conditioned_positive, anchor)
        else:
            equiv_loss_total = mse_loss(anchor_conv1, positive_conv1) + mse_loss(conditioned_anchor,
                                                                                 positive) + mse_loss(
                conditioned_positive, anchor)
        losses_dict['equiv_loss'] = equiv_loss_total
        total_loss += self.equivalent_loss_weight * losses_dict['equiv_loss']
        return losses_dict, total_loss

    @tf.function
    def train_on_batch(self, inputs):
        anchor, positive, negative = inputs
        input_tensor = tf.concat([anchor, positive, negative], axis=0)
        with tf.GradientTape() as tape1:
            anchor_y_pred, anchor_embed, _, _ = self.model(compose_input(anchor, self.pool_shape), training=True)
            positive_y_pred, positive_embed, _, _ = self.model(compose_input(positive, self.pool_shape), training=True)
            negative_y_pred, negative_embed, _, _ = self.model(compose_input(negative, self.pool_shape), training=True)
            y_pred = tf.concat([anchor_y_pred, positive_y_pred, negative_y_pred], axis=0)
            losses_1, loss_1 = self.compute_losses_part_1(input_tensor, anchor, positive, y_pred, positive_y_pred,
                                                          anchor_embed, positive_embed, negative_embed)
        gradients = [g / (tf.norm(g) + 1e-8) if g is not None else g for g in
                     tape1.gradient(loss_1, self.model.trainable_weights)]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        metrics = compute_metrics(input_tensor, y_pred)
        with tf.GradientTape() as tape2:
            anchor_conv1, anchor_embed2 = self.model.embed(anchor, training=True)
            positive_conv1, positive_embed2 = self.model.embed(positive, training=True)
            if self.perturbation:
                anchor_embed2 = anchor_embed2 + tf.random.normal(shape=anchor_embed2.shape, mean=0.,
                                                                 stddev=self.perturbation_intensity)
            conditioned_positive = self.model.pop_out(anchor_embed2, positive_conv1, training=True)
            conditioned_anchor = self.model.pop_out(positive_embed2, anchor_conv1, training=True)
            losses_2, loss_2 = self.compute_losses_part_2(anchor, positive, anchor_conv1, positive_conv1,
                                                          conditioned_positive, conditioned_anchor)
        gradients = [g / (tf.norm(g) + 1e-8) if g is not None else g for g in
                     tape2.gradient(loss_2, self.model.trainable_weights)]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        losses = {**losses_1, **losses_2}
        return losses, metrics

    @tf.function
    def validate_on_batch(self, inputs):
        anchor, positive, negative = inputs
        input_tensor = tf.concat([anchor, positive, negative], axis=0)
        anchor_y_pred, anchor_embed, _, _ = self.model(compose_input(anchor, self.pool_shape), training=False)
        positive_y_pred, positive_embed, _, _ = self.model(compose_input(positive, self.pool_shape), training=False)
        negative_y_pred, negative_embed, _, _ = self.model(compose_input(negative, self.pool_shape), training=False)
        y_pred = tf.concat([anchor_y_pred, positive_y_pred, negative_y_pred], axis=0)
        losses_1 = self.compute_losses_part_1(input_tensor, anchor, positive, y_pred,
                                              positive_y_pred, anchor_embed, positive_embed, negative_embed)[0]
        anchor_conv1, anchor_embed2 = self.model.embed(anchor, training=False)
        positive_conv1, positive_embed2 = self.model.embed(positive, training=False)
        if self.perturbation:
            anchor_embed2 = anchor_embed2 + tf.random.normal(shape=anchor_embed2.shape, mean=0.,
                                                             stddev=self.perturbation_intensity)
        conditioned_positive = self.model.pop_out(anchor_embed2, positive_conv1, training=False)
        conditioned_anchor = self.model.pop_out(positive_embed2, anchor_conv1, training=False)
        losses_2 = self.compute_losses_part_2(anchor, positive, anchor_conv1, positive_conv1, conditioned_positive,
                                              conditioned_anchor)[0]
        losses = {**losses_1, **losses_2}
        metrics = compute_metrics(input_tensor, y_pred)
        return losses, metrics

    def train(self, resume=False):
        if resume:
            self.resume()
            print('Resuming AE Training...')
        for ep in range(self.curr_epoch, self.epochs):
            self.curr_epoch += 1
            for pos, value in enumerate(self.train_gen):
                losses, metrics = self.train_on_batch(value)
                self.display_status(ep, pos, losses, metrics, validation=False)
            val_losses = []
            val_metrics = []
            for value in self.val_gen:
                losses, metrics = self.validate_on_batch(value)
                val_losses.append(losses)
                val_metrics.append(metrics)
            self.display_status(ep, ep, val_losses, val_metrics, validation=True)
            self.save_model(val_losses, val_metrics)
            if ep % self.display_frequency == 0:
                self.display_performance_image(path=self.images_dir)
        self.display_performance_image(path=self.images_dir)
        return self.best_model


class LatentTrainerCA(Trainer):
    def __init__(self, params):
        super().__init__(params)
        self.model_params = self.params['model_params']
        self.input_shape = self.model_params['input_shape']
        self.pool_shape = self.model_params['pool_shape']
        self.num_down_sampling = self.params['num_down_sampling']
        if self.num_down_sampling == 2:
            self.model = AutoEncoderDown2(self.model_params)
            self.best_model = AutoEncoderDown2(self.model_params)
        else:
            self.model = AutoEncoderDown3(self.model_params)
            self.best_model = AutoEncoderDown3(self.model_params)
        self.model(
            compose_input(tf.random.normal(self.input_shape, dtype=tf.float32), self.pool_shape, 64, 0.5, None, False,
                          True))
        self.best_model(
            compose_input(tf.random.normal(self.input_shape, dtype=tf.float32), self.pool_shape, 64, 0.5, None, False,
                          True))
        self.model.set_weights(self.params['model'].get_weights())
        self.best_model.set_weights(self.params['model'].get_weights())
        self.model.change_mod(latent_ca_trainable=True, autoencoder_trainable=False)
        self.best_model.change_mod(latent_ca_trainable=True, autoencoder_trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(self.params['learning_rate'],
                                                                    self.epochs * len(self.train_gen)))
        self.train_gen.switch_mode(train_ca=True)
        self.val_gen.switch_mode(train_ca=True)
        self.test_gen.switch_mode(train_ca=True)
        self.logs_directory_train = os.path.join(self.logs_dir, 'train_ca_log.pickle')
        self.logs_directory_val = os.path.join(self.logs_dir, 'val_ca_log.pickle')
        self.zero_factor = tf.constant(0.)
        self.reconstruction_loss = self.params['reconstruction_loss']
        self.reconstruction_loss_weight = self.params['reconstruction_loss_weight']
        self.latent_loss = self.params['latent_loss']
        self.latent_loss_weight = self.params['latent_loss_weight']
        self.output_overflow_loss = self.params['output_overflow_loss']
        self.output_overflow_loss_weight = self.params['output_overflow_loss_weight']
        self.hidden_overflow_loss = self.params['hidden_overflow_loss']
        self.hidden_overflow_loss_weight = self.params['hidden_overflow_loss_weight']
        self.total_variation_loss = self.params['total_variation_loss']
        self.total_variation_loss_weight = self.params['total_variation_loss_weight']
        self.convX = None
        self.convY = None
        if self.total_variation_loss:
            self.convX, self.convY = init_total_variation_loss_filters(input_shape=self.input_shape)

        self.pool_length = self.params['pool_length']
        self.pool = []
        self.update_probability = self.params['update_probability']
        self.min_cell_updates = self.params['min_cell_updates']
        self.max_cell_updates = self.params['max_cell_updates']

    def display_performance_image(self, path):
        self.test_gen.on_epoch_end()
        anchor, positive = self.test_gen.__getitem__(0)
        anchor, positive = anchor[: self.test_length], positive[: self.test_length]
        positive_pred, _, _, _ = self.best_model(compose_input(positive, self.pool_shape, self.max_cell_updates * 2,
                                                               self.update_probability, None, False, True),
                                                 training=False)
        images = (tensor_to_images(anchor), tensor_to_images(positive_pred), tensor_to_images(positive))
        rows, columns = self.test_length, 3
        f, ax = plt.subplots(rows, columns, figsize=(self.test_length * 32, self.test_length * 16))
        for i in range(rows):
            for pos, block in enumerate(images):
                ax[i, pos].imshow(block[i])
        if path is not None:
            plt.savefig(os.path.join(path, f'epoch_{self.curr_epoch}_ca.png'))
        plt.close()

    def display_status(self, epoch, step, losses, metrics, validation=False):
        message = ''
        if validation:
            losses = merge_metrics(losses)
            metrics = merge_metrics(metrics)
            for loss, metric in zip(losses, metrics):
                for lo in loss.items():
                    name_lo = 'val_{}'.format(lo[0])
                    self.log_history.setdefault(name_lo, []).append(lo[1])
                for me in metric.items():
                    name_me = 'val_{}'.format(me[0])
                    self.log_history.setdefault(name_me, []).append(me[1])
            for key in self.log_history.keys():
                if 'val_' in key:
                    message += ' - ' + key + ': {:.3f}'.format(
                        np.mean(self.log_history[key][-self.log_history_length_val:]))
            print(' || ' + message[3:], end='\n')
            self.log_history = {}
        else:
            losses = {normal[0]: (normal[1].numpy() + sample[1].numpy()) / 2 for normal, sample in
                      zip(losses[0].items(), losses[1].items())}
            metrics = {normal[0]: (normal[1].numpy() + sample[1].numpy()) / 2 for normal, sample in
                       zip(metrics[0].items(), metrics[1].items())}
            for data in {**losses, **metrics}.items():
                name = 'train_{}'.format(data[0])
                self.log_history.setdefault(name, []).append(data[1])
                message += ' - ' + name + ': {:.3f}'.format(
                    np.mean(self.log_history[name][-self.log_history_length_train:]))
            print('\r{}/{} epoch - {}/{} batch'.format(epoch + 1, self.epochs, step, len(self.train_gen) - 1) + message,
                  end='')
        self.log_on_file(epoch, losses, metrics, validation)

    def pool_sample(self):
        pool_y, pool_x, pool_latent_ca_out = zip(*self.pool[:self.input_shape[0]])
        y_true = tf.concat([tf.expand_dims(t, axis=0) for t in pool_y], axis=0)
        x = tf.concat([tf.expand_dims(t, axis=0) for t in pool_x], axis=0)
        latent_ca_out = tf.concat([tf.expand_dims(t, axis=0) for t in pool_latent_ca_out], axis=0)
        return y_true, x, latent_ca_out

    def pool_update(self, sampled, y_true, x, latent_ca_out):
        if sampled:
            self.pool[:self.input_shape[0]] = zip(y_true, x, latent_ca_out)
        else:
            self.pool += zip(y_true, x, latent_ca_out)
        np.random.shuffle(self.pool)
        self.pool = self.pool[:self.pool_length]

    @tf.function
    def compute_losses_ca(self, y_true, y_pred, latent_ca_out, z_layer_true):
        losses_dict = {}
        total_loss = 0.
        channel_out = self.model.get_channel_out(latent_ca_out)
        channel_hidden = self.model.get_channel_hidden(latent_ca_out)
        if self.reconstruction_loss is not None:
            if self.reconstruction_loss == 'SSIM':
                losses_dict['rec_loss'] = ssim_loss(y_true, y_pred)
            elif self.reconstruction_loss == 'PSNR':
                losses_dict['rec_loss'] = psnr_loss(y_true, y_pred)
            else:
                losses_dict['rec_loss'] = mse_loss(y_true, y_pred)
            total_loss += self.reconstruction_loss_weight * losses_dict['rec_loss']
        if self.latent_loss is not None:
            if self.latent_loss == 'SSIM':
                losses_dict['latent_loss'] = ssim_loss(z_layer_true, channel_out)
            elif self.latent_loss == 'PSNR':
                losses_dict['latent_loss'] = psnr_loss(z_layer_true, channel_out)
            else:
                losses_dict['latent_loss'] = mse_loss(z_layer_true, channel_out)
            total_loss += self.latent_loss_weight * losses_dict['latent_loss']
        if self.output_overflow_loss:
            losses_dict['out_loss'] = tf.reduce_mean(tf.abs(channel_out - tf.clip_by_value(channel_out, 0., 1.)))
            total_loss += self.output_overflow_loss_weight * losses_dict['out_loss']
        if self.hidden_overflow_loss:
            losses_dict['hid_loss'] = tf.reduce_mean(tf.abs(channel_hidden - tf.clip_by_value(channel_hidden, -1., 1.)))
            total_loss += self.hidden_overflow_loss_weight * losses_dict['hid_loss']
        if self.total_variation_loss:
            x_grad = tf.reduce_sum(tf.exp(-tf.abs(self.convX(y_true))) * tf.abs(self.convX(y_pred)), axis=[1, 2, 3])
            y_grad = tf.reduce_sum(tf.exp(-tf.abs(self.convY(y_true))) * tf.abs(self.convY(y_pred)), axis=[1, 2, 3])
            losses_dict['tv_loss'] = tf.reduce_mean(x_grad + y_grad)
            total_loss += self.total_variation_loss_weight * losses_dict['tv_loss']
        return losses_dict, total_loss

    @tf.function
    def train_on_batch_ca(self, inputs):
        y_true, x = inputs
        _, z_layer_true = self.model.embed(y_true, training=True)
        input_data = compose_input(x, self.pool_shape, random.randint(self.min_cell_updates, self.max_cell_updates + 1),
                                   self.update_probability, None, False, True)
        with tf.GradientTape() as tape:
            y_pred, _, latent_ca_out, _ = self.model(input_data, training=True)
            losses, loss = self.compute_losses_ca(y_true, y_pred, latent_ca_out, z_layer_true)
        gradients = [g / (tf.norm(g) + 1e-8) if g is not None else g for g in
                     tape.gradient(loss, self.model.trainable_weights)]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        metrics = compute_metrics_relative(y_true, y_pred, x)
        return losses, metrics, latent_ca_out

    @tf.function
    def train_on_batch_ca_sampled(self, inputs):
        y_true, x, latent_ca_out = inputs
        _, z_layer_true = self.model.embed(y_true, training=True)
        input_data = compose_input(x, self.pool_shape, random.randint(self.min_cell_updates, self.max_cell_updates + 1),
                                   self.update_probability, latent_ca_out, True, True)
        with tf.GradientTape() as tape:
            y_pred, _, latent_ca_out, _ = self.model(input_data, training=True)
            losses, loss = self.compute_losses_ca(y_true, y_pred, latent_ca_out, z_layer_true)
        gradients = [g / (tf.norm(g) + 1e-8) if g is not None else g for g in
                     tape.gradient(loss, self.model.trainable_weights)]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        metrics = compute_metrics_relative(y_true, y_pred, x)
        return losses, metrics, latent_ca_out

    @tf.function
    def validate_on_batch_ca(self, inputs):
        y_true, x = inputs
        _, z_layer_true = self.model.embed(y_true, training=False)
        input_data = compose_input(x, self.pool_shape, 2 * self.max_cell_updates,
                                   self.update_probability, None, False, True)
        y_pred, _, latent_ca_out, _ = self.model(input_data, training=False)
        losses = self.compute_losses_ca(y_true, y_pred, latent_ca_out, z_layer_true)[0]
        metrics = compute_metrics_relative(y_true, y_pred, x)
        return losses, metrics

    def final_test(self):
        self.test_gen.on_epoch_end()
        val_losses = []
        val_metrics = []
        for value in self.test_gen:
            y_true, x = value
            _, z_layer_true = self.model.embed(y_true, training=False)
            input_data = compose_input(x, self.pool_shape, 2 * self.max_cell_updates,
                                       self.update_probability, None, False, True)
            y_pred, _, latent_ca_out, _ = self.model(input_data, training=False)
            val_losses.append(self.compute_losses_ca(y_true, y_pred, latent_ca_out, z_layer_true)[0])
            val_metrics.append(compute_metrics_relative(y_true, y_pred, x))
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
            print('Resuming CA Training...')
        for ep in range(self.curr_epoch, self.epochs):
            self.curr_epoch += 1
            for pos, value in enumerate(self.train_gen):
                losses, metrics, latent_ca_out = self.train_on_batch_ca(value)
                self.pool_update(sampled=False, y_true=value[0], x=value[1], latent_ca_out=latent_ca_out)
                value = self.pool_sample()
                losses_sample, metrics_sample, latent_ca_out_sample = self.train_on_batch_ca_sampled(value)
                self.pool_update(sampled=True, y_true=value[0], x=value[1], latent_ca_out=latent_ca_out_sample)
                self.display_status(ep, pos, [losses, losses_sample], [metrics, metrics_sample], validation=False)
            val_losses = []
            val_metrics = []
            for value in self.val_gen:
                losses, metrics = self.validate_on_batch_ca(value)
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


class LatentTrainerWrapper(Trainer):
    def __init__(self, params):
        super().__init__(params)
        self.latent_trainer_ae = None
        self.latent_trainer_ca = None
        self.train_status = False

    def resume(self):
        print('Resuming Wrapper...')
        if os.path.exists(os.path.join(self.results_dir, 'Logs', 'train_ca_log.pickle')):
            self.train_status = True

    def train(self, resume=False):
        if resume:
            self.resume()
        if not self.train_status:
            self.latent_trainer_ae = LatentTrainerAE(self.params)
            self.params['model'] = self.latent_trainer_ae.train(resume)
        else:
            if self.params['num_down_sampling'] == 2:
                self.params['model'] = AutoEncoderDown2(self.params['model_params'])
            else:
                self.params['model'] = AutoEncoderDown3(self.params['model_params'])
            self.params['model'](compose_input(tf.random.normal(self.params['model_params']['input_shape'],
                                                                dtype=tf.float32),
                                               self.params['model_params']['pool_shape'], 64, 0.5, None, False, True))
            self.params['model'].load_weights(os.path.join(self.results_dir, 'Checkpoints', 'Best', 'weights'))
            self.params['model'].change_mod(latent_ca_trainable=True, autoencoder_trainable=False)
        self.latent_trainer_ca = LatentTrainerCA(self.params)
        return self.latent_trainer_ca.train(self.train_status)
