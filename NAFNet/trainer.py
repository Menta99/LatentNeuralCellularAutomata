import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from NAFNet.model import NAFNet
from Utils.trainer_utils import compute_metrics_relative, Trainer, merge_metrics, send_to_telegram


class TrainerNAFNet(Trainer):
    def __init__(self, params):
        super().__init__(params)
        self.model_params = self.params['model_params']
        self.input_shape = self.model_params['input_shape']
        self.batch_size = self.input_shape[0]
        self.model = NAFNet(self.model_params)
        self.best_model = NAFNet(self.model_params)
        self.model(tf.convert_to_tensor(np.random.random(self.input_shape), dtype=tf.float32))
        self.best_model(tf.convert_to_tensor(np.random.random(self.input_shape), dtype=tf.float32))
        self.best_model.set_weights(self.model.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.CosineDecay(self.params['learning_rate'],
                                                                    self.epochs * len(self.train_gen)))
        self.train_gen.switch_mode(train_ca=True)
        self.val_gen.switch_mode(train_ca=True)
        self.test_gen.switch_mode(train_ca=True)

    @tf.function
    def compute_losses(self, y_true, y_pred):
        losses_dict = {'PSNR_loss': tf.reduce_mean(-tf.image.psnr(y_true, y_pred, 1.0))}
        total_loss = losses_dict['PSNR_loss']
        return losses_dict, total_loss

    @tf.function
    def train_on_batch(self, inputs):
        y_true, x = inputs
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            losses, loss = self.compute_losses(y_true, y_pred)
        gradients = [g / (tf.norm(g) + 1e-8) for g in tape.gradient(loss, self.model.trainable_weights)]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        metrics = compute_metrics_relative(y_true, y_pred, x)
        return losses, metrics

    @tf.function
    def validate_on_batch(self, inputs):
        y_true, x = inputs
        y_pred = self.model(x, training=False)
        losses = self.compute_losses(y_true, y_pred)[0]
        metrics = compute_metrics_relative(y_true, y_pred, x)
        return losses, metrics

    def final_test(self):
        self.test_gen.on_epoch_end()
        val_losses = []
        val_metrics = []
        for value in self.test_gen:
            y_true, x = value
            y_pred = self.best_model(x, training=False)
            val_losses.append(self.compute_losses(y_true, y_pred)[0])
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
        self.final_test()
        self.clean_folder()
        return self.best_model
