import copy
import cv2
import multiprocessing
import os
import pickle
import random
import shutil
import threading
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from Utils.generic_utils import tensor_to_images, merge_metrics, send_to_telegram


@tf.function
def ssim_loss(x, y):
    return tf.reduce_mean(-tf.image.ssim(x, y, 1.0))


@tf.function
def psnr_loss(x, y):
    return tf.reduce_mean(-tf.image.psnr(x, y, 1.0))


@tf.function
def mse_loss(x, y):
    return tf.reduce_mean(tf.keras.losses.mse(x, y))


@tf.function
def cosine_loss(x, y):
    return 1 - tf.reduce_mean(tf.keras.losses.cosine_similarity(x, y))


@tf.function
def compute_metrics(y_true, y_pred):
    return {'PSNR': -psnr_loss(y_true, y_pred), 'SSIM': -ssim_loss(y_true, y_pred)}


@tf.function
def compute_metrics_relative(y_true, y_pred, x):
    metrics = compute_metrics(y_true, y_pred)
    base_psnr = -psnr_loss(y_true, x)
    base_ssim = -ssim_loss(y_true, x)
    metrics['REL_PSNR'] = 100 * ((metrics['PSNR'] - base_psnr + 1e-8) / (base_psnr + 1e-8))
    metrics['REL_SSIM'] = 100 * ((metrics['SSIM'] - base_ssim + 1e-8) / (base_ssim + 1e-8))
    return metrics


def compose_input(x, pool_shape, num_iter=0, update_prob=0., pool=None, pool_sampled=False, ca_mod=False):
    if ca_mod:
        if pool_sampled:
            return [x, tf.constant(value=num_iter, dtype=tf.int32, shape=(1,)),
                    tf.constant(value=update_prob, dtype=tf.float32, shape=(1,)),
                    pool, tf.constant(value=True, dtype=tf.bool, shape=(1,))]
        else:
            return [x, tf.constant(value=num_iter, dtype=tf.int32, shape=(1,)),
                    tf.constant(value=update_prob, dtype=tf.float32, shape=(1,)),
                    tf.zeros(shape=pool_shape), tf.constant(value=False, dtype=tf.bool, shape=(1,))]

    else:
        return [x, tf.constant(value=0, dtype=tf.int32, shape=(1,)),
                tf.constant(value=0, dtype=tf.float32, shape=(1,)),
                tf.zeros(shape=pool_shape), tf.constant(value=False, dtype=tf.bool, shape=(1,))]


def init_total_variation_loss_filters(input_shape):
    conv_x = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape[1:]),
        tf.keras.layers.ZeroPadding2D((0, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 2), strides=1, padding='valid', use_bias=False,
                               trainable=False)
    ])
    conv_x.get_layer(index=1).set_weights(
        [np.expand_dims(np.repeat(np.array([[[-1], [1]]]), axis=-1, repeats=input_shape[-1]), axis=-1)])
    conv_y = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape[1:]),
        tf.keras.layers.ZeroPadding2D((1, 0)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 1), strides=1, padding='valid', use_bias=False,
                               trainable=False)
    ])
    conv_y.get_layer(index=1).set_weights(
        [np.expand_dims(np.repeat(np.array([[[1]], [[-1]]]), axis=-1, repeats=input_shape[-1]), axis=-1)])
    return conv_x, conv_y


def derangement_check(a, b):
    for i, j in zip(a, b):
        if i == j:
            return False
    return True


def get_derangement(tensor, batch_size):
    x = [i for i in range(batch_size)]
    y = copy.copy(x)
    random.shuffle(y)
    while not derangement_check(x, y):
        y = copy.copy(x)
        random.shuffle(y)
    return tf.gather(params=tensor, indices=y, axis=0)


def load_image(array, base_dir, image_list, img_size, start, pbar):
    for pos, path in enumerate(image_list):
        array[pos + start] = np.array(tf.keras.preprocessing.image.load_img(os.path.join(base_dir, path),
                                                                            target_size=img_size,
                                                                            interpolation='lanczos'),
                                      dtype=np.uint8)
        pbar.update(1)


def inference_latency(model, batch_shape, num_runs, model_type='classic', warmup=1):
    batch = tf.random.uniform(shape=batch_shape, minval=0., maxval=1., dtype=tf.float32)
    time_scores = []
    for _ in tqdm(range(num_runs + warmup)):
        start_time = time.time_ns()
        if model_type == 'latent_nca':
            model(compose_input(batch, model.params['pool_shape'], 64, 0.5, None, False, True))
        elif model_type == 'nca':
            model([model.seed(batch), tf.constant(64, dtype=tf.float32, shape=(1,)),
                   tf.constant(0.5, dtype=tf.float32, shape=(1,))])
        else:
            model(batch)
        time_scores.append(time.time_ns() - start_time)
        batch = tf.random.uniform(shape=batch_shape, minval=0., maxval=1., dtype=tf.float32)
    return time_scores[warmup:]


def create_dummy_dataset(num_images, shape, path):
    for i in tqdm(range(num_images)):
        Image.fromarray(np.random.uniform(low=0., high=255., size=shape).astype(np.uint8)).save(os.path.join(path, '{}.png'.format(i)))


def get_elapsed_time(base_dir, model_name, latent=False):
    if latent:
        with open(os.path.join(base_dir, model_name, 'Logs', 'val_log.pickle'), 'rb') as f:
            df_base = pickle.load(f)
        df_base = df_base.groupby(by='epoch').mean()['elapsed_time']
        with open(os.path.join(base_dir, model_name, 'Logs', 'val_ca_log.pickle'), 'rb') as f:
            df_ca = pickle.load(f)
        df_ca = df_ca.groupby(by='epoch').mean()['elapsed_time']
        df_base += df_ca
    else:
        with open(os.path.join(base_dir, model_name, 'Logs', 'val_log.pickle'), 'rb') as f:
            df_base = pickle.load(f)
        df_base = df_base.groupby(by='epoch').mean()['elapsed_time']
    return df_base


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, filenames, img_size, task_info, train_ca=False, shuffle=False, num_iterations=50,
                 curriculum=False, train=True, real_dataset=False):
        self.directory = directory
        self.filenames = filenames
        self.batch_size = img_size[0]
        self.img_size = img_size[1:]
        self.step = -1
        self.train_ca = train_ca
        self.shuffle = shuffle
        self.num_iterations = num_iterations
        self.curriculum = curriculum
        self.train = train
        self.task_info = task_info
        self.task = self.task_info['task']
        self.real_dataset = real_dataset
        if self.real_dataset:
            self.directory = os.path.join(self.directory, 'GT')
        self.image_array = None
        self.noisy_image_array = None
        self.preload_data_multithread()
        self.on_epoch_end()

    def __getitem__(self, index):
        anchor = tf.convert_to_tensor(self.image_array[index * self.batch_size: (index + 1) * self.batch_size, :, :, :],
                                      dtype=tf.float32) / 255.
        if self.real_dataset:
            positive = tf.convert_to_tensor(
                self.noisy_image_array[index * self.batch_size: (index + 1) * self.batch_size, :, :, :],
                dtype=tf.float32) / 255.
        else:
            if self.curriculum and self.train:
                self.task_info['intensity'] = np.clip(a=(self.step / self.num_iterations) * self.task_info['range'],
                                                      a_min=self.task_info['a_min'], a_max=self.task_info['a_max'])
            if self.task == 'Noise':
                positive = tf.convert_to_tensor(self.create_mask_noise(self.task_info['intensity'], anchor),
                                                dtype=tf.float32)
            else:
                positive = tf.convert_to_tensor(self.create_mask_blur(self.task_info['intensity'], anchor),
                                                dtype=tf.float32)

        if index == self.__len__() - 1:
            self.on_epoch_end()

        if self.train_ca:
            return anchor, positive
        else:
            negative = get_derangement(anchor, self.batch_size)
            return anchor, positive, negative

    def __len__(self):
        return self.image_array.shape[0] // self.batch_size

    def switch_mode(self, train_ca=False):
        self.train_ca = train_ca
        self.step = 0

    def create_mask_noise(self, intensity, img):
        mask = np.random.choice([0., 1.], size=(self.batch_size,) + self.img_size[:-1] + (1,),
                                p=[1 - intensity, intensity])
        signal = np.random.normal(loc=0., scale=0.2, size=(self.batch_size,) + self.img_size[:-1] + (1,))
        return np.clip(img + np.multiply(mask, signal), a_min=0., a_max=1.)

    def create_mask_blur(self, intensity, img):
        intensity = int(intensity)
        angle = np.random.randint(low=0, high=360, size=self.batch_size)
        kernels = np.zeros(shape=(self.batch_size, intensity, intensity), dtype=np.float32)
        kernels[:, (intensity - 1) // 2, :] = np.ones(intensity, dtype=np.float32)
        d = intensity / 2 - 0.5
        kernels = [cv2.warpAffine(kernels[pos], cv2.getRotationMatrix2D((d, d), an, 1.0), (intensity, intensity))
                   for pos, an in enumerate(angle)]
        kernels = [
            np.repeat(np.expand_dims(np.expand_dims(sl * (1. / np.sum(sl)), axis=-1), axis=-1), repeats=3, axis=-1)
            for sl in kernels]
        return tf.concat(
            [tf.nn.conv2d(tf.expand_dims(b, axis=0), k, [1, 1, 1, 1], 'SAME') for b, k in zip(img, kernels)], axis=0)

    def preload_data_multithread(self):
        self.image_array = np.empty(shape=(len(self.filenames), *self.img_size), dtype=np.uint8)
        threads = []
        with tqdm(total=len(self.filenames)) as pbar:
            for pos in range(multiprocessing.cpu_count()):
                start = int((len(self.filenames) / multiprocessing.cpu_count()) * pos)
                end = int((len(self.filenames) / multiprocessing.cpu_count()) * (pos + 1))
                x = threading.Thread(target=load_image,
                                     args=(
                                         self.image_array, self.directory, self.filenames[start:end], self.img_size,
                                         start,
                                         pbar))
                threads.append(x)
                x.start()

            for t in threads:
                t.join()
        if self.real_dataset:
            self.noisy_image_array = np.empty(shape=(len(self.filenames), *self.img_size), dtype=np.uint8)
            threads = []
            with tqdm(total=len(self.filenames)) as pbar:
                for pos in range(multiprocessing.cpu_count()):
                    start = int((len(self.filenames) / multiprocessing.cpu_count()) * pos)
                    end = int((len(self.filenames) / multiprocessing.cpu_count()) * (pos + 1))
                    x = threading.Thread(target=load_image,
                                         args=(self.noisy_image_array, self.directory.replace('GT', 'DAMAGED'),
                                               self.filenames[start:end],
                                               self.img_size, start, pbar))
                    threads.append(x)
                    x.start()

                for t in threads:
                    t.join()

    def on_epoch_end(self):
        self.step += 1
        if self.shuffle:
            if self.real_dataset:
                p = np.random.permutation(len(self.image_array))
                self.image_array = self.image_array[p]
                self.noisy_image_array = self.noisy_image_array[p]
            else:
                np.random.shuffle(self.image_array)


class Trainer:
    def __init__(self, params):
        self.params = params
        self.model_name = self.params['model_name']
        self.epochs = self.params['epochs']
        self.curr_epoch = 0
        self.train_gen = self.params['train']
        self.val_gen = self.params['val']
        self.test_gen = self.params['test']

        self.model = None
        self.best_model = None
        self.optimizer = None

        self.monitor_metric = self.params['monitor_metric']
        self.best_value = 0.
        self.log_history = {}
        self.full_log_history_train = pd.DataFrame()
        self.full_log_history_val = pd.DataFrame()
        self.log_history_length_train = len(self.train_gen)
        self.log_history_length_val = len(self.val_gen)
        self.display_frequency = self.params['display_frequency']
        self.test_length = self.params['test_length']
        self.results_dir = self.params['results_dir']
        self.last_weights_dir = ''
        self.best_weights_dir = ''
        self.images_dir = ''
        self.logs_dir = ''
        self.create_log_dir()
        self.logs_directory_train = os.path.join(self.logs_dir, 'train_log.pickle')
        self.logs_directory_val = os.path.join(self.logs_dir, 'val_log.pickle')
        self.logs_directory_test = os.path.join(self.logs_dir, 'test_log.pickle')

        self.token = self.params['token']
        self.chat_id = self.params['chat_id']

        self.start_time = time.time()
        self.end_time = time.time()
        self.total_elapsed_time = 0

    def create_log_dir(self):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        base_checkpoints_dir = os.path.join(self.results_dir, 'Checkpoints')
        if not os.path.exists(base_checkpoints_dir):
            os.makedirs(base_checkpoints_dir)
        self.last_weights_dir = os.path.join(base_checkpoints_dir, 'Last')
        if not os.path.exists(self.last_weights_dir):
            os.makedirs(self.last_weights_dir)
        self.best_weights_dir = os.path.join(base_checkpoints_dir, 'Best')
        if not os.path.exists(self.best_weights_dir):
            os.makedirs(self.best_weights_dir)
        self.images_dir = os.path.join(self.results_dir, 'Images')
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        self.logs_dir = os.path.join(self.results_dir, 'Logs')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def save_model(self, val_losses, val_metrics):
        self.model.save_weights(os.path.join(self.last_weights_dir, 'weights'))
        joined = [{**i, **j} for i, j in zip(val_losses, val_metrics)]
        value = sum(d[self.monitor_metric] for d in joined) / len(joined)
        if value > self.best_value:
            print('Found new Best, saving...')
            self.best_value = value
            self.best_model.set_weights(self.model.get_weights())
            self.best_model.save_weights(os.path.join(self.best_weights_dir, 'weights'))

    def display_performance_image(self, path):
        self.test_gen.on_epoch_end()
        anchor, positive = self.test_gen.__getitem__(0)
        anchor, positive = anchor[: self.test_length], positive[: self.test_length]
        positive_pred = self.best_model(positive, training=False)
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
            losses = {key: value.numpy() for key, value in losses.items()}
            metrics = {key: value.numpy() for key, value in metrics.items()}
            for data in {**losses, **metrics}.items():
                name = 'train_{}'.format(data[0])
                self.log_history.setdefault(name, []).append(data[1])
                message += ' - ' + name + ': {:.3f}'.format(
                    np.mean(self.log_history[name][-self.log_history_length_train:]))
            print('\r{}/{} epoch - {}/{} batch'.format(epoch + 1, self.epochs, step, len(self.train_gen) - 1) + message,
                  end='')
        self.log_on_file(epoch, losses, metrics, validation)

    def log_on_file(self, epoch, losses, metrics, validation=False):
        if validation:
            new_line = pd.DataFrame([{**i, **j} for i, j in zip(losses, metrics)])
            new_line['epoch'] = epoch
            new_line['elapsed_time'] = self.time_update()
            self.full_log_history_val = pd.concat([self.full_log_history_val, new_line]).reset_index(drop=True)
            with open(self.logs_directory_val, "wb") as log_file:
                pickle.dump(self.full_log_history_val, log_file)
            if self.token is not None:
                send_to_telegram('Model   {}\n{}'.format(self.model_name, new_line.mean().to_string()),
                                 self.token, self.chat_id)
        else:
            new_line = pd.DataFrame({**losses, **metrics}, index=[0])
            new_line['epoch'] = epoch
            self.full_log_history_train = pd.concat([self.full_log_history_train, new_line]).reset_index(drop=True)
            with open(self.logs_directory_train, "wb") as log_file:
                pickle.dump(self.full_log_history_train, log_file)

    def time_update(self):
        self.end_time = time.time()
        delta = self.end_time - self.start_time
        self.total_elapsed_time += delta
        print(' || elapsed_time: {:.3f}'.format(delta), end='\n')
        self.start_time = time.time()
        return delta

    def resume(self):
        print('Resuming Training...')
        if os.path.exists(os.path.join(self.last_weights_dir, 'checkpoint')):
            print('Reloaded Last...')
            self.model.load_weights(os.path.join(self.last_weights_dir, 'weights'))
        if os.path.exists(os.path.join(self.best_weights_dir, 'checkpoint')):
            print('Reloaded Best...')
            self.best_model.load_weights(os.path.join(self.best_weights_dir, 'weights'))
        if os.path.exists(self.logs_directory_val):
            with open(self.logs_directory_val, "rb") as log_file_val:
                self.full_log_history_val = pickle.load(log_file_val)
            self.curr_epoch = self.full_log_history_val['epoch'].max() + 1
            self.full_log_history_val = self.full_log_history_val[self.full_log_history_val.epoch < self.curr_epoch]
            self.best_value = self.full_log_history_val.groupby(by='epoch').mean()[self.monitor_metric].max()
        if os.path.exists(self.logs_directory_train):
            with open(self.logs_directory_train, "rb") as log_file_train:
                self.full_log_history_train = pickle.load(log_file_train)
            self.full_log_history_train = self.full_log_history_train[
                self.full_log_history_train.epoch < self.curr_epoch]
        self.train_gen.step = self.curr_epoch
        self.val_gen.step = self.curr_epoch
        self.test_gen.step = self.curr_epoch

    def final_test(self):
        pass

    def clean_folder(self):
        shutil.rmtree(self.last_weights_dir)

    @tf.function
    def train_on_batch(self, inputs):
        pass

    @tf.function
    def validate_on_batch(self, inputs):
        pass

    def train(self, resume=False):
        pass


class TestLauncher:
    def __init__(self, params):
        self.params = params
        self.base_dir = self.params['base_dir']
        self.task = self.params['task']
        self.dataset = self.params['dataset']
        self.test_dir = os.path.join(self.base_dir, self.task, self.dataset)
        self.epochs = self.params['epochs']
        self.train_gen = self.params['train']
        self.val_gen = self.params['val']
        self.test_gen = self.params['test']
        self.monitor_metric = self.params['monitor_metric']
        self.display_frequency = self.params['display_frequency']
        self.test_length = self.params['test_length']
        self.model_params_dicts = self.params['model_params_dicts']
        self.token = self.params['telegram_token']
        self.chat_id = self.params['telegram_chat_id']
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def check_finished(self, path):
        if os.path.exists(os.path.join(path, 'Logs', 'val_log.pickle')):
            with open(os.path.join(path, 'Logs', 'val_log.pickle'), 'rb') as f:
                curr_epoch = pickle.load(f)['epoch'].max()
            if curr_epoch == self.epochs - 1 and 'Latent' in path:
                if os.path.exists(os.path.join(path, 'Logs', 'val_ca_log.pickle')):
                    with open(os.path.join(path, 'Logs', 'val_ca_log.pickle'), 'rb') as f:
                        curr_epoch = pickle.load(f)['epoch'].max()
                else:
                    curr_epoch = 0
        else:
            curr_epoch = 0
        return curr_epoch == self.epochs - 1

    def complete_dict(self, d):
        d['results_dir'] = os.path.join(self.test_dir, d['model_name'])
        self.train_gen.step = 0
        self.val_gen.step = 0
        self.test_gen.step = 0
        d['train'] = self.train_gen
        d['val'] = self.val_gen
        d['test'] = self.test_gen
        d['epochs'] = self.epochs
        d['monitor_metric'] = self.monitor_metric
        d['display_frequency'] = self.display_frequency
        d['test_length'] = self.test_length
        d['token'] = self.token
        d['chat_id'] = self.chat_id
        return d

    def launch(self):
        for d in self.model_params_dicts:
            folder = os.path.join(self.test_dir, d['model_name'])
            exists = os.path.exists(folder)
            if exists:
                if self.check_finished(folder):
                    print('{} already trained, skipping...'.format(d['model_name']))
                    continue
            d = self.complete_dict(d)
            t = d['trainer'](d)
            print('Start {}\'s Training...'.format(d['model_name']))
            t.train(exists)
