import copy
import tensorflow as tf

from LatentCA.model import NAFCA
from LatentCA.trainer import LatentTrainerWrapper
from NAFNet.trainer import TrainerNAFNet
from Restormer.trainer import TrainerRestormer
from Utils.generic_utils import get_reduced_name_list
from Utils.trainer_utils import DataGenerator, TestLauncher
from ViTCA.model import ViTCA
from ViTCA.trainer import TrainerViTCA

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    relative_path = '../'
    dataset_directory = relative_path + 'Datasets/CelebA/'
    img_shape = (8, 32, 32, 3)
    num_iterations = 10
    reduction = 1.
    test_split = 0.8
    val_split = 0.8
    curriculum_learning = True
    task_dict = {'task': 'Noise', 'range': 1., 'intensity': 0.5, 'a_min': 0.25, 'a_max': 0.75}
    # task_dict = {'task': 'Blur', 'range': 15, 'intensity': 7, 'a_min': 4, 'a_max': 10}
    real_dataset = False
    train, val, test = get_reduced_name_list(path=dataset_directory, reduction=reduction, test_split=test_split,
                                             val_split=val_split, real_dataset=real_dataset)
    train_gen = DataGenerator(directory=dataset_directory, filenames=train, img_size=img_shape,
                              task_info=copy.deepcopy(task_dict), train_ca=True, shuffle=True,
                              num_iterations=num_iterations, curriculum=curriculum_learning, train=True,
                              real_dataset=real_dataset)
    val_gen = DataGenerator(directory=dataset_directory, filenames=val, img_size=img_shape,
                            task_info=copy.deepcopy(task_dict), train_ca=True, shuffle=False,
                            num_iterations=num_iterations, curriculum=False, train=False, real_dataset=real_dataset)
    test_gen = DataGenerator(directory=dataset_directory, filenames=test, img_size=img_shape,
                             task_info=copy.deepcopy(task_dict), train_ca=True, shuffle=False,
                             num_iterations=num_iterations, curriculum=False, train=False, real_dataset=real_dataset)

    common_params = {
        'base_dir': relative_path + 'Results/',
        'dataset': dataset_directory.split('/')[-2],
        'task': task_dict['task'],
        'epochs': num_iterations,
        'train': train_gen,
        'val': val_gen,
        'test': test_gen,
        'monitor_metric': 'SSIM',
        'display_frequency': 2,
        'test_length': 4,
        'model_params_dicts': None,
        'telegram_token': None,
        'telegram_chat_id': None
    }

    max_depth = 16
    num_down_sampling = 2
    cell_in_channels = cell_out_channels = max_depth
    cell_hidden_channels = 32
    pool_shape = (img_shape[0], img_shape[1] // (2 ** num_down_sampling), img_shape[1] // (2 ** num_down_sampling),
                  cell_in_channels + cell_out_channels + cell_hidden_channels)
    latent_shape = (img_shape[1] // (2 ** num_down_sampling), img_shape[1] // (2 ** num_down_sampling), max_depth)
    latent_params = {
        'model_params': {
            'input_shape': img_shape,
            'pool_shape': pool_shape,
            'latent_shape': latent_shape,
            'CAParams': None,
            'Conv2DParams1': {
                'filters': max_depth // 2,
                'kernel_size': (3, 3),
                'strides': (2, 2),
                'padding': 'same',
                'activation': 'swish',
                'batch_normalization': True,
                'dropout_rate': 0.0
            },
            'Conv2DParams2': {
                'filters': max_depth,
                'kernel_size': (3, 3),
                'strides': (2, 2),
                'padding': 'same',
                'activation': 'swish',
                'batch_normalization': True,
                'dropout_rate': 0.0
            },
            'PassThroughParams1': {
                'filters': max_depth // 2,
                'kernel_size': (3, 3),
                'strides': (2, 2),
                'padding': 'same',
                'activation': 'swish',
                'batch_normalization': True,
                'dropout_rate': 0.0
            },
            'Conv2DTransposeParams2': {
                'filters': max_depth // 2,
                'kernel_size': (3, 3),
                'strides': (2, 2),
                'padding': 'same',
                'activation': 'swish',
                'batch_normalization': True,
                'dropout_rate': 0.0
            },
            'MixParams': {
                'filters': max_depth // 2,
                'kernel_size': (3, 3),
                'strides': (1, 1),
                'padding': 'same',
                'activation': 'swish',
                'batch_normalization': True,
                'dropout_rate': 0.0
            },
            'Conv2DTransposeParams1': {
                'filters': 3,
                'kernel_size': (3, 3),
                'strides': (2, 2),
                'padding': 'same',
                'activation': 'sigmoid',
                'batch_normalization': True,
                'dropout_rate': 0.0
            }},
        'trainer': LatentTrainerWrapper,
        'learning_rate': 1e-2,
        'num_down_sampling': num_down_sampling,
        'margin': 1.,
        'reconstruction_loss_ae': 'MSE',
        'reconstruction_loss_weight_ae': 100,
        'distance_loss': 'MSE',
        'distance_loss_weight': 10,
        'task_loss': 'MSE',  # PSNR
        'task_loss_weight': 10,
        'equivalent_loss': 'PureNoiseMSE',
        'equivalent_loss_weight': 100,
        'perturbation_intensity': 0.5,
        'total_variation_loss_ae': False,
        'total_variation_loss_weight_ae': 1,
        'reconstruction_loss': 'MSE',
        'reconstruction_loss_weight': 100,
        'latent_loss': 'MSE',
        'latent_loss_weight': 100,
        'output_overflow_loss': True,
        'output_overflow_loss_weight': 1,
        'hidden_overflow_loss': True,
        'hidden_overflow_loss_weight': 1,
        'total_variation_loss': False,
        'total_variation_loss_weight': 1,
        'update_probability': 1.,
        'min_cell_updates': 8,
        'max_cell_updates': 32,
        'pool_length': 1024
    }
    latent_nafca_params = {
        'class': NAFCA,
        'localized_attention_neighbourhood': (3, 3),
        'ffn_expand': 4,
        'dropout': 0.1,
        'embed_dim': 128,
        'cell_in_channels': cell_in_channels,
        'cell_out_channels': cell_out_channels,
        'cell_hidden_channels': cell_hidden_channels
    }
    latent_vitca_params = {
        'class': ViTCA,
        'localized_attention_neighbourhood': (3, 3),
        'patch_size': 1,
        'pos_encoding_max_freq': 5,
        'depth': 1,
        'heads': 4,
        'mlp_dim': 64,
        'dropout': 0.0,
        'embed_cells': True,
        'embed_dim': 128,
        'embed_dropout': 0.0,
        'pool_shape': pool_shape,
        'cell_in_channels': cell_in_channels,
        'cell_out_channels': cell_out_channels,
        'cell_hidden_channels': cell_hidden_channels
    }
    nafnet_params = {
        'model_name': 'NAFNet',
        'trainer': TrainerNAFNet,
        'model_params': {
            'input_shape': img_shape,
            'width': 64,
            'enc_block_nums': [2, 2, 4, 8],
            'middle_block_num': 12,
            'dec_block_nums': [2, 2, 2, 2]},
        'learning_rate': 1e-3
    }
    restormer_params = {
        'model_name': 'Restormer',
        'trainer': TrainerRestormer,
        'model_params': {
            'input_shape': img_shape,
            'out_channels': 3,
            'dim': 48,
            'num_blocks': [4, 6, 6, 8],
            'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8],
            'ffn_expansion_factor': 2.66,
            'bias': False},
        'learning_rate': 3e-4,
    }
    vitca_params = {
        'model_name': 'ViTCA',
        'trainer': TrainerViTCA,
        'model_params': {
            'input_shape': img_shape,
            'localized_attention_neighbourhood': [3, 3],
            'patch_size': 1,
            'overlapping_patches': False,
            'pos_encoding_method': 'vit_handcrafted',
            'pos_encoding_basis': 'raw_xy',
            'pos_encoding_max_freq': 5,
            'depth': 1,
            'heads': 4,
            'mlp_dim': 64,
            'dropout': 0.0,
            'cell_init': 'constant',
            'cell_in_channels': img_shape[-1],
            'cell_out_channels': img_shape[-1],
            'cell_hidden_channels': cell_hidden_channels,
            'embed_cells': True,
            'embed_dim': 128,
            'embed_dropout': 0.0},
        'learning_rate': 1e-3,
        'output_overflow_loss': True,
        'hidden_overflow_loss': True,
        'reconstruction_loss_factor': 1e2,
        'overflow_loss_factor': 1e2,
        'pool_length': 1024,
        'update_probability': 0.5,
        'min_cell_updates': 8,
        'max_cell_updates': 32,
    }
    latent_nafca_params_complete = copy.deepcopy(latent_params)
    latent_nafca_params_complete['model_params']['CAParams'] = latent_nafca_params
    latent_nafca_params_complete['model_name'] = 'LatentNAFCA'
    latent_vitca_params_complete = copy.deepcopy(latent_params)
    latent_vitca_params_complete['model_params']['CAParams'] = latent_vitca_params
    latent_vitca_params_complete['model_name'] = 'LatentViTCA'

    common_params['model_params_dicts'] = [latent_nafca_params_complete, latent_vitca_params_complete, nafnet_params,
                                           restormer_params, vitca_params]
    # common_params['model_params_dicts'] = [latent_vitca_params_complete]

    launcher = TestLauncher(common_params)
    launcher.launch()
