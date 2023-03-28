import numpy as np

CONFIG_LAYER = [(24, 24, 1),
                (24, 48, 2),
                (48, 48, 1),
                (48, 48, 1),
                (48, 48, 1),
                (48, 48, 1),
                (48, 96, 2),
                (96, 96, 1),
                (96, 120, 2),
                (120, 180, 2),
                (180, 180, 1),
                ]

CONFIG_SUPERNET = {
    'default_settings': {
        'gpu_ids': 0,
        'db_id': 'root',
        'db_password': '1204',
        'seed': 100,
    },
    'dataloading': {
        'img_size': 32,
        'batch_size': 128,
        'test_batch_size': 256,
        'w_share_in_train': 0.8,
        'path_to_save_data': './cifar10_data',
        'path_to_save_logger': './log.txt',
    },
    'optimizer': {
        # SGD parameters for w
        'w_lr': 0.01,
        'w_momentum': 0.9,
        'w_weight_decay': 1e-4,
        # Adam parameters for thetas
        'thetas_lr': 0.01,
        'thetas_weight_decay': 5 * 1e-4
    },
    'loss': {
        'alpha': 0.2,
        'beta': 0.6,
    },
    'train_settings': {
        'cnt_epochs': 90,
        'train_thetas_from_the_epoch': 20,
        'path_to_save_model': './best_model.pth',
        'path_to_save_log': './log.txt',
        # for Gumbel Softmax
        'init_temperature': 5.0,
        'exp_anneal_rate': np.exp(-0.045),
        # first, last config
        'last_feature_size': 32,
        'first_stride': 1,
        'first_inchannel': 3,
        'cnt_classes': 10,
        'max_priority': 4,
    },
}


CONFIG_SAMPLE = {
    'train_settings': {
        'train_from_scratch': False,
        'cnt_epochs': 10,
    },
    'optimizer': {
        'w_lr': 0.01,
        'w_momentum': 0.9,
        'w_weight_decay': 1e-4,
    },
    'dataloading': {
        'path_to_load_model': './best_model.pth',
        'path_to_save_logger': './log.txt',
        'path_to_save_model': './sampled_model.pth',
        'path_to_save_jit': './jit_model.pt',
        'batch_size': 256,
    }
}
