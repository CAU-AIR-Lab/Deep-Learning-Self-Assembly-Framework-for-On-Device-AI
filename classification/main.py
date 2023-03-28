import numpy as np
import os
import argparse
import torch
from torch import nn
from utils import weights_init, check_tensor_in_list, get_module_list
import logging
from nas_builder.train_supernet import TrainerSupernet
from nas_builder.config_cifar10 import CONFIG_SUPERNET, CONFIG_LAYER
from nas_builder.dataloaders_cifar10 import get_loaders, get_test_loader
from nas_builder.model import SuperNet, SuperNetLoss
import sys


#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG_SUPERNET['gpu_settings']['gpu_ids']
def train_supernet(device, logger):
    module_list = get_module_list(CONFIG_SUPERNET['train_settings']['max_priority'], 
                                  CONFIG_SUPERNET['default_settings']['db_id'],
                                  CONFIG_SUPERNET['default_settings']['db_password'])
    supernet_param = {
        'config_layer': CONFIG_LAYER,
        'module_list': module_list,
        'first_inchannel': CONFIG_SUPERNET['train_settings']['first_inchannel'],
        'first_stride': CONFIG_SUPERNET['train_settings']['first_stride'],
        'last_feature_size': CONFIG_SUPERNET['train_settings']['last_feature_size'],
        'cnt_classes': CONFIG_SUPERNET['train_settings']['cnt_classes'],
    }
    model = SuperNet(supernet_param, device)
    model = model.apply(weights_init)
    model.to(device)
    criterion = SuperNetLoss(CONFIG_SUPERNET['loss']['alpha'], 
                             CONFIG_SUPERNET['loss']['beta']).to(device)

    thetas_params = [param for name, param in model.named_parameters() if 'thetas' in name]
    params_except_thetas = [param for param in model.parameters() if not check_tensor_in_list(param, thetas_params)]

    w_optimizer = torch.optim.SGD(params=params_except_thetas, 
                                  lr=CONFIG_SUPERNET['optimizer']['w_lr'],
                                  momentum=CONFIG_SUPERNET['optimizer']['w_momentum'],
                                  weight_decay=CONFIG_SUPERNET['optimizer']['w_weight_decay'])

    theta_optimizer = torch.optim.Adam(params=thetas_params,
                                       lr=CONFIG_SUPERNET['optimizer']['thetas_lr'],
                                       weight_decay=CONFIG_SUPERNET['optimizer']['thetas_weight_decay'])
     
    w_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optimizer,
                                                             T_max=CONFIG_SUPERNET['train_settings']['cnt_epochs'])
    
    train_param = {
            'temperature': CONFIG_SUPERNET['train_settings']['init_temperature'],
            'cnt_epochs': CONFIG_SUPERNET['train_settings']['cnt_epochs'],
            'train_thetas_from_the_epoch': CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch'],
            'path_to_save_model': CONFIG_SUPERNET['train_settings']['path_to_save_model'],
            'path_to_save_log': CONFIG_SUPERNET['train_settings']['path_to_save_log'],
            'exp_anneal_rate':CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] 
            }

    train_w_loader, train_thetas_loader = get_loaders(CONFIG_SUPERNET['dataloading']['w_share_in_train'],
                                                    CONFIG_SUPERNET['dataloading']['path_to_save_data'],CONFIG_SUPERNET['dataloading']['batch_size'],
                                                    CONFIG_SUPERNET['dataloading']['img_size'])
    test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['path_to_save_data'], 
                                  CONFIG_SUPERNET['dataloading']['test_batch_size'],
                                  CONFIG_SUPERNET['dataloading']['img_size'])
    trainer = TrainerSupernet(criterion, w_optimizer, theta_optimizer, w_scheduler, train_param, logger)
    trainer.train_loop(train_w_loader, train_thetas_loader, test_loader, model)


if __name__ == "__main__":
    manual_seed = CONFIG_SUPERNET['default_settings']['seed']
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(CONFIG_SUPERNET['dataloading']['path_to_save_logger'])
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger("TrainLogger")
    logger.addHandler(fh)

    logger.info('-----------------------------------START--------------------------------------------')
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info('device : mps')
    elif torch.cuda.is_available():
        torch.cuda.set_device(CONFIG_SUPERNET['default_settings']['gpu_ids'])
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        logger.info('gpu device = %d' % CONFIG_SUPERNET['default_settings']['gpu_ids'])
        device = torch.device('cuda')
    else:
        logger.info('no gpu device available')
        sys.exit(1)
    train_supernet(device, logger)
    logger.info('------------------------------------END------------------------------------------------')
