import torch
from torch import nn
import sys
import os
import time
import logging
from utils import weights_init, accuracy
import torch.nn.functional as F
from torchsummary import summary
from utils import AverageMeter, weights_init, get_module_list
from thop import profile
from nas_builder.config_cifar10 import CONFIG_SAMPLE, CONFIG_LAYER, CONFIG_SUPERNET
from nas_builder.model import SuperNet
from nas_builder.dataloaders_cifar10 import get_loaders, get_test_loader
import copy

SEED = 100

def sample_architecture_from_the_supernet():
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(CONFIG_SAMPLE['dataloading']['path_to_save_logger'])
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger("SamplingLogger")
    logger.addHandler(fh)

    logger.info('SAMPLING---------------------------START--------------------------------------------')
    if not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info('device : mps')
    elif torch.cuda.is_available():
        torch.cuda.set_device(CONFIG_SUPERNET['default_settings']['gpu_ids'])
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.enabled = True
        logger.info('gpu device = %d' % CONFIG_SUPERNET['default_settings']['gpu_ids'])
        device = torch.device('cuda')
    else:
        logger.info('no gpu device available')
        sys.exit(1)

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
    model.to(device)
    model.load_state_dict(torch.load(CONFIG_SAMPLE['dataloading']['path_to_load_model']))
    layer_list = []
    layer_list.append(model.first)
    for layer in model.stages_to_search:
        thetas = layer.thetas
        idx = torch.argmax(thetas).item()
        layer_list.append(layer.ops[idx])
    layer_list.append(model.last)
    sampled_model = nn.Sequential(*layer_list)
    if CONFIG_SAMPLE['train_settings']['train_from_scratch']:
        logger.info('Train from scratch')
        sampled_model = sampled_model.apply(weights_init)
    sampled_model.to(device)
    
    train_loader = get_loaders(1,
                               CONFIG_SUPERNET['dataloading']['path_to_save_data'],
                               CONFIG_SUPERNET['dataloading']['batch_size'],
                               CONFIG_SUPERNET['dataloading']['img_size'])
    test_loader = get_test_loader(CONFIG_SUPERNET['dataloading']['path_to_save_data'], 
                                  CONFIG_SUPERNET['dataloading']['test_batch_size'],
                                  CONFIG_SUPERNET['dataloading']['img_size'])

    num_epoch = CONFIG_SAMPLE['train_settings']['cnt_epochs']
    if num_epoch != 0:
        optimizer = torch.optim.SGD(params=sampled_model.parameters(), 
                                      lr=CONFIG_SAMPLE['optimizer']['w_lr'],
                                      momentum=CONFIG_SAMPLE['optimizer']['w_momentum'],
                                      weight_decay=CONFIG_SAMPLE['optimizer']['w_weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
        criterion = nn.CrossEntropyLoss().cuda()
        best_top1 = 0.0                                                        

        for epoch in range(num_epoch):
            # train
            tr_loss, tr_top1, tr_top5 = train_step(sampled_model, train_loader, criterion, optimizer, logger, epoch, num_epoch)
            
            # test
            ts_loss, ts_top1, ts_top5 = test_step(sampled_model, test_loader, criterion, logger, epoch, num_epoch)
            scheduler.step()

            if best_top1 < ts_top1:
                best_top1 = ts_top1 
                torch.save(sampled_model.state_dict(), './sampled_{epoch}.pth'.format(epoch=epoch))
                best_model = copy.deepcopy(sampled_model)

    # Measure model size & params size
    img_size = CONFIG_SUPERNET['dataloading']['img_size']
    summary(sampled_model, input_size=(3, img_size, img_size), batch_size=1, device='cuda')
    logger.info("Best Top1 : {}".format(best_top1))
    inp = torch.rand(1, 3, img_size, img_size).to('cuda')
    macs, params = profile(sampled_model, inputs=(inp, ))    
    
    logger.info("FLOPS: {flops}, PARAMS: {params}".format(
        flops=macs,
        params=params
    ))

    # Data load for measuring inference time
    # Gpu Inference Time
    inference_time = AverageMeter()

    for i in range(1000):
        batch = torch.rand(1, 3, img_size, img_size).to('cuda')
        
        start = time.time()
        sampled_model(batch)
        inference_time.update(time.time() - start)
        
    print('Average {device} inference time : {time:.3f}'.format(device=batch.device.type, time=inference_time.avg))
            

    # Cpu Inference Time
    inference_time = AverageMeter()
    sampled_model = sampled_model.to('cpu')
    for i in range(1000):
        batch = torch.rand(1, 3, img_size, img_size)
        start = time.time()
        sampled_model(batch)
        inference_time.update(time.time() - start)
        
    print('Average {device} inference time : {time:.3f}'.format(device=batch.device.type, time=inference_time.avg))
    best_model = best_model.to('cpu')
    traced_model = torch.jit.trace(best_model, torch.randn(1, 3, 224, 224))
    torch.jit.save(traced_model, CONFIG_SAMPLE['dataloading']['path_to_save_jit'])
    #loaded_model = torch.jit.load('your_model.pt')
    logger.info('SAMPLING----------------------------END---------------------------------------------')


def train_step(model, loader, criterion, optimizer, logger, epoch, num_epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()

    for step, (X, y) in enumerate(loader):
        X, y = X.cuda(), y.cuda()
        N = X.shape[0]

        outs = model(X)
        loss = criterion(outs, y)

        prec1, prec3 = accuracy(outs, y, topk=(1, 5))
        losses.update(loss.item(), X.size(0))
        top1.update(prec1.item(), X.size(0))
        top5.update(prec3.item(), X.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info('Epoch (Train):[{epoch}/{num_epoch}] | Train Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'.format(
        epoch=epoch+1,
        num_epoch=num_epoch,
        loss=losses.avg,
        top1=top1.avg,
        top5=top5.avg))

    return losses.avg, top1.avg, top5.avg

def test_step(model, loader, criterion, logger, epoch, num_epoch):
    # Measure Test Accuracy
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        model.eval()

        for step, (X, y) in enumerate(loader):
            X, y = X.cuda(), y.cuda()
            N = X.shape[0]

            outs = model(X)
            loss = criterion(outs, y)
            prec1, _ = accuracy(outs, y, topk=(1, 5))
            prec1, prec3 = accuracy(outs, y, topk=(1, 5))
            losses.update(loss.item(), X.size(0))
            top1.update(prec1.item(), X.size(0))
            top5.update(prec3.item(), X.size(0))

    logger.info('Epoch (Test):[{epoch}/{num_epoch}] | Test Loss: {loss:.4f} | top1: {top1:.4f} | top5: {top5:.4f}'.format(
        epoch=epoch+1,
        num_epoch=num_epoch,
        loss=losses.avg,
        top1=top1.avg,
        top5=top5.avg))

    return losses.avg, top1.avg, top5.avg


if __name__ == "__main__":
    sample_architecture_from_the_supernet()
