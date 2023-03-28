#

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

from nas.enas import ENAS
from nas.eval import fine_tune
from utils.datasets import load_dataset
from models.model import load_models
from yolo5_utils.accelerate import check_amp, check_train_batch_size
from yolo5_utils.general import (check_dataset, check_img_size, print_args,
                            labels_to_class_weights)
from yolo5_utils.torch_utils import select_device, torch_distributed_zero_first


# 현재 파일의 directory 경로
BASEPATH = os.path.dirname(os.path.abspath(__file__))
# 기본 경로에 BASEPATH 추가하여 해당 파일 import 가능
if str(BASEPATH) not in sys.path:
    sys.path.append(str(BASEPATH))

# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', -1)) # LOCAL_RANK 존재하면 get, 존재하지 않으면 -1 반환
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_option():
    parser = argparse.ArgumentParser()
    # General Args[device, data, hyperparameters]
    parser.add_argument('--weights', type=str, default=ROOT / 'models/yolov5s.pt', help='initial weights path')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--data', type=str, default=ROOT / 'data/scripts/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/search_hyps.yaml', help='hyperparameters path')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # -- Training Args -- #
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # -- save and log -- #
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')

    # Backbone Args
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')

    opt = parser.parse_args()
    return opt

opt = parse_option()

# def search_architecture(opt, callbacks=Callbacks()):
def search_architecture(opt):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_requirements()
    
    # Resume (from specified or most recent last.pt)
    # if opt.resume:
    #     last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
    #     opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
    #     opt_data = opt.data  # original dataset
    #     if opt_yaml.is_file():
    #         with open(opt_yaml, errors='ignore') as f:
    #             d = yaml.safe_load(f)
    #     else:
    #         d = torch.load(last, map_location='cpu')['opt']
    #     opt = argparse.Namespace(**d)  # replace
    #     opt.weights, opt.resume = str(last), True  # reinstate
    #     if is_url(opt_data):
    #         opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    # else:
    #     opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
    #         check_file(opt.data), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
    #     opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    
    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')
    
    data_dict = None
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(opt.data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])  # the number of classes
    names = data_dict['names']
    assert len(
        names) == nc, f'{len(names)} names found \
            for nc={nc} dataset in {opt.data}'  # check
            
    # Hyperparameters
    if opt.hyp:
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    
    # Model
    # Backbone SuperNet
    base_model_weights = 'models/yolov5s.pt'
    base_model, supernet = load_models(opt.weights, nc, device)
    stride = base_model.stride
    
    # Image size
    gs = max(int(stride.max()), 32)  # grid size (max stride)
    # verify imgsz is gs-multiple
    imgsz = check_img_size(224, gs, floor=gs * 2)
    
    # Batch size
    # DDP mode TODO
    if opt.batch_size == -1:  # single-GPU only, estimate best batch size
        amp = check_amp(base_model)  # check AMP
        opt.batch_size = check_train_batch_size(base_model, imgsz, amp)
    
    print("auto batch size: ", opt.batch_size)

    train_loader, dataset = load_dataset(train_path,
                                        imgsz,
                                        opt.batch_size // WORLD_SIZE,
                                        gs,
                                        cache="ram", # or disk
                                        rect=False,
                                        rank=LOCAL_RANK,
                                        workers=opt.workers,
                                        image_weights=True,
                                        quad=True,
                                        prefix='train: ',
                                        shuffle=True)

    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {opt.data}. \
        Possible class labels are 0-{nc - 1}'

    val_loader = load_dataset(val_path,
                            imgsz,
                            opt.batch_size // WORLD_SIZE * 2,
                            gs,
                            cache="ram",
                            rect=True,
                            rank=-1,
                            workers=opt.workers*2,
                            pad=0.5,
                            prefix='val: ')[0]

    # yolov5
    base_model = base_model.model[10:]
    base_model.class_weights = labels_to_class_weights(
        dataset.labels, nc).to(device) * nc
    base_model.nc = nc
    base_model.names = names
    base_model.stride = stride

    # ENAS ######  --> final net
    enas = ENAS(hyp,
                val_loader, 
                base_model, 
                supernet, 
                device, 
                nc, 
                names
                )

    _, best_net = enas.run_evolution_search()

    amp = check_amp(best_net, final=True)  # check AMP
    best_net = fine_tune(train_loader, best_net, amp)
    best_net.eval()
    
    return best_net

    
if __name__ == "__main__":
    best_net = search_architecture(opt)
    print(best_net)