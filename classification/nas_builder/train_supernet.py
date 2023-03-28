import torch
from torch.autograd import Variable
from utils import AverageMeter, accuracy
import time
import logging
import os
import sys
import numpy as np

class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, train_param, logger):
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.lat = AverageMeter()
        self.losses = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce = AverageMeter()
        self.logger = logger

        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler

        self.temperature = train_param['temperature']
        self.cnt_epochs = train_param['cnt_epochs']
        self.exp_anneal_rate = train_param['exp_anneal_rate']
        self.train_thetas_from_the_epoch = train_param['train_thetas_from_the_epoch']
        self.path_to_save_model = train_param['path_to_save_model']
        self.print_freq = 50
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
            format=log_format, datefmt='%m/%d %I:%M:%S %p')

    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):
        
        best_top1 = 0.0
        # firstly, train weights only
        start_time = time.time()

        for epoch in range(self.train_thetas_from_the_epoch):
            loss_val, ce_val, top1_val, lat_val = self._training_step(model, train_w_loader, self.w_optimizer, epoch)
            self.w_scheduler.step()
            self.logger.info('Epoch[{cur_epoch}/{max_epoch}] Pretrain Train: Loss: {loss:.4f}, CE: {ce:.4f}, lat: {lat:.4f}, acc: {top1:.4f}'.format(
                cur_epoch=epoch,
                max_epoch=self.cnt_epochs,
                loss=loss_val,
                ce=ce_val,
                lat=lat_val,
                top1=top1_val
            ))
        self.logger.info("pretrain finished")

        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            loss_val_w, ce_val_w, top1_val_w, lat_val_w = self._training_step(model, train_w_loader, self.w_optimizer, epoch)
            self.logger.info('Epoch[{cur_epoch}/{max_epoch}] Weight Train: Loss: {loss:.4f}, CE: {ce:.4f}, lat: {lat:.4f}, acc: {top1:.4f}'.format(
                cur_epoch=epoch,
                max_epoch=self.cnt_epochs,
                loss=loss_val_w,
                ce=ce_val_w,
                lat=lat_val_w,
                top1=lat_val_w                
            ))
            loss_val_t, ce_val_t, top1_val_t, lat_val_t = self._training_step(model, train_thetas_loader, self.theta_optimizer, epoch)
            self.w_scheduler.step()
            self.logger.info('Epoch[{cur_epoch}/{max_epoch}] Thetas Train: Loss: {loss:.4f}, CE: {ce:.4f}, lat: {lat:.4f}, acc: {top1:.4f}'.format(
                cur_epoch=epoch,
                max_epoch=self.cnt_epochs,
                loss=loss_val_t,
                ce=ce_val_t,
                lat=lat_val_t,
                top1=top1_val_t                
            ))
            top1_val, loss_val = self._validate(model, test_loader, epoch)
            self.logger.info('acc: {top1:.4f} Loss: {loss:.4f}'.format(
                top1=top1_val,
                loss=loss_val
            ))
            if best_top1 < top1_val:
                best_top1 = top1_val
                torch.save(model.state_dict(), self.path_to_save_model)
            self.temperature = self.temperature * self.exp_anneal_rate
        self.logger.info('Time:'+str(time.time() - start_time))
        self.logger.info('Best Top1: ' + str(best_top1))


    def _training_step(self, model, loader, optimizer, epoch):
        model = model.train()
        loss_list = AverageMeter()
        ce_list = AverageMeter()
        lat_list = AverageMeter()
        for step, (X, y) in enumerate(loader):

            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            N = X.shape[0]

            optimizer.zero_grad()
            outs, lats_to_accumulate = model(X, self.temperature)
            self.lat.update(lats_to_accumulate.item(), N)
            loss, ce, lat = self.criterion(outs, y, lats_to_accumulate)
            loss_list.update(loss.item(), X.size(0))
            ce_list.update(ce.item(), X.size(0))
            lat_list.update(lat.item(), X.size(0))
            loss.backward()
            optimizer.step()
            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")
        top1_avg = self.top1.get_avg()
        lat_avg = self.lat.get_avg()
        for avg in [self.top1, self.top5, self.losses, self.lat]:
            avg.reset()
        return loss_list.get_avg(), ce_list.get_avg(), top1_avg, lat_avg

    def _validate(self, model, loader, epoch):
        model.eval()
        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]

                outs, lats_to_accumulate = model(X, self.temperature)
                loss, _, _ = self.criterion(outs, y, lats_to_accumulate)
                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")

        top1_avg = self.top1.get_avg()
        loss_avg = self.losses.get_avg()
        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()
        return top1_avg, loss_avg

    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
            prec1, prec5 = accuracy(outs, y, topk=(1, 5))
            self.losses.update(loss.item(), N)
            self.top1.update(prec1.item(), N)
            self.top5.update(prec5.item(), N)
            
            if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
                self.logger.info(val_or_train+
                   ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
                   "Prec@(1,5) ({:.1%}, {:.1%})".format(
                       epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
                       self.top1.get_avg(), self.top5.get_avg()))
