'''
It is modified by Wonseon-Lim, Wangduk Seo and based on Once for ALL Paper code.
Source: https://github.com/mit-han-lab/once-for-all
'''

import copy
import os
import sys
import time
import torch
import torch.nn as nn


def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


class FLOPsTable:
    def __init__(
        self,
        pred_type="flops",
        device="cuda:0",
        multiplier=1.2,
        batch_size=64,
        load_efficiency_table=None,
    ):
        assert pred_type in ["flops", "latency"]
        self.multiplier = multiplier
        self.pred_type = pred_type
        self.device = device
        self.batch_size = batch_size
        self.efficiency_dict = {}

    @torch.no_grad()
    def measure_single_layer_latency(
        self, layer: nn.Module, input_size: tuple, warmup_steps=10, measure_steps=50
    ):
        if not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                synchronize = torch.mps.synchronize
        elif torch.cuda.is_available():
            synchronize = torch.cuda.synchronize
        else:
            error = 'no gpu device available'
            raise ValueError(error)

        total_time = 0
        inputs = torch.randn(*input_size, device=self.device)
        layer.eval()
        rm_bn_from_net(layer)
        network = layer.to(self.device)
        synchronize()
        for i in range(warmup_steps):
            network(inputs)
        synchronize()

        synchronize()
        st = time.time()
        for i in range(measure_steps):
            network(inputs)
        synchronize()
        ed = time.time()
        total_time += ed - st

        latency = total_time / measure_steps * 1000

        return latency

# if __name__ == "__main__":
