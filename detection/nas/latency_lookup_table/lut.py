'''
It is modified by Wonseon-Lim and based on Once for ALL Paper code.
Source: https://github.com/mit-han-lab/once-for-all
'''

import copy
import os
import sys
import time
from pathlib import Path


import numpy as np
import torch
import torch.nn as nn
import yaml
from ofa.utils import download_url
from ofa.utils.layers import *

# from utils.arch_utils import MyNetwork, make_divisible
# from utils.downloads import download_url


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def count_conv_flop(out_size, in_channels, out_channels, kernel_size, groups):
    '''
    compute conv flops
    '''
    out_h = out_w = out_size
    delta_ops = (
        in_channels * out_channels * kernel_size *
        kernel_size * out_h * out_w / groups
    )
    return delta_ops

def rm_bn_from_net(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.forward = lambda x: x


class LatencyEstimator:
    '''
    get Latency (it will be modified)
    '''

    def __init__(self,
                 local_dir="~/.hancai/latency_tools/",
                 url="https://hanlab.mit.edu/files/proxylessNAS/\
                    LatencyTools/mobile_trim.yaml",
                 ):
        # if url.startswith("http"):
        #     fname = download_url(url, local_dir, overwrite=True)
        # else:
        #     fname = ROOT / url
        fname = ROOT / 'mobile_lut.yaml'

        with open(fname, "r") as fp:
            self.lut = yaml.load(fp, Loader=yaml.FullLoader)

    @staticmethod
    def repr_shape(shape):
        '''
        get shape
        '''
        if isinstance(shape, (list, tuple)):
            return "x".join(str(_) for _ in shape)
        if isinstance(shape, str):
            return shape
        return TypeError

    def query(self,
              l_type: str,
              input_shape,
              output_shape,
              mid=None,
              ks=None,
              stride=None,
              id_skip=None,
              se=None,
              h_swish=None,
              ):
        '''
        get latency
        '''
        infos = [
            l_type,
            "input:%s" % self.repr_shape(input_shape),
            "output:%s" % self.repr_shape(output_shape),
        ]

        if l_type in ("expanded_conv",):
            assert None not in (mid, ks, stride,
                                id_skip, se, h_swish)
            infos += [
                "expand:%d" % mid,
                "kernel:%d" % ks,
                "stride:%d" % stride,
                "idskip:%d" % id_skip,
                "se:%d" % se,
                "hs:%d" % h_swish,
            ]
        key = "-".join(infos)
        return self.lut[key]["mean"]

    def predict_network_latency(self, net, image_size=224):
        '''
        get lut
        '''
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            "Conv",
            [image_size, image_size, 3],
            [(image_size + 1) // 2, (image_size + 1) //
             2, net.first_conv.out_channels],
        )
        # blocks
        fsize = (image_size + 1) // 2
        for block in net.blocks:
            mb_conv = block.mobile_inverted_conv
            shortcut = block.shortcut

            if mb_conv is None:
                continue
            if shortcut is None:
                idskip = 0
            else:
                idskip = 1
            out_fz = int((fsize - 1) / mb_conv.stride + 1)
            block_latency = self.query(
                "expanded_conv",
                [fsize, fsize, mb_conv.in_channels],
                [out_fz, out_fz, mb_conv.out_channels],
                mid=mb_conv.depth_conv.conv.in_channels,
                ks=mb_conv.kernel_size,
                stride=mb_conv.stride,
                id_skip=idskip,
                se=1 if mb_conv.use_se else 0,
                h_swish=1 if mb_conv.act_func == "h_swish" else 0,
            )
            predicted_latency += block_latency
            fsize = out_fz
        # final expand layer
        predicted_latency += self.query(
            "Conv_1",
            [fsize, fsize, net.final_expand_layer.in_channels],
            [fsize, fsize, net.final_expand_layer.out_channels],
        )
        # global average pooling
        predicted_latency += self.query(
            "AvgPool2D",
            [fsize, fsize, net.final_expand_layer.out_channels],
            [1, 1, net.final_expand_layer.out_channels],
        )
        # feature mix layer
        predicted_latency += self.query(
            "Conv_2",
            [1, 1, net.feature_mix_layer.in_channels],
            [1, 1, net.feature_mix_layer.out_channels],
        )
        # classifier
        predicted_latency += self.query(
            "Logits", [1, 1, net.classifier.in_features], [
                net.classifier.out_features]
        )
        return predicted_latency

    def predict_network_latency_given_spec(self, spec):
        '''
        predict latency
        '''
        imgsz = spec["r"][0]
        predicted_latency = 0
        # first conv
        predicted_latency += self.query(
            "Conv",
            [imgsz, imgsz, 3],
            [(imgsz + 1) // 2, (imgsz + 1) // 2, 24],
        )
        # blocks
        fsize = (imgsz + 1) // 2
        # first block
        predicted_latency += self.query(
            "expanded_conv",
            [fsize, fsize, 24],
            [fsize, fsize, 24],
            mid = 24,
            ks = 3,
            stride = 1,
            id_skip = 1,
            se = 0,
            h_swish = 0,
        )
        in_channel = 24
        stride_stages = [2, 2, 2, 1, 2]
        width_stages = [32, 48, 96, 136, 192]
        act_stages = ["relu", "relu", "h_swish", "h_swish", "h_swish"]
        se_stages = [False, True, False, True, True]
        for i in range(20):
            stage = i // 4
            depth_max = spec["d"][stage]
            if i % 4 + 1 > depth_max:
                continue
            ks, _e = spec["ks"][i], spec["e"][i]
            if i % 4 == 0:
                stride = stride_stages[stage]
                idskip = 0
            else:
                stride = 1
                idskip = 1
            out_channel = width_stages[stage]
            out_fz = int((fsize - 1) / stride + 1)

            mid_channel = round(in_channel * _e)
            block_latency = self.query(
                "expanded_conv",
                [fsize, fsize, in_channel],
                [out_fz, out_fz, out_channel],
                mid=mid_channel,
                ks=ks,
                stride=stride,
                id_skip=idskip,
                se=1 if se_stages[stage] else 0,
                h_swish=1 if act_stages[stage] == "h_swish" else 0,
            )
            predicted_latency += block_latency
            fsize = out_fz
            in_channel = out_channel
        # final expand layer
        predicted_latency += self.query(
            "Conv_1",
            [fsize, fsize, 192],
            [fsize, fsize, 1152],
        )
        # global average pooling
        predicted_latency += self.query(
            "AvgPool2D",
            [fsize, fsize, 1152],
            [1, 1, 1152],
        )
        # feature mix layer
        predicted_latency += self.query("Conv_2", [1, 1, 1152], [1, 1, 1536])
        # classifier
        predicted_latency += self.query("Logits", [1, 1, 1536], [1000])
        return predicted_latency


class LatencyTable:
    '''
    Latency Look-Up table
    '''
    # 160, 176, 192, 208

    def __init__(self, device="note10", resolutions=224):
        self.latency_tables = {}
        if not isinstance(resolutions, list):
            if isinstance(resolutions, int):
                resolutions = [resolutions]
            else:
                resolutions = list(resolutions)
        self.get_lut(device, resolutions)

    def get_lut(self, device, resolutions):
        '''
        get lut
        '''
        prefix = "https://hanlab.mit.edu/files/OnceForAll/tutorial/"
        for image_size in resolutions:
            self.latency_tables[image_size] = LatencyEstimator(
                url=prefix + \
                    "latency_table@%s/%d_lookup_table.yaml"
                % (device, image_size)
            )
            print("Built latency table for image size: %d." % image_size)

    def predict_efficiency(self, spec: dict):
        '''
        get latency
        '''
        return self.latency_tables[
            spec["r"][0]].predict_network_latency_given_spec(spec)


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
        # if load_efficiency_table is not None:
        #     self.efficiency_dict = np.load(
        #         load_efficiency_table, allow_pickle=True
        #     ).item()
        # else:
        #     self.build_lut(batch_size)
    
    @torch.no_grad()
    def measure_single_layer_latency(
        self, layer: nn.Module, input_size: tuple, warmup_steps=10, measure_steps=50
    ):
        total_time = 0
        inputs = torch.randn(*input_size, device=self.device)
        layer.eval()
        rm_bn_from_net(layer)
        network = layer.to(self.device)
        torch.cuda.synchronize()
        for i in range(warmup_steps):
            network(inputs)
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        st = time.time()
        for i in range(measure_steps):
            network(inputs)
        torch.cuda.synchronize()
        ed = time.time()
        total_time += ed - st

        latency = total_time / measure_steps * 1000

        return latency
    
    @torch.no_grad()
    def measure_single_layer_flops(self, layer: nn.Module, input_size: tuple):
        import thop

        inputs = torch.randn(*input_size, device=self.device)
        network = layer.to(self.device)
        layer.eval()
        rm_bn_from_net(layer)
        flops, params = thop.profile(network, (inputs,), verbose=False)
        return flops / 1e6
  
  
    
# if __name__ == "__main__":
    