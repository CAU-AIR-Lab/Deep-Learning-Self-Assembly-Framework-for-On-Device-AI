import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from nas_builder.operators import Flatten
from nas_builder.block import BlockUnit
from nas_builder.lut import FLOPsTable


class FirstUnit(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(FirstUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=3, stride=stride, padding=1)
        self.norm = nn.BatchNorm2d(out_planes)
        self.act = nn.ReLU(inplace=True) 

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class LastUnit(nn.Module):
    def __init__(self, in_planes, last_feature_size, cnt_classes):
        super(LastUnit, self).__init__()
        self.lastconv = nn.Conv2d(in_channels=in_planes,
                              out_channels=last_feature_size,
                              kernel_size=1, stride=1)
        self.flatten = Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_feature_size, out_features=cnt_classes)
        )
    def forward(self, x):
        out = self.lastconv(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = self.flatten(out)
        out = self.fc(out)
        return out


class MixedOp(nn.Module):
    def __init__(self, module_list, layer_parameters, stage, flops_table):
        super(MixedOp, self).__init__()
        self.stage = stage
        self.flops_table = flops_table
        self.ops = nn.ModuleList()
        self.lat_list = []
        for module_str in module_list:
            block, lat = self.get_block(module_str, *layer_parameters)
            self.ops.append(block)
            self.lat_list.append(lat)
        self.thetas = nn.Parameter(torch.Tensor([1.0 / float(len(module_list)) for i in range(len(module_list))]))

    def forward(self, x, temperature, lats_to_accumulate):
        soft_mask_variables = nn.functional.gumbel_softmax(self.thetas, temperature)

        output = sum(m * op(x) for m, op in zip(soft_mask_variables, self.ops))
        num_of_lats = sum(m * lat for m, lat in zip(soft_mask_variables, self.lat_list))
        lats_to_accumulate += num_of_lats
        return output, lats_to_accumulate

    def get_block(self, module_str, *layer_parameters):
        
        block = BlockUnit(module_str, *layer_parameters) 
        input_size = (1, layer_parameters[0], 128, 128)
        lat = self.flops_table.measure_single_layer_latency(
                                             layer=block, 
                                             input_size=input_size,
                                             warmup_steps=10,
                                             measure_steps=50
                                            )
        return block, lat


class SuperNet(nn.Module):
    def __init__(self, supernet_param, device):
        super(SuperNet, self).__init__()
        self.layer_table = supernet_param['config_layer']
        self.module_list = supernet_param['module_list']
        self.first_inchannel = supernet_param['first_inchannel']
        self.first_stride = supernet_param['first_stride']
        self.last_feature_size = supernet_param['last_feature_size']
        self.cnt_classes = supernet_param['cnt_classes']
        self.device = device

        self.flops_table = FLOPsTable("latency", self.device)
        self.first = FirstUnit(self.first_inchannel, self.layer_table[0][0], self.first_stride)
        self.stages_to_search = nn.ModuleList()
        for idx, layer in enumerate(self.layer_table):
            self.stages_to_search.append(MixedOp(self.module_list, layer, idx, self.flops_table))
        self.last = LastUnit(self.layer_table[-1][1], self.last_feature_size, self.cnt_classes)

    def forward(self, x, temperature):
        lats_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).to(self.device)
        out = self.first(x)
        for mixed_op in self.stages_to_search:
            out, lats_to_accumulate = mixed_op(out, temperature, lats_to_accumulate)
        out = self.last(out)
        return out, lats_to_accumulate


class SuperNetLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(SuperNetLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.weight_criterion = nn.CrossEntropyLoss()

    def forward(self, outs, targets, lats):
        ce = self.weight_criterion(outs, targets)
        lats = lats ** self.beta
        loss = self.alpha * ce * lats
        return loss, ce, lats
