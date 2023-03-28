import torch
import torch.nn as nn
from nas_builder.operators import PointConv2d, GroupedConv2d, MixedConv2d, SELayer, Swish, ReLU, Conv2dSame

def module_builder(module_str, in_channels, out_channels, stride=1):
    module_str_list = module_str.split('/')
    module_list = []
    
    if len(module_str_list) < 4:
        # error event
        error_str = 'module_str is not correct'
        raise ValueError(error_str)

    expansion_ratio = int(module_str_list[-1])
    exp_planes = int(in_channels * expansion_ratio)

    act_str = module_str_list[-2]

    if act_str == 'S':
        act = Swish()
    elif act_str == 'R':
        act = ReLU()

    first_str_list = module_str_list[0].split('_')
    
    if first_str_list[0] == 'P':
        f_conv = PointConv2d(in_channels, exp_planes)
        t_conv = PointConv2d(exp_planes, out_channels)
    elif first_str_list[0] == 'G':
        group_size = int(first_str_list[1])
        f_conv = GroupedConv2d(in_channels, exp_planes, num_groups=group_size)
        t_conv = GroupedConv2d(exp_planes, out_channels, num_groups=group_size)

    module_list.append(f_conv)
    module_list.append(act)
    second_str_list = module_str_list[1].split('_')
    if second_str_list[0] == 'M':
        kernel_list = []
        for i in range(len(second_str_list) - 1):
            kernel_list.append(int(second_str_list[i+1]))
        s_conv = MixedConv2d(exp_planes, exp_planes, kernel_size=kernel_list, stride=stride)
    elif second_str_list[0] == 'D':
        kernel = int(second_str_list[1])
        s_conv = Conv2dSame(exp_planes, exp_planes, kernel_size=kernel, stride=stride, groups=exp_planes)
    
    module_list.append(s_conv)
    module_list.append(act)
    if len(module_str_list) == 5:
        se_str_list = module_str_list[2].split('_')
        if se_str_list[0] == 'yes':
            se_ratio = int(se_str_list[1])
            se = SELayer(exp_planes, se_ratio)
            module_list.append(se)

    module_list.append(t_conv)
    
    if stride == 1 and in_channels == out_channels:
        shortcut = True
    else:
        shortcut = False

    return nn.Sequential(*module_list), shortcut

    
        
class BlockUnit(nn.Module):
    def __init__(self, module_str, in_channels, out_channels, stride):
        super(BlockUnit, self).__init__()
        self.module, self.shortcut = module_builder(module_str, in_channels, out_channels, stride)
        self.out_channels = out_channels
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.shortcut:
            return x + self.bn(self.module(x))
        else:
            return self.bn(self.module(x))
