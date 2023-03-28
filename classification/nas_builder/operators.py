import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return torch.max(torch.zeros_like(x), x)


class Conv2dSame(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation,
            groups, bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = _calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
        pad_w = _calc_same_pad(iw, kw, self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class PointConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super(PointConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

    def forward(self, x):
        return self.conv(x)

class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, num_groups=3, stride=1, padding=0):
        super(GroupedConv2d, self).__init__()

        self.num_groups = num_groups
        self.split_in_channels = _split_channels(in_channels, self.num_groups)
        self.split_out_channels = _split_channels(out_channels, self.num_groups)

        self.grouped_conv = nn.ModuleList()
        for i in range(self.num_groups):
            self.grouped_conv.append(nn.Conv2d(
                self.split_in_channels[i],
                self.split_out_channels[i],
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ))

    def forward(self, x):
        if self.num_groups == 1:
            return self.grouped_conv[0](x)

        x_split = torch.split(x, self.split_in_channels, dim=1)
        x = [conv(t) for conv, t in zip(self.grouped_conv, x_split)]
        x = torch.cat(x, dim=1)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_size = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_size, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MixedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilated=False, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.stride = stride
        if depthwise:
            conv_groups = out_splits
        else:
            groups = kwargs.pop('groups', 1)
            if groups > 1:
                conv_groups = _split_channels(groups, num_groups)
            else:
                conv_groups = [1] * num_groups

        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            d = 1
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            self.add_module(
                str(idx),
                Conv2dSame(
                    in_ch, out_ch, k, stride,**kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
        x = torch.cat(x_out, 1)
        return x


#  class ModuleBlock(nn.Module):
    #  def __init__(self, in_planes, out_planes, kernel_size, stride, num_groups, expansion_ratio, module_name):
        #  super(ModuleBlock, self).__init__()
        #  conv_list = []
        #  exp_planes = int(in_planes * expansion_ratio)
        #  self.num_grups = num_groups
        #  self.stride = stride
        #  self.module_name = module_name
        #  self.shortcut = (stride == 1 and in_planes == out_planes)
        #  exp_conv = nn.Sequential(
            #  GroupedConv2d(in_planes, exp_planes, kernel_size=1, num_groups=num_groups),
            #  nn.BatchNorm2d(exp_planes),
            #  Swish())
        #  conv_list.append(exp_conv)
            
        #  if module_name == 'Conv':
            #  d_conv = Conv2dSame(exp_planes, exp_planes, kernel_size, stride=stride, groups=exp_planes) 
        #  elif module_name == 'Mix':
            #  d_conv = MixedConv2d(exp_planes, exp_planes, kernel_size, stride=stride)

        #  mid_conv = nn.Sequential(
            #  d_conv,
            #  nn.BatchNorm2d(exp_planes),
            #  Swish()
        #  )
        #  conv_list.append(mid_conv)

        #  seq_conv = nn.Sequential(
            #  GroupedConv2d(exp_planes, out_planes, kernel_size=1, num_groups=num_groups),
            #  nn.BatchNorm2d(out_planes)
            #  )
        #  conv_list.append(seq_conv)
        #  self.conv = nn.Sequential(*conv_list)

    #  def forward(self, x):
        #  if self.shortcut:
            #  shortcut = x
        #  x = self.conv(x)
        #  if self.shortcut:
            #  if self.stride == 2:
                #  shortcut = F.adaptive_avg_pool2d(shortcut, (x.shape[2], x.shape[3]))
            #  x +=shortcut
        #  x = channel_shuffle(x, self.num_grups)
        #  return x


#  def make_oplist(layer_parameters, group_size, reduced=True):
    #  (in_planes, out_planes, k1, k2, stride, expansion_ratio) = layer_parameters
    #  if reduced:
        #  expansion_ratio = 0.5
    #  kernel_size = {operator_list[0] : k1, operator_list[1] : k2}
    #  op_list = nn.ModuleList()
    #  for i in group_size:
        #  for op in operator_list:
            #  op_list.append(ModuleBlock(in_planes, out_planes, kernel_size[op], stride, i, expansion_ratio, op))

    #  return op_list
