import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from functools import reduce
from  torch.cuda.amp import autocast

def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)

## A modified version of https://github.com/sksq96/pytorch-summary
def pt_to_arch(model, input_size, batch_size=-1, device="cuda", print_ok=True):

    def register_hook(module):

        def hook(module, input, output):
            module_idx = len(summary)
            m_key = str(module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["module_name"] = str(module.__class__).split("'")[1]
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                # params += torch.prod(torch.LongTensor(list(module.weight.size())))
                params += torch.count_nonzero(module.weight).cpu()
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    module_name_dict = {}
    for name,module in model.named_modules():
        module_name_dict[str(module.__class__).split("'")[1]] = name

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    with autocast():
        model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    if print_ok == True:
        print("----------------------------------------------------------------"*2)
        line_new = "{:>3} {:>70}  {:>25} {:>15}".format("Index","Layer (type)","Output Shape", "Param #")
        print(line_new)
        print("================================================================"*2)
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>3} {:>70}  {:>25} {:>15}".format(
            layer,
            summary[layer]["module_name"],
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        summary[layer]["module_name"] = module_name_dict[summary[layer]["module_name"]]
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if print_ok == True:
            print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    if print_ok == True:
        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")
    return summary, total_params_size
