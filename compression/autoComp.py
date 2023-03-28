import copy
import torch
import torch.nn
import torch.nn.utils.prune
from torch.cuda.amp import autocast
from .utils import pt_to_arch, get_module_by_name



def deepComp(model, 
             image_size, 
             pruning_rate=0.1, 
             quantize_rate=0.1, 
             accelerator=False):
    '''
    param model -> class:
        the final network by NAS
    param accelerator -> bool:
        True if there exists the accelerator of the target device such as mobile GPU, NPU, etc.
    '''
    net = copy.deepcopy(model)
    arch, prev_params = pt_to_arch(net, image_size, batch_size=1, print_ok=False)

    if accelerator == False:
        # Unstructured Pruning
        for layer in arch:
            # input_shape, output_shape, trainable, nb_params
            module_name = arch[layer]["module_name"]
            module_type = module_name.split(".")[-1]
            if module_type in ["conv", "linear"] and arch[layer]["nb_params"] > 0:
                torch.nn.utils.prune.l1_unstructured(get_module_by_name(net, module_name), 
                                                     name="weight", 
                                                     amount=pruning_rate, 
                                                     importance_scores=None)
                torch.nn.utils.prune.remove(get_module_by_name(net, module_name), "weight")
    else:
        # Structured Pruning
        for layer in arch:
            # input_shape, output_shape, trainable, nb_params
            module_name = arch[layer]["module_name"]
            module_type = module_name.split(".")[-1]
            if module_type == "linear" and arch[layer]["nb_params"] > 0:
                torch.nn.utils.prune.l1_unstructured(get_module_by_name(net, module_name), 
                                                     name="weight", 
                                                     amount=pruning_rate, 
                                                     importance_scores=None)
            elif module_type == "conv" and arch[layer]["nb_params"] > 0:
                torch.nn.utils.prune.ln_structured(get_module_by_name(net, module_name), 
                                                   name="weight", 
                                                   amount=pruning_rate, 
                                                   n=2, 
                                                   dim=0) # channel selection if dim=0
                                                          # filter selection if dim=1
            else:
                continue
            torch.nn.utils.prune.remove(get_module_by_name(net, module_name), "weight")
    
    arch, pruned_params = pt_to_arch(net, image_size, batch_size=1, print_ok=False)

    model_pruning_rate = (1 - pruned_params/prev_params) * 100 
    print("Compression Rate by Pruning: {:0.1f}%".format(model_pruning_rate))
    net.half()
    print("Compression Rate by Quantization: {:0.1f}%".format(100-(100 - model_pruning_rate)/2))

    return net

if __name__ == "__main__":
    ## load the final network by NAS
    ofa_specialized_get = torch.hub.load('mit-han-lab/once-for-all', "ofa_specialized_get")
    net, image_size = ofa_specialized_get("flops@595M_top1@80.0_finetune@75", pretrained=True)
    net = torch.nn.DataParallel(net).cuda()
    net.eval()
    image_size = (3, image_size, image_size)

    ## compression
    #compressed_model = deepComp(net, image_size, pruning_rate=0.3)
    compressed_model = deepComp(net, image_size, pruning_rate=0.3, accelerator=True)

    ## Use amp
    # inference example
    x = torch.rand((1,3,224,224))
    with autocast():
        compressed_model(x)

 
