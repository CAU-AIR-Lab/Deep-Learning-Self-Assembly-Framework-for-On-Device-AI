# ondevice_ai


## Installation

### Dependencies

Detection NAS requires **Python 3.6+**.

- pytorch>=1.12.0
- torchvision>=0.13.0

## How to use
```bash
python search_run.py --data data/scripts/coco128.yaml \
                     --hyp data/hyps/search_hyps.yaml \
                     --weights models/yolov5s.pt \
                     --batch-size 16
```

## Measuring block latency Example
* Weight Sharing Module belongs to the supernet code.

```bach
from supernet.weight_sharing_module import *
from nas.latency_lookup_table.lut import FLOPsTable

# batch, in_channel, input_size, input_size
input_size = (1, 3, 128, 128)
layer = DynamicMBConvLayer(
                           in_channel_list=[3],
                           out_channel_list=[32],
                           kernel_size_list=[3,5,7],
                           )

table = FLOPsTable("latency")
latency = table.measure_single_layer_latency(
                                             layer=layer, 
                                             input_size=input_size,
                                             warmup_steps=10,
                                             measure_steps=50
                                            )
flops = table.measure_single_layer_flops(
                                         layer=layer, 
                                         input_size=input_size
                                        )
print("50 repeats average - Latency : {:0.5f}, Flops : {:0.3f}".format(latency, flops))
```