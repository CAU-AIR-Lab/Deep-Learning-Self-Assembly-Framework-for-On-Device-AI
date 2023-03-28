# Auto Compression

## Run
```bash
python -m pip install -U pip
pip install -r requirements.txt
pip install protobuf==3.20.1
pip install --upgrade tensorflow-probability==0.17.0
```

> Pytorch and Tensorflow are also required

## Example
```python
from ondevice_ai.compression import *

## load the final network by NAS
net.eval()
compressed_model = deepComp(net, image_size, pruning_rate=0.3, accelerator=True)
## param accelerator -> bool:
## True if there exists the accelerator of the target device such as mobile GPU, NPU, etc.

from torch.cuda.amp import autocast

x = torch.rand((1,3,224,224))
with autocast():
    compressed_model(x)

onnx_path = "saves/model.onnx"
tf_path = "saves/model_tf"
tflite_path = "saves/model.tflite"

torch_to_onnx(compressed_model, x.half().cuda(), onnx_path)
onnx_to_tflite(onnx_path, tf_path, tflite_path)
```

# Android App

> It is implemented using tensorflow lite demo App. 
Every time the battery usage falls below 50% and 10%, the reduced models runs.