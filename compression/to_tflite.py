import torch
import torch.nn
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf

def torch_to_onnx(model, sample, save_path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    torch.onnx.export(
        model,
        sample, 
        save_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        opset_version=12
    )

def onnx_to_tflite(onnx_path, tf_path, tflite_path):
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model, auto_cast=True)
    tf_rep.export_graph(tf_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.experimental_new_converter = True
    converter.allow_custom_ops = True
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)