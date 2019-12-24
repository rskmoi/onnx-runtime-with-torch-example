# onnx-runtime-with-torch-example
Sample of serving pytorch model with onnxruntime.

## Version
```
torch==1.3.1
onnxruntime==1.0.0
onnx==1.6.0
```

## Usage
```
python 01_train_model_with_torch.py 
python 02_export_model.py
python 03_onnxruntime_local.py

< run onnx runtime server >

python 04_onnxruntime_server_json.py
python 05_onnxruntime_server_binary.py
python 06_onnxruntime_server_grpc.py
```

## How to run onnx runtime server container
- [Official Dockerfile](https://github.com/microsoft/onnxruntime/blob/master/dockerfiles/Dockerfile.server)
- [Build & Run](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#onnx-runtime-server)

## onnx_model_cli
onnx_model_cli helps checking onnx model structure.

Inspired by tensorflow [saved_model_cli](https://www.tensorflow.org/guide/saved_model#details_of_the_savedmodel_command_line_interface).

### Usage
```
python onnx_model_cli.py show --path foo.onnx
```

## Great Links
- [onnxruntime official](https://github.com/microsoft/onnxruntime)
- [pytorch document](https://pytorch.org/docs/stable/onnx.html)
- [機械学習モデルのServingとONNX Runtime Serverについて](https://qiita.com/lain21/items/4d68ee30b7fd497453d4)