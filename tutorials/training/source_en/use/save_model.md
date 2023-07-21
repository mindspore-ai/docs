# Saving Models

`Linux` `Ascend` `GPU` `CPU` `Model Export` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/use/save_model.md)

## Overview

During model training, you can add CheckPoints to save model parameters for inference and retraining after interruption. If you want to do inference on different hardware platforms, you need to generate corresponding models based on the network and CheckPoint, such as MINDIR, AIR, ONNX.

- MINDIR: MindSpore IR (MindIR) is a function-style IR based on graph representation. Its core purpose is to serve automatic differential transformation. It can be used on device inference now.
- CheckPoint: A CheckPoint file of MindSpore is a binary file that stores the values of all training parameters. The Google Protocol Buffers mechanism with good scalability is adopted, which is independent of the development language and platform.
The protocol format of CheckPoints is defined in `mindspore/ccsrc/utils/checkpoint.proto`.
- AIR: Ascend Intermediate Representation (AIR) is an open file format defined by Huawei for machine learning and can better adapt to the Ascend AI processor. It is similar to ONNX.
- ONNX: Open Neural Network Exchange (ONNX) is an open file format designed for machine learning. It is used to store trained models.

The following uses examples to describe how to save MindSpore CheckPoint files, and how to export MINDIR, AIR, ONNX files.

## Saving CheckPoint files

During model training, use the callback mechanism to transfer the object of the callback function `ModelCheckpoint` to save model parameters and generate CheckPoint files.

You can use the `CheckpointConfig` object to set the CheckPoint saving policies. The saved parameters are classified into network parameters and optimizer parameters.

`ModelCheckpoint` provides default configuration policies for users to quickly get started. The following describes the usage:

```python
from mindspore.train.callback import ModelCheckpoint
ckpoint_cb = ModelCheckpoint()
model.train(epoch_num, dataset, callbacks=ckpoint_cb)
```

You can configure the CheckPoint policies as required. The following describes the usage:

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix='resnet50', directory=None, config=config_ck)
model.train(epoch_num, dataset, callbacks=ckpoint_cb)
```

In the preceding code, initialize a `TrainConfig` class object to set the saving policies.

- `save_checkpoint_steps` indicates the saving frequency. That is, parameters are saved every specified number of steps.
- `keep_checkpoint_max` indicates the maximum number of CheckPoint files that can be saved.
- `prefix` indicates the prefix name of the generated CheckPoint file.
- `directory` indicates the directory for storing the file.

Create a `ModelCheckpoint` object and transfer it to the model.train method. Then you can use the CheckPoint function during training.

Generated CheckPoint files are as follows:

```
resnet50-graph.meta # Generate compiled computation graph.
resnet50-1_32.ckpt  # The file name extension is .ckpt.
resnet50-2_32.ckpt  # The file name format contains the epoch and step correspond to the saved parameters.
resnet50-3_32.ckpt  # The file name indicates that the model parameters generated during the 32th step of the third epoch are saved.
...
```

If you use the same prefix and run the training script for multiple times, CheckPoint files with the same name may be generated. MindSpore adds underscores (_) and digits at the end of the user-defined prefix to distinguish CheckPoints with the same name. If you want to delete the `.ckpt` file, please delete the `.meta` file simultaneously.

For example, `resnet50_3-2_32.ckpt` indicates the CheckPoint file generated during the 32th step of the second epoch after the script is executed for the third time.

> - When the saved single model parameter is large (more than 64M), it will fail to be saved due to the limitation of Protobuf's own data size. At this time, the restriction can be lifted by setting the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
> - When performing distributed parallel training tasks, each process needs to set different `directory` parameters to save the CheckPoint file to a different directory to prevent files from being read or written incorrectly.

### CheckPoint Configuration Policies

MindSpore provides two types of CheckPoint saving policies: iteration policy and time policy. You can create the `CheckpointConfig` object to set the corresponding policies.
CheckpointConfig contains the following four parameters:

- save_checkpoint_steps: indicates the step interval for saving a CheckPoint file. That is, parameters are saved every specified number of steps. The default value is 1.
- save_checkpoint_seconds: indicates the interval for saving a CheckPoint file. That is, parameters are saved every specified number of seconds. The default value is 0.
- keep_checkpoint_max: indicates the maximum number of CheckPoint files that can be saved. The default value is 5.
- keep_checkpoint_per_n_minutes: indicates the interval for saving a CheckPoint file. That is, parameters are saved every specified number of minutes. The default value is 0.

`save_checkpoint_steps` and `keep_checkpoint_max` are iteration policies, which can be configured based on the number of training iterations.
`save_checkpoint_seconds` and `keep_checkpoint_per_n_minutes` are time policies, which can be configured during training.

The two types of policies cannot be used together. Iteration policies have a higher priority than time policies. When the two types of policies are configured at the same time, only iteration policies take effect.
If a parameter is set to None, the related policy is cancelled.
After the training script is normally executed, the CheckPoint file generated during the last step is saved by default.

## Export MINDIR Model

When you have a CheckPoint file, if you want to do inference on device, you need to generate MINDIR models based on the network and CheckPoint. MINDIR format file can be applied to MindSpore Lite. Currently, it supports inference network based on static graph without controlling flow semantics.

If you want to do inference on the device, then you need to generate corresponding MINDIR models based on the network and CheckPoint.
Currently we support the export of MINDIR models for inference based on graph mode, which don't contain control flow. Taking the export of MINDIR model as an example to illustrate the implementation of model export,
the code is as follows:

```python
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
resnet = ResNet50()
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32.mindir', file_format='MINDIR')
```

It is recommended to use '.mindir' as the suffix of MINDIR format files.

> `input` is the input parameter of the `export` method, representing the input of the network. If the network has multiple inputs, they need to be passed into the `export` method together.
> eg：`export(network, Tensor(input1), Tensor(input2), file_name='network.mindir', file_format='MINDIR')`.

## Export AIR Model

When you have a CheckPoint file, if you want to do inference on Ascend AI processor, you need to generate AIR models based on the network and CheckPoint. AIR format file only supports Ascend AI processor. The code example of exporting this format file is as follows:

```python
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
resnet = ResNet50()
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32.air', file_format='AIR')
```

Before using the `export` interface, you need to import `mindspore.train.serialization`.

The `input` parameter is used to specify the input shape and the data type of the exported model.

It is recommended to use '.air' as the suffix of AIR format files.

> `input` is the input parameter of the `export` method, representing the input of the network. If the network has multiple inputs, they need to be passed into the `export` method together.
> eg：`export(network, Tensor(input1), Tensor(input2), file_name='network.air', file_format='AIR')`.

## Export ONNX Model

When you have a CheckPoint file, if you want to do inference on Ascend AI processor, GPU, or CPU, you need to generate ONNX models based on the network and CheckPoint. ONNX format file is a general model file, which can be applied to many kinds of hardware, such as Ascend AI processor, GPU, CPU, etc. The code example of exporting this format file is as follows:

```python
from mindspore.train.serialization import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
resnet = ResNet50()
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32.onnx', file_format='ONNX')
```

It is recommended to use '.onnx' as the suffix of ONNX format files.

> `input` is the input parameter of the `export` method, representing the input of the network. If the network has multiple inputs, they need to be passed into the `export` method together.
> eg：`export(network, Tensor(input1), Tensor(input2), file_name='network.onnx', file_format='ONNX')`.
