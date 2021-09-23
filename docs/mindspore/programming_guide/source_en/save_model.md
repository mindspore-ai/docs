# Saving Models

`Linux` `Ascend` `GPU` `CPU` `Model Export` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Saving Models](#saving-models)
    - [Overview](#overview)
    - [Saving CheckPoint files](#saving-checkpoint-files)
        - [Using Callback Mechanism](#use-callback-mechanism)
            - [CheckPoint Configuration Strategy](#checkpoint-configuration-policies)
        - [Using Save Checkpoint Method](#use-save-checkpoint-method)
    - [Export MindIR Model](#export-mindir-model)
    - [Export AIR Model](#export-air-model)
    - [Export ONNX Model](#export-onnx-model)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_en/save_model.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

During model training, you can add CheckPoints to save model parameters for inference and retraining after interruption. If you want to perform inference on different hardware platforms, you need to generate corresponding models based on the network and CheckPoint, such as MindIR, AIR and ONNX.

- MindIR: A functional IR of MindSpore based on graph representation, which defines an extensible graph structure and IR representation of operators, which eliminates the model differences between different backends. The model trained on Ascend 910 can be used for reasoning on the upper side of Ascend 310, GPU and MindSpore Lite.
- CheckPoint: A CheckPoint file of MindSpore is a binary file that stores the values of all training parameters. The Google Protocol Buffers mechanism with good scalability is adopted, which is independent of the development language and platform. The protocol format of CheckPoints is defined in `mindspore/ccsrc/utils/checkpoint.proto`.
- AIR: Ascend Intermediate Representation (AIR) is an open file format defined by Huawei for machine learning and can better adapt to the Ascend AI processor. It is similar to ONNX.
- ONNX: Open Neural Network Exchange (ONNX) is an open file format designed for machine learning. It is used to store trained models.

The following uses examples to describe how to save MindSpore CheckPoint files, and how to export MindIR, AIR and ONNX files.

## Saving CheckPoint files

Here are two ways to save checkpoint files

### Using Callback Mechanism

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

```text
resnet50-graph.meta # Generate compiled computation graph.
resnet50-1_32.ckpt  # The file name extension is .ckpt.
resnet50-2_32.ckpt  # The file name format contains the epoch and step correspond to the saved parameters.
resnet50-3_32.ckpt  # The file name indicates that the model parameters generated during the 32th step of the third epoch are saved.
...
```

If you use the same prefix and run the training script for multiple times, CheckPoint files with the same name may be generated. MindSpore adds underscores (_) and digits at the end of the user-defined prefix to distinguish CheckPoints with the same name. If you want to delete the `.ckpt` file, please delete the `.meta` file simultaneously.

For example, `resnet50_3-2_32.ckpt` indicates the CheckPoint file generated during the 32th step of the second epoch after the script is executed for the third time.

> - When performing distributed parallel training tasks, each process needs to set different `directory` parameters to save the CheckPoint file to a different directory to prevent files from being read or written incorrectly.

#### CheckPoint Configuration Policies

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

### Using Save_checkpoint Method

You can use the `save_checkpoint` function to save the custom information as a checkpoint file. The function declaration is as follows:

```python
def save_checkpoint(save_obj, ckpt_file_name, integrated_save=True,
                    async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM")
```

The required parameters are: `save_obj`, `ckpt_file_name`.

The following uses specific examples to illustrate how to use each parameter.

#### `save_obj` and `ckpt_file_name` parameters

**`save_obj`**: You can pass in a Cell class object or a list.
**`ckpt_file_name`**: string type, representing the name of the saved checkpoint file.

```python
from mindspore import save_checkpoint, Tensor
from mindspore import dtype as mstype
```

1. Pass in the Cell object

    ```python
    net = LeNet()
    save_checkpoint(net, "lenet.ckpt")
    ```

    ​After execution, you can save the parameters in net as a `lenet.ckpt` file.

2. Pass in the list object

    The format of the list is as follows: [{"name": param_name, "data": param_data}], which consists of a set of dict objects.

    `param_name` is the name of the object that needs to be saved, and `param_data` is the data that needs to be saved, and it is of type Tensor.

    ```python
    save_list = [{"name": "lr", "data": Tensor(0.01, mstype.float32)}, {"name": "train_epoch", "data": Tensor(20, mstype.int32)}]
    save_checkpoint(save_list, "hyper_param.ckpt")
    ```

    After execution, you can save `save_list` as a `hyper_param.ckpt` file.

#### `integrated_save` parameter

**`integrated_save`**: bool type, indicating whether the parameters are merged and saved, the default value is True. In the model parallel scenario, Tensor will be split into programs running on different cards. If `integrated_save` is set to True, these split Tensors will be merged and saved to each checkpoint file, so that the checkpoint file saves the complete training parameters.

```python
save_checkpoint(net, "lenet.ckpt", integrated_save=True)
```

#### `async_save` parameter

**`async_save`**: bool type, indicating whether to enable the asynchronous save function, the default value is False. If set to True, multi-threaded execution of checkpoint file writing operations will be enabled, so that training and saving tasks can be executed in parallel, which will save the total time of script running when training large-scale networks.

```python
save_checkpoint(net, "lenet.ckpt", async_save=True)
```

#### `append_dict` parameter

**`append_dict`**: dict type, indicating additional information that needs to be saved, for example:

```python
save_dict = {"epoch_num": 2, "lr": 0.01}
save_checkpoint(net, "lenet.ckpt",append_dict=save_dict)
```

After execution, in addition to the parameters in net, the information of `save_dict` will also be saved in `lenet.ckpt`.
Currently, only basic types of storage are supported, including int, float, bool, etc.

## Export MindIR Model

If you want to perform inference across platforms or hardware (Ascend AI processor, MindSpore on-device, GPU, etc.), you can generate the corresponding MindIR format model file through the network definition and CheckPoint. MindIR format file can be applied to MindSpore Lite. Currently, it supports inference network based on static graph without controlling flow semantics.

If you want to perform inference on the device, then you need to generate corresponding MindIR models based on the network and CheckPoint.
Currently we support the export of MindIR models for inference based on the graph mode, which do not contain control flow. Taking the export of MindIR model as an example to illustrate the implementation of model export,
the code is as follows:

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net
import numpy as np

resnet = ResNet50()
# load the parameter into net
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='MINDIR')
```

If you wish to save the data preprocess operations into MindIR and use them to perform inference,
you can pass the Dataset object into export method:

```python
de_dataset = create_dataset_for_renset(mode="eval")
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='MINDIR', dataset=de_dataset)
```

> - `input` is the input parameter of the `export` method, representing the input of the network. If the network has multiple inputs, they need to be passed into the `export` method together. eg: `export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='MINDIR')`.
> - If `file_name` does not contain the ".mindir" suffix, the system will automatically add the ".mindir" suffix to it.
> - Make sure that the Dataset object is using the preprocess operations of evaluation, otherwise you may can not get the
expected preprocess results in inference.

In order to avoid the hardware limitation of protobuf, when the exported model parameter size exceeds 1G, the framework will save the network structure and parameters separately by default.

-The name of the network structure file ends with the user-specified prefix plus `_graph.mindir`.
-In the same level directory, there will be a folder with user-specified prefix plus `_variables`, which stores network parameters.

Taking the above code as an example, if the parameter size in the model exceeds 1G, the generated directory structure is as follows:

```text
resnet50-2_32_graph.mindir
resnet50-2_32_variables
     data_0
     data_1
     ...
```

## Export AIR Model

If you want to perform inference on the Shengteng AI processor, you can also generate the corresponding AIR format model file through the network definition and CheckPoint. The code example of exporting this format file is as follows:

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net

resnet = ResNet50()
# load the parameter into net
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='AIR')
```

The `input` parameter is used to specify the input shape and the data type of the exported model.

> - `input` is the input parameter of the `export` method, representing the input of the network. If the network has multiple inputs, they need to be passed into the `export` method together. eg: `export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='AIR')`.
> - If `file_name` does not contain the ".air" suffix, the system will automatically add the ".air" suffix to it.

## Export ONNX Model

When you have a CheckPoint file, if you want to do inference on Ascend AI processor, GPU, or CPU, you need to generate ONNX models based on the network and CheckPoint. ONNX format file is a general model file, which can be applied to many kinds of hardware, such as Ascend AI processor, GPU, CPU, etc. The code example of exporting this format file is as follows:

```python
import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net

resnet = ResNet50()
# load the parameter into net
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='ONNX')
```

> - `input` is the input parameter of the `export` method, representing the input of the network. If the network has multiple inputs, they need to be passed into the `export` method together. eg: `export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='ONNX')`.
> - If `file_name` does not contain the ".onnx" suffix, the system will automatically add the ".onnx" suffix to it.
> - Currently, only the ONNX format export of ResNet series networks, YOLOV3, YOLOV4 and BERT are supported.
