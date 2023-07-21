# Saving and Loading Model Parameters

[![View Source On Gitee](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r0.6/tutorials/source_en/use/saving_and_loading_model_parameters.md)

## Overview

During model training, you can add CheckPoints to save model parameters for inference and retraining after interruption.

The application scenarios are as follows:

- Inference after training.
    - After model training, save model parameters for inference or prediction.
    - During training, through real-time accuracy validation, save the model parameters with the highest accuracy for prediction.
- Retraining.
    - During a long-time training, save the generated CheckPoint files to prevent the training from starting from the beginning after it exits abnormally.
    - Train a model, save parameters, and perform fine-tuning for different tasks.

A CheckPoint file of MindSpore is a binary file that stores the values of all training parameters. The Google Protocol Buffers mechanism with good scalability is adopted, which is independent of the development language and platform.
The protocol format of CheckPoints is defined in `mindspore/ccsrc/utils/checkpoint.proto`.

The following uses an example to describe the saving and loading functions of MindSpore. The ResNet-50 network and the MNIST dataset are selected.

## Saving Model Parameters
During model training, use the callback mechanism to transfer the object of the callback function `ModelCheckpoint` to save model parameters and generate CheckPoint files.
You can use the `CheckpointConfig` object to set the CheckPoint saving policies.
The saved parameters are classified into network parameters and optimizer parameters.

`ModelCheckpoint` provides default configuration policies for users to quickly get started.
The following describes the usage:
```python
from mindspore.train.callback import ModelCheckpoint
ckpoint_cb = ModelCheckpoint()
model.train(epoch_num, dataset, callbacks=ckpoint_cb)
```

You can configure the CheckPoint policies as required.
The following describes the usage:

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ckpoint_cb = ModelCheckpoint(prefix='resnet50', directory=None, config=config_ck)
model.train(epoch_num, dataset, callbacks=ckpoint_cb)
```

In the preceding code, initialize a `TrainConfig` class object to set the saving policies.
`save_checkpoint_steps` indicates the saving frequency. That is, parameters are saved every specified number of steps. `keep_checkpoint_max` indicates the maximum number of CheckPoint files that can be saved.
`prefix` indicates the prefix name of the generated CheckPoint file. `directory` indicates the directory for storing the file.
Create a `ModelCheckpoint` object and transfer it to the model.train method. Then you can use the CheckPoint function during training.
If you want to delete the `.ckpt` file, please delete the `.meta` file simultaneously.

Generated CheckPoint files are as follows:

> - resnet50-graph.meta # Generate compiled computation graph.
> - resnet50-1_32.ckpt  # The file name extension is .ckpt.
> - resnet50-2_32.ckpt  # The file name format contains the epoch and step correspond to the saved parameters.
> - resnet50-3_32.ckpt  # The file name indicates that the model parameters generated during the 32th step of the third epoch are saved.
> - ...


If you use the same prefix and run the training script for multiple times, CheckPoint files with the same name may be generated.
MindSpore adds underscores (_) and digits at the end of the user-defined prefix to distinguish CheckPoints with the same name.

For example, `resnet50_3-2_32.ckpt` indicates the CheckPoint file generated during the 32th step of the second epoch after the script is executed for the third time.


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
If a parameter is set to None, the related policy is canceled.
After the training script is normally executed, the CheckPoint file generated during the last step is saved by default.


## Loading Model Parameters

After saving CheckPoint files, you can load parameters.

### For Inference Validation

In inference-only scenarios, use `load_checkpoint` to directly load parameters to the network for subsequent inference validation.

The sample code is as follows:

```python
resnet = ResNet50()
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
dateset_eval = create_dataset(os.path.join(mnist_path, "test"), 32, 1) # define the test dataset
loss = CrossEntropyLoss()
model = Model(resnet, loss)
acc = model.eval(dataset_eval)
```

The `load_checkpoint` method loads network parameters in the parameter file to the model. After the loading, parameters in the network are those saved in CheckPoints.
The `eval` method validates the accuracy of the trained model.

### For Retraining

In the retraining after task interruption and fine-tuning scenarios, you can load network parameters and optimizer parameters to the model.

The sample code is as follows:
```python
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
resnet = ResNet50()
opt = Momentum()
# load the parameter into net
load_param_into_net(resnet, param_dict)
# load the parameter into operator
load_param_into_net(opt, param_dict)
loss = SoftmaxCrossEntropyWithLogits()
model = Model(resnet, loss, opt)
model.train(epoch, dataset)
```

The `load_checkpoint` method returns a parameter dictionary and then the `load_param_into_net` method loads parameters in the parameter dictionary to the network or optimizer.

## Export GEIR Model and ONNX Model
When you have a CheckPoint file, if you want to do inference, you need to generate corresponding models based on the network and CheckPoint.
Currently we support the export of GEIR models based on Ascend AI processor and the export of ONNX models based on GPU. Taking the export of GEIR model as an example to illustrate the implementation of model export,
the code is as follows:
```python
from mindspore.train.serialization import export
import numpy as np
resnet = ResNet50()
# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# load the parameter into net
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size = [32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name = 'resnet50-2_32.pb', file_format = 'GEIR')
```
Before using the `export` interface, you need to import` mindspore.train.serialization`.
The `input` parameter is used to specify the input shape and data type of the exported model.
If you want to export the ONNX model, you only need to specify the `file_format` parameter in the` export` interface as ONNX: `file_format = 'ONNX'`.
