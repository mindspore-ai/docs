# Saving and Loading the Model

`Ascend` `GPU` `CPU` `Beginner` `Model Export` `Model Loading`

<!-- TOC -->

- [Saving and Loading the Model](#saving-and-loading-the-model)
    - [Saving the Model](#saving-the-model)
    - [Loading the Model](#loading-the-model)
        - [Validating the Model](#validating-the-model)
        - [For Transfer Learning](#for-transfer-learning)
    - [Exporting the Model](#exporting-the-model)
        - [Exporting a MindIR File](#exporting-a-mindir-file)
        - [Exporting in Other Formats](#exporting-in-other-formats)
            - [Exporting an AIR File](#exporting-an-air-file)
            - [Exporting an ONNX File](#exporting-an-onnx-file)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/tutorials/source_en/save_load_model.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

In the previous tutorial, you learn how to train the network. In this tutorial, you will learn how to save and load a model, and how to export a saved model in a specified format to different platforms for inference.

## Saving the Model

There are two main ways to save the interface of the model:

1. One is to simply save the network model, which can be saved before and after training. The advantage is that the interface is simple and easy to use, but only the state of the network model when the command is executed is retained;

2. The other one is to save the interface during networkmodel training. In the process of network model training,MindSpore automatically saves the parameters of the epochnumber and step number set during training, that is, theintermediate weight parameters generated during the modeltraining process are also saved to facilitate networkfine-tuning and stop training.

### Saving the Model Directly

Use the save_checkpoint provided by MindSpore to save the model, pass it to the network and save the path:

```python
import mindspore as ms

# The defined network model is net, which is generally used before or after training
ms.save_checkpoint(net, "./MyNet.ckpt")
```

### Saving the Model During Training

In the process of model training, use the `callbacks` parameter in `model.train` to pass in the object `ModelCheckpoint` that saves the model, which can save the model parameters and generate CheckPoint (abbreviated as ckpt) files.

```python
from mindspore.train.callback import ModelCheckpoint

ckpt_cb = ModelCheckpoint()
model.train(epoch_num, dataset, callbacks=ckpt_cb)
```

You can configure the checkpoint policies as required. The following describes the usage:

```python
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ckpt_cb = ModelCheckpoint(prefix='resnet50', directory=None, config=config_ck)
model.train(epoch_num, dataset, callbacks=ckpt_cb)
```

In the preceding code, you need to initialize a `CheckpointConfig` class object to set the saving policy.

- `save_checkpoint_steps` indicates the interval (in steps) for saving the checkpoint file.
- `keep_checkpoint_max` indicates the maximum number of checkpoint files that can be retained.
- `prefix` indicates the prefix of the generated checkpoint file.
- `directory` indicates the directory for storing files.

Create a `ModelCheckpoint` object and pass it to the `model.train` method. Then the checkpoint function can be used during training.

The generated checkpoint file is as follows:

```text
resnet50-graph.meta # Computational graph after build.
resnet50-1_32.ckpt  # The extension of the checkpoint file is .ckpt.
resnet50-2_32.ckpt  # The file name format contains the epoch and step correspond to the saved parameters.
resnet50-3_32.ckpt  # The file name indicates that the model parameters generated during the 32nd step of the third epoch are saved.
...
```

If you use the same prefix and run the training script for multiple times, checkpoint files with the same name may be generated. To help users distinguish files generated each time, MindSpore adds underscores (_) and digits to the end of the user-defined prefix. If you want to delete the `.ckpt` file, delete the `.meta` file at the same time.

For example, `resnet50_3-2_32.ckpt` indicates the checkpoint file generated during the 32nd step of the second epoch after the script is executed for the third time.

## Loading the Model

To load the model weight, you need to create an instance of the same model and then use the `load_checkpoint` and `load_param_into_net` methods to load parameters.

The sample code is as follows:

```python
from mindspore import load_checkpoint, load_param_into_net

resnet = ResNet50()
# Store model parameters in the parameter dictionary.
param_dict = load_checkpoint("resnet50-2_32.ckpt")
# Load parameters to the network.
load_param_into_net(resnet, param_dict)
model = Model(resnet, loss, metrics={"accuracy"})
```

- The `load_checkpoint` method loads the network parameters in the parameter file to the `param_dict` dictionary.
- The `load_param_into_net` method loads the parameters in the `param_dict` dictionary to the network or optimizer. After the loading, parameters in the network are stored by the checkpoint.

### Validating the Model

In the inference-only scenario, parameters are directly loaded to the network for subsequent inference and validation. The sample code is as follows:

```python
# Define a validation dataset.
dataset_eval = create_dataset(os.path.join(mnist_path, "test"), 32, 1)

# Call eval() for inference.
acc = model.eval(dataset_eval)
```

### For Transfer Learning

You can load network parameters and optimizer parameters to the model in the case of task interruption, retraining, and fine-tuning. The sample code is as follows:

```python
# Set the number of training epochs.
epoch = 1
# Define a training dataset.
dataset = create_dataset(os.path.join(mnist_path, "train"), 32, 1)
# Call train() for training.
model.train(epoch, dataset)
```

## Exporting the Model

During model training, you can add checkpoints to save model parameters for inference and retraining. If you want to perform inference on different hardware platforms, you can generate MindIR, AIR, or ONNX files based on the network and checkpoint files.

The following describes how to save a checkpoint file and export a MindIR, AIR, or ONNX file.

> MindSpore is an all-scenario AI framework that uses MindSpore IR to unify intermediate representation of network models. Therefore, you are advised to export files in MindIR format.

### Exporting a MindIR File

If you want to perform inference across platforms or hardware (such as the Ascend AI Processors, MindSpore devices, or GPUs) after obtaining a checkpoint file, you can define the network and checkpoint to generate a model file in MINDIR format. Currently, the inference network export based on static graphs is supported and does not contain control flow semantics. An example of the code for exporting the file is as follows:

```python
from mindspore import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np

resnet = ResNet50()
# Store model parameters in the parameter dictionary.
param_dict = load_checkpoint("resnet50-2_32.ckpt")

# Load parameters to the network.
load_param_into_net(resnet, param_dict)
input = np.random.uniform(0.0, 1.0, size=[32, 3, 224, 224]).astype(np.float32)
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='MINDIR')
```

> - `input` specifies the input shape and data type of the exported model. If the network has multiple inputs, you need to pass them to the `export` method.  Example: `export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='MINDIR')`
> - If `file_name` does not contain the ".mindir" suffix, the system will automatically add the ".mindir" suffix to it.

### Exporting in Other Formats

#### Exporting an AIR File

If you want to perform inference on the Ascend AI Processor after obtaining a checkpoint file, use the network and checkpoint to generate a model file in AIR format. An example of the code for exporting the file is as follows:

```python
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='AIR')
```

> - `input` specifies the input shape and data type of the exported model. If the network has multiple inputs, you need to pass them to the `export` method. Example: `export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='AIR')`
> - If `file_name` does not contain the ".air" suffix, the system will automatically add the ".air" suffix to it.

#### Exporting an ONNX File

If you want to perform inference on other third-party hardware after obtaining a checkpoint file, use the network and checkpoint to generate a model file in ONNX format. An example of the code for exporting the file is as follows:

```python
export(resnet, Tensor(input), file_name='resnet50-2_32', file_format='ONNX')
```

> - `input` specifies the input shape and data type of the exported model. If the network has multiple inputs, you need to pass them to the `export` method. Example: `export(network, Tensor(input1), Tensor(input2), file_name='network', file_format='ONNX')`
> - If `file_name` does not contain the ".onnx" suffix, the system will automatically add the ".onnx" suffix to it.
> - Currently, only the ONNX format export of ResNet series networks and BERT are supported.
