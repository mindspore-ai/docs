# Loading a Model for Inference and Transfer Learning

`Linux` `Ascend` `GPU` `CPU` `Model Loading` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Loading a Model for Inference and Transfer Learning](#loading-a-model-for-inference-and-transfer-learning)
    - [Overview](#overview)
    - [Loading the local Model](#loading-the-local-model)
        - [For Inference Validation](#for-inference-validation)
        - [For Transfer Training](#for-transfer-training)
    - [Modify and Saving the Checkpoint File](#modify-and-saving-the-checkpoint-file)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/load_model_for_inference_and_transfer.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

CheckPoints which are saved locally during model training, they are used for inference and transfer training.

The following uses examples to describe how to load models from local.

> You can view the definition of the network and dataset here:
>
> <https://gitee.com/mindspore/docs/tree/master/docs/sample_code/save_model>

## Loading the local Model

After saving CheckPoint files, you can load parameters.

### For Inference Validation

In inference-only scenarios, use `load_checkpoint` to directly load parameters to the network for subsequent inference validation.

The sample code is as follows:

```python
from mindspore import Model, load_checkpoint
from mindspore.nn import SoftmaxCrossEntropyWithLogits

resnet = ResNet50()
load_checkpoint("resnet50-2_32.ckpt", net=resnet)
# create eval dataset, mnist_path is the data path
dataset_eval = create_dataset(mnist_path)
loss = CrossEntropyLoss()
model = Model(resnet, loss, metrics={"accuracy"})
acc = model.eval(dataset_eval)
```

The `load_checkpoint` method loads network parameters in the parameter file to the model. After the loading, parameters in the network are those saved in CheckPoints.
The `eval` method validates the accuracy of the trained model.

### For Transfer Training

In the retraining and fine-tuning scenarios for task interruption, you can load network parameters and optimizer parameters to the model.

The sample code is as follows:

```python
from mindspore import Model, load_checkpoint
from mindspore.nn import Momentum
from mindspore.nn import SoftmaxCrossEntropyWithLogits

# return a parameter dict for model
param_dict = load_checkpoint("resnet50-2_32.ckpt")
resnet = ResNet50()
opt = Momentum(resnet.trainable_params(), 0.01, 0.9)
# load the parameter into net
load_param_into_net(resnet, param_dict)
# load the parameter into optimizer
load_param_into_net(opt, param_dict)
loss = SoftmaxCrossEntropyWithLogits()
model = Model(resnet, loss, opt)
model.train(epoch, dataset)
```

The `load_checkpoint` method returns a parameter dictionary and then the `load_param_into_net` method loads parameters in the parameter dictionary to the network or optimizer.

## Modify and Saving the Checkpoint File

If you want to modify the checkpoint file, you can use the `load_checkpoint` interface, which returns a dict.

This dict can be modified for subsequent operations.

```python
from mindspore import Parameter, Tensor, load_checkpoint, save_checkpoint
# Load the checkpoint file
param_dict = load_checkpoint("lenet.ckpt")
# You can view the key and value by traversing this dict
for key, value in param_dict.items():
  # key is string type
  print(key)
  # value is the parameter type, use the data.asnumpy() method to view its value
  print(value.data.asnumpy())

# After getting param_dict, you can perform basic additions and deletions to it for subsequent use

# 1. Delete the element named "conv1.weight"
del param_dict["conv1.weight"]
# 2. Add an element named "conv2.weight" and set its value to 0
param_dict["conv2.weight"] = Parameter(Tensor([0]))
# 3. Modify the name "conv1.bias" to 1
param_dict["fc1.bias"] = Parameter(Tensor([1]))

# Restore the modified param_dict as a checkpoint file
save_list = []
# Traverse the modified dict, convert it into a storage format supported by MindSpore, and store it as a checkpoint file
for key, value in param_dict.items():
  save_list.append({"name": key, "data": value.data})
save_checkpoint(save_list, "new.ckpt")
```
