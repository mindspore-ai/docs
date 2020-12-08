# Creating MindSpore ToD Models

<!-- TOC -->

- [Creating MindSpore ToD Models](#creating-mindspore-tod-model)
    - [Overview](#overview)
    - [Exporting python written models to .mindir format](#exporting-python-written-models-to-.mindir-format)
        - [MindSpore Environment Preparation](#mindspore-environment-preparation)
        - [Model Exporting Examples](#model-exporting-examples)
    - [Converting into the MindSpore ToD Model](#converting-into-the-mindspore-tod-model)
        - [Environment Preparation](#environment-preparation)
        - [Parameters Description](#parameters-description)
        - [Example](#example)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/lite/source_en/use/converter_train.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Creating your MindSpore ToD(Train on Device) model is a two step procedure:

- In the first step the model is defined and the layers that should be trained must be declared. This is being done on the server, using a MindSpore-based Python code. The model is then <b>exported</b> into a protobuf format, which is called MINDIR.
- In the seconde step this `.mindir` model is <b>converted</b> into a `.ms` format that can be loaded onto an embedded device and can be trained using the MindSpore ToD framework. The converted `.ms` models can be used for both training and inference.

## Exporting python written models to .mindir format

### MindSpore Environment Preparation

The first step in creating a MindSpore ToD(Train on Device) model starts on the server using [MindSpore infrastructure](https://www.mindspore.cn/tutorial/training/en/master/index.html). [MindSpore installation](https://gitee.com/mindspore/mindspore/blob/master/README.md#installation) requires specific environment configuration and it is common to use a docker based environment for developers. We refer the reader to [MindSpore Docker Image Instalation instructions](https://gitee.com/mindspore/mindspore/blob/master/README.md#docker-image).

Once MindSpore is installed, the developer can define the network that will be used for training. It could be an off-the-shelf network from the [model_zoo directory](https://gitee.com/mindspore/mindspore/tree/master/model_zoo) or a custom implemented network. To export it, the user must also define the loss function and the optimizer that will be used.

### Model Exporting Examples

The code below shows how to export the custom network into a `.mindir`:

```python
import mindspore as M
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple

n,c,h,w = 32,3,224,224
num_of_classes = 10

class MyNet(M.nn.Cell):
  def __init__(self, num_of_classes):
    super(MyNet, self).__init__()
    self.fc = M.nn.Dense(c*h*w, num_of_classes)

  def construct(self, x):
    x = P.Reshape()(x, (-1, c*h*w))
    x = self.fc(x)
    return x

my_net = MyNet(num_of_classes)

loss_net = M.nn.WithLossCell(my_net, M.nn.SoftmaxCrossEntropyWithLogits())
optimizer = M.nn.Adam(ParameterTuple(my_net.trainable_params()), learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False, use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
train_net = M.nn.TrainOneStepCell(loss_net, optimizer)

import numpy as np
x=M.Tensor(np.ones((n,c,h,w)), M.float32)
label = M.Tensor(np.zeros([n, num_of_classes]).astype(np.float32))

from mindspore.train.serialization import export
export(train_net, x, label, file_name="my_net_tod.mindir", file_format='MINDIR')
```

> `x` and `label` does not have to be initialized with real values to perform the export, but they must have the correct shape and type.
>
> The constructor of the optimizer get `my_net.trainable_params()`. This means that all the layers of the model will be trained.

The code below shows how to initialize googlenet model from the model_zoo and export it such that only the last layer (which is called classifer) will be updated during the training.

```python
import sys
sys.path.append('../../../../../mindspore/model_zoo/official/cv/googlenet/src/')
from googlenet import GoogleNet

import mindspore as M
from mindspore.ops import operations as P
from mindspore.common.parameter import ParameterTuple

n,c,h,w = 32,3,224,224
num_of_classes = 10

gnet = GoogleNet(num_of_classes)

trainable_weights_list = []
trainable_weights_list.extend(gnet.classifier.trainable_params())
trainable_weights = ParameterTuple(trainable_weights_list)

loss_net = M.nn.WithLossCell(gnet, M.nn.SoftmaxCrossEntropyWithLogits())
optimizer = M.nn.Adam(trainable_weights, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, use_locking=False, use_nesterov=False, weight_decay=0.0, loss_scale=1.0)
train_net = M.nn.TrainOneStepCell(loss_net, optimizer)

import numpy as np
x=M.Tensor(np.ones((n,c,h,w)), M.float32)
label = M.Tensor(np.zeros([n, num_of_classes]).astype(np.float32))

from mindspore.train.serialization import export
export(train_net, x, label, file_name="gnet_tod.mindir", file_format='MINDIR')
```

> MindSpore environment allows the developer to run MindSpore python code on server or PC. It differs from MindSpore Lite framework that allows to compile and run code on embedded devices.

## Converting into the MindSpore ToD(Train on Device) Model

### Environment Preparation

To use the MindSpore ToD(Train on Device) model conversion tool, you need to prepare the environment as follows:

- Compile or download the pre-compiled converter for ToD as explained in the [previous page](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html).
- Configure the converter environment variables as explained [here](https://www.mindspore.cn/tutorial/lite/en/master/use/build.html#output-description).

> Conversion of models is currently supported only on Linux environments

### Parameters Description

MindSpore ToD(Train on Device) model conversion tool provides multiple parameters, the majority of which are relevant only to inference models and are currently irrelavant to ToD models.

The following table describes the parameters that are relevant to ToD(Train on Device) model conversion:

| Parameter  |  Mandatory or Not   |  Parameter Description  | Value Range | Default Value |
| -------- | ------- | ----- | --- | ---- |
| `--help` | No | Prints all the help information | - | - |
| `--fmk=<FMK>`  | Yes | Original format of the input model | MINDIR | - |
| `--trainModel=true` | Yes | When set to true, converter will convert ToD models | on, off | off |
| `--modelFile=<MODELFILE>` | Yes | MINDIR model filename (including path) | - | - |
| `--outputFile=<OUTPUTFILE>` | Yes | Output model filename (including path). The suffix `.ms` is automatically added | - | - |

> The parameter name and parameter value are separated by an equal sign (=) and no space is allowed between them.

### Example

The following describes how to use the conversion command:

- Assuming the model my_model.mindir should be converted, run the following conversion command:

```bash
./converter_lite --fmk=MINDIR --trainModel=true --modelFile=my_model.mindir --outputFile=lenet_tod
```

On success the output is as follows:

```text
CONVERTER RESULT SUCCESS:0
```

This indicates that the Mindspore model is successfully converted into the MindSpore ToD(Train on Device) model and the new file `my_model.ms` is generated.

- If the conversion command failed, the output is as follows:

```text
CONVERT RESULT FAILED:
```

which is followed by an [error code](https://www.mindspore.cn/doc/api_cpp/en/master/errorcode_and_metatype.html) and error description.

## Windows Environment Instructions

Conversion of models on Windows platform will be supported only in the next release.

