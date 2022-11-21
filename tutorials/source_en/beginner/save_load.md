<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/save_load.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Dataset](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Transforms](https://www.mindspore.cn/tutorials/en/master/beginner/transforms.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || **Save and Load**

# Saving and Loading the Model

The previous section describes how to adjust hyperparameters and train network models. During network model training, we want to save the intermediate and final results for fine-tuning and subsequent model deployment and inference. This section describes how to save and load a model.

```python
import numpy as np
import mindspore
from mindspore import nn
from mindspore import Tensor
```

```python
def network():
    model = nn.SequentialCell(
                nn.Flatten(),
                nn.Dense(28*28, 512),
                nn.ReLU(),
                nn.Dense(512, 512),
                nn.ReLU(),
                nn.Dense(512, 10))
    return model
```

## Saving and Loading the Model Weight

Saving model by using the `save_checkpoint` interface, and the specified saving path of passing in the network:

```python
model = network()
mindspore.save_checkpoint(model, "model.ckpt")
```

To load the model weights, you need to create instances of the same model and then load the parameters by using the `load_checkpoint` and `load_param_into_net` methods.

```python
model = network()
param_dict = mindspore.load_checkpoint("model.ckpt")
param_not_load = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)
```

```text
[]
```

> `param_not_load` is an unloaded parameter list, and empty means all parameters are loaded successfully.

## Saving and Loading MindIR

In addition to Checkpoint, MindSpore provides a unified [Intermediate Representation (IR)](https://www.mindspore.cn/docs/en/master/design/mindir.html) for cloud side (training) and end side (inference). Models can be saved as MindIR directly by using the `export` interface.

```python
model = network()
inputs = Tensor(np.ones([1, 1, 28, 28]).astype(np.float32))
mindspore.export(model, inputs, file_name="model", file_format="MINDIR")
```

> MindIR saves both Checkpoint and model structure, so it needs to define the input Tensor to get the input shape.

The existing MindIR model can be easily loaded through the `load` interface and passed into `nn.GraphCell` for inference.

> `nn.GraphCell` only supports graph mode.

```python
mindspore.set_context(mode=mindspore.GRAPH_MODE)

graph = mindspore.load("model.mindir")
model = nn.GraphCell(graph)
outputs = model(inputs)
print(outputs.shape)
```

```text
(1, 10)
```
