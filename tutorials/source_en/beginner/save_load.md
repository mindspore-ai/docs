[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/beginner/save_load.md)

[Introduction](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/tensor.html) || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/train.html) || **Save and Load** || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/accelerate_with_static_graph.html) || [Mixed Precision](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/beginner/mixed_precision.html)

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
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)
```

```text
[]
```

> - `param_not_load` is an unloaded parameter list, and empty means all parameters are loaded successfully.
> - When MindX DL (Ascend deep learning component) version 6.0 or later is installed in the environment, the MindIO acceleration CheckPoint function is enabled by default. For details, please refer to [MindIO Introduction](https://www.hiascend.com/document/detail/en/mindx-dl/500/mindio/mindioug/mindio_001.html). MindX DL can be downloaded [here](https://www.hiascend.com/en/software/mindx-dl/community).

## Saving and Loading MindIR

In addition to Checkpoint, MindSpore provides a unified [Intermediate Representation (IR)](https://www.mindspore.cn/docs/en/r2.6.0rc1/design/all_scenarios.html#mindspore-ir-mindir) for cloud side (training) and end side (inference). Models can be saved as MindIR directly by using the `export` interface (only support strict graph mode).

```python
mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_syntax_level=mindspore.STRICT)
model = network()
inputs = Tensor(np.ones([1, 1, 28, 28]).astype(np.float32))
mindspore.export(model, inputs, file_name="model", file_format="MINDIR")
```

> MindIR saves both Checkpoint and model structure, so it needs to define the input Tensor to get the input shape.

The existing MindIR model can be easily loaded through the `load` interface and passed into `nn.GraphCell` for inference.

> `nn.GraphCell` only supports graph mode.

```python
graph = mindspore.load("model.mindir")
model = nn.GraphCell(graph)
outputs = model(inputs)
print(outputs.shape)
```

```text
(1, 10)
```

### Syntax Support Scope

Not all Python syntax and data types are supported for MindIR export. MindIR export has a specific support scope, and if the syntax falls outside this scope, an error will be reported during the export process.

First, MindIR export only supports **strict-level graph mode**. For detailed support scope, please refer to the [Static Graph Syntax Support Documentation](https://www.mindspore.cn/tutorials/en/r2.6.0rc1/compile/static_graph.html).

Second, in addition to the syntax restrictions of strict-level graph mode, MindIR has additional constraints on the types of return values. For example, returning `mindspore.dtype` is not supported. The following program will raise an error during MindIR export.

```python
import mindspore as ms
from mindspore import nn, ops, Tensor

class Model(nn.Cell):
    def __init__(self):
        super().__init__()
        self.dtype = ops.DType()

    def construct(self, x: Tensor) -> ms.dtype:
        return self.dtype(x)
```

Furthermore, if a `Parameter` object is created outside `nn.Cell`, MindIR does not support exporting that Parameter. This typically occurs in the following scenarios:

- A `Parameter` is created directly in the global scope of the script.
- A `Parameter` is created in a non `nn.Cell` class.
- Random number generation api from the [mindspore.mint](https://www.mindspore.cn/docs/en/r2.6.0rc1/api_python/mindspore.mint.html) package are used, such as `mint.randn`, `mint.randperm`, etc., because these random number interfaces create `Parameter` in the global scope.

For example, the following two programs will raise errors during the export process.

```python
from mindspore import Tensor, Parameter, nn

param = Parameter(Tensor([1, 2, 3, 4]))  # Created outside nn.Cell

class Model(nn.Cell):
    def construct(self, x: Tensor) -> Tensor:
        return x + param
```

```python
from mindspore import Tensor, nn, mint

class Model(nn.Cell):
    def construct(self, n: int) -> Tensor:
        return mint.randn(n)
```
