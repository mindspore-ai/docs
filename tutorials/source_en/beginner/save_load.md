[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/beginner/save_load.md)

[Introduction](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) || [Quick Start](https://www.mindspore.cn/tutorials/en/master/beginner/quick_start.html) || [Tensor](https://www.mindspore.cn/tutorials/en/master/beginner/tensor.html) || [Data Loading and Processing](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html) || [Model](https://www.mindspore.cn/tutorials/en/master/beginner/model.html) || [Autograd](https://www.mindspore.cn/tutorials/en/master/beginner/autograd.html) || [Train](https://www.mindspore.cn/tutorials/en/master/beginner/train.html) || **Save and Load** || [Accelerating with Static Graphs](https://www.mindspore.cn/tutorials/en/master/beginner/accelerate_with_static_graph.html) || [Mixed Precision](https://www.mindspore.cn/tutorials/en/master/beginner/mixed_precision.html)

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

Saving model by using the [mindspore.save_checkpoint](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.save_checkpoint.html) interface, and the specified saving path of passing in the network:

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

In addition to Checkpoint, MindSpore provides a unified [Intermediate Representation (IR)](https://www.mindspore.cn/docs/en/master/design/all_scenarios.html#mindspore-ir-mindir) for cloud side (training) and end side (inference). Models can be saved as MindIR directly by using the `export` interface (only support strict graph mode).

```python
mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_syntax_level=mindspore.STRICT)
model = network()
inputs = Tensor(np.ones([1, 1, 28, 28]).astype(np.float32))
mindspore.export(model, inputs, file_name="model", file_format="MINDIR")
```

> MindIR saves both Checkpoint and model structure, so it needs to define the input Tensor to get the input shape.

The existing MindIR model can be easily loaded through the `load` interface and passed into [mindspore.nn.GraphCell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.GraphCell.html) for inference.

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

Not all Python syntax and data types are supported for MindIR export. Unsupported cases will raise errors during export.

1. MindIR export only supports **basic syntax at the STRICT level**. For detailed coverage, refer to [Static Graph Syntax Support Documentation](https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html).

2. Return value data types are limited to:

    - Python built-in types: `int`, `float`, `bool`, `str`, `tuple`, `list`.
    - MindSpore framework types: [Tensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Tensor.html), [Parameter](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html), [COOTensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.COOTensor.html), [CSRTensor](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.CSRTensor.html).

    For example, in the following program, the return value type is [mindspore.dtype](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html), which is not supported. As a result, an error is reported when MindIR is exported.

    ```python
    import mindspore
    from mindspore import nn, Tensor

    class Model(nn.Cell):

        def construct(self, x: Tensor) -> mindspore.dtype:
            return x.dtype
    ```

3. In `nn.Cell`'s `construct()` method, random number generators from [mindspore.mint](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html) (e.g., `mint.rand`, `mint.randn`, `mint.randint`, `mint.randperm`) are prohibited. Use equivalent [mindspore.ops](https://www.mindspore.cn/docs/en/master/api_python/mindspore.ops.html) interfaces instead.

4. `Parameter` objects must be defined either in `nn.Cell`'s `__init__()` method or as function input arguments. Otherwise, MindIR export will fail. For instance, a globally defined `Parameter` (as shown below) triggers an unsupported error.

    ```python
    import mindspore
    from mindspore import Parameter, nn

    # The Parameter is created outside nn.Cell and used by the Model as a global variable.
    global_param = Parameter([1, 2, 3], name='global_param')

    class Model(nn.Cell):

        def __init__(self):
            super().__init__()
            # Parameters defined within nn.Cell.__init__() are exportable.
            self.bias = Parameter([0, 1, -1])

        def construct(self, x: Parameter):  # Parameters passed as function arguments are exportable.
            # The global_param is a global variable and will cause an error during export.
            return x + global_param + self.bias

    model = Model()
    param = Parameter([1, 2, 3], name='input_param')
    mindspore.export(model, param, file_name="model", file_format="MINDIR")
    ```
