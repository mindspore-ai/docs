# 算子增量编译

`Linux` `Ascend` `模型训练` `初级` `中级` `高级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_zh_cn/incremental_operator_build.md)

## 概述

在执行网络模型的过程中，MindSpore会对所使用的算子进行编译，该阶段耗时会随网络模型规模的增大而增大。为提升用户二次执行模型的性能体验，我们提供了一种算子增量编译机制。MindSpore执行网络模型时会在执行目录下生成`kernel_meta`目录，并在执行过程中保存网络编译生成的算子缓存文件到此目录，包括`.o`文件，`.info`文件以及`.json`文件。若用户再次执行相同的网络模型，或者仅有部分变化，MindSpore会自动调用`kernel_meta`目录下可复用的算子缓存文件，显著减少网络编译时间，提升执行性能。目前算子增量编译功能仅支持在昇腾AI芯片上使用。

下面，本教程将演示如何使用算子增量编译。

## 使用方法

算子增量编译在MindSpore中默认开启，用户无需对其进行控制。下面我们在`src`目录下构造一个简单的网络用例`test_square.py`。当前目录结构为：

```text
└─src
    └── test_square.py
```

执行如下用例：

```python
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
import mindspore.ops as ops
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.square = ops.Square()

    def construct(self, data):
        return self.square(data)

def test_net():
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    square = Net()
    output = square(Tensor(x))
    print("x: ", x)
    print("output: ", output)


```

该网络由一个单算子`Square`构成，输出为输入的平方值。执行结果如下：

```text
x: [1. 4. 9.]
output: [1. 16. 81.]
```

在执行目录下，生成了`kernel_meta`文件夹，其中包含Square算子的`.o`文件，`.json`文件以及`.info`文件，当前目录结构为：

```text
└─src
    ├── test_square.py
    └── kernel_meta
        ├── Square_3307185124911971026_7.info
        ├── Square_3307185124911971026_7.json
        └── Square_3307185124911971026_7.o
```

对于一个算子来说：

`.o`文件即MindSpore在网络执行过程中对该算子生成的可执行文件。

`.info`文件记录了该算子的所有有效信息，包括算子名称、算子属性、输入输出格式、输入输出数据类型等等。`.info`文件用于查找并确定算子的`.o`文件是否可复用。详细内容如下：

```text
{"SocInfo":{"autoTilingMode":"NO_TUNE","coreNum":"","coreType":"","l1Fusion":"false","l2Fusion":"false","l2Mode":"2","op_debug_level":"","op_impl_mode":"","op_impl_mode_list":[],"socVersion":"Ascend910A"},"impl_path":"","op_info":{"Type":"Square","attrs":null,"full_name":"Default/Square-op1","gen_model":"single","graph_id":0,"inputs":[[{"dtype":"float32","format":"NCHW","name":"x_0","ori_format":"NCHW","ori_shape":[3],"param_type":"required","range":[[3,3]],"shape":[3],"valid":true}]],"is_dynamic_shape":false,"kernel_name":"Square_2989580383048251395_7","module_name":"impl.square","name":"square","outputs":[[{"dtype":"float32","format":"NCHW","name":"y","ori_format":"NCHW","ori_shape":[3],"param_type":"required","range":[[3,3]],"shape":[3],"valid":true}]],"py_module_path":"/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe","socVersion":"Ascend910A"},"platform":"TBE"}
```

`.json`文件存放了算子编译结果，在运行时将会使用到。详细内容如下：

```text
{
  "batchBindOnly":1,
  "binFileName":"Square_3307185124911971026_7",
  "binFileSuffix":".o",
  "blockDim":1,
  "kernelName":"Square_3307185124911971026_7__kernel0",
  "magic":"RT_DEV_BINARY_MAGIC_ELF",
  "opParaSize":0,
  "parameters":[
    0,
    0
  ],
  "sha256":"64d4963bf6b619c2d85da67611f5677e0ea11bba0413ed3620b0926b1d072a1a"
}
```

在生成如上的三种算子缓存文件之后，用户在执行网络模型时即可进行算子增量编译，即仅编译新增或者有改动的算子，大幅提升网络编译性能。

## 常见问题

- 不同场景下缓存文件通常不能共用，例如多卡与单卡、训练与推理等。

- 在多卡运行时，执行网络模型将会在多个`device`目录下均生成`kernel_meta`文件夹。

  请注意，在多卡运行的情况下，如果仅删除部分卡的`kernel_meta`下的算子缓存文件后重复执行相同的网络模型，可能会引起不需重新编译算子的部分卡等候超时，导致执行失败。在这种情况下，可以通过设置环境变量`HCCL_CONNECT_TIMEOUT`，即多卡间等待时间来避免失败，但该方式耗时等同于全部删除缓存重新编译。

- 如果在网络编译的过程中打断进程，有概率会导致`kernel_meta`中的缓存文件生成错误，并使得后续重新执行的过程失败。此时需要用户去删除`kernel_meta`文件夹，重新编译网络。
