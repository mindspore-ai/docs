# Incremental Operator Build

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/optimize/op_compilation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

When a network model is executed, MindSpore builds the used operators. The time consumed in this stage increases with the scale of the network model. To improve the performance of secondary model execution, an incremental operator build mechanism is provided. When MindSpore executes a network model, the default `rank_0/kernel_meta` folder is generated in the directory where the execution is performed. During the execution, operator cache files (in the `.o`, `.info`, or `.json` format) generated during network build are saved to this directory. If you execute the same network model again or only part of the model changes, MindSpore automatically calls the reusable operator cache files in the `rank_0/kernel_meta` folder, which significantly reduces the network build time and improves the execution performance. Currently, the incremental operator build function can be used only on the Ascend AI chips.

The following demonstrates how to use the incremental operator build function.

## Usage

Incremental operator build is enabled by default on MindSpore and does not need to be controlled. The following describes how to build a simple network model case `test_square.py`.

Execute the following test case:

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops as ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.square = ops.Square()

    def construct(self, data):
        return self.square(data)

def test_net():
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    square = Net()
    output = square(ms.Tensor(x))
    print("x: ", x)
    print("output: ", output)


if __name__ == "__main__":
    test_net()

```

The network model consists of a single operator `Square`, and the output is a square value of the input. The execution result is as follows:

```text
x: [1. 4. 9.]
output: [1. 16. 81.]
```

In the current execution directory, a `rank_0/kernel_meta` folder is generated containing the Square operator's `.o` file, `.json` file, `.info` file, and other files. For an operator:

The `.o` file is an executable file generated by MindSpore for the operator during network model execution.

The `.info` file records all valid information about the operator, including the operator name, attributes, input and output formats, and input and output data types. The `.info` file is used to search for and determine whether the `.o` file of the operator can be reused.

The `.json` file stores the operator build result, which will be used during running.

After the preceding three types of operator cache files are generated, you can perform incremental operator build when executing the network model. That is, only new or modified operators will be built, greatly improving the network build performance.

## FAQs

- Cache files cannot be shared in different scenarios, such as multi-device and single-device scenarios, or training and inference scenarios.

- The `rank_0` is the default value if the env `RANK_ID` is empty. If the `RANK_ID` is not empty, for example`RANK_ID=3`, the path `rank_3/kernel_meta` will be generated.

- The path of `kernel_meta` can be specified by the environment variable `MS_COMPILER_CACHE_PATH`. For example, `export MS_COMPILER_CACHE_PATH=/home/xxx/`,`export RANK_ID=2`, the operator compilation cache file will be saved in `/home/xxx/rank_2/kernel_meta/`.

- When multiple devices are running, the `rank_{ID}/kernel_meta` folder is generated in multiple `device` directories when the network model is executed(The `ID`is the value of environment variable `RANK_ID`).

  Note that when multiple devices are running, if the operator cache files in `rank_{ID}/kernel_meta` of some devices are deleted and the same network model is executed again, devices that do not need to be rebuilt may time out. As a result, the execution fails. In this case, you can set the environment variable `HCCL_CONNECT_TIMEOUT`, that is, the waiting time between multiple devices, to avoid failure. However, this method takes a long time, which is equivalent to deleting and rebuilding all devices(The `ID`is the value of environment variable `RANK_ID`).