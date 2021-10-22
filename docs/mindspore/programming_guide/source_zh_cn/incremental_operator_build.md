# 算子增量编译

`Ascend` `模型调试`

<!-- TOC -->

- [算子增量编译](#算子增量编译)
    - [概述](#概述)
    - [使用方法](#使用方法)
    - [常见问题](#常见问题)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/incremental_operator_build.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

在执行网络模型的过程中，MindSpore会对所使用的算子进行编译，该阶段耗时会随网络模型规模的增大而增大。为提升用户二次执行模型的性能体验，我们提供了一种算子增量编译机制。MindSpore执行网络模型时会在执行目录下生成`rank_0/kernel_meta`默认目录，并在执行过程中保存网络编译生成的算子缓存文件到此目录，包括`.o`文件，`.info`文件以及`.json`文件。若用户再次执行相同的网络模型，或者仅有部分变化，MindSpore会自动调用`rank_0/kernel_meta`目录下可复用的算子缓存文件，显著减少网络编译时间，提升执行性能。目前算子增量编译功能仅支持在昇腾AI芯片上使用。

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

在执行目录下，生成了`rank_0/kernel_meta`文件夹，其中包含Square算子的`.o`文件，`.json`文件以及`.info`文件，当前目录结构为：

```text
└─src
    ├── test_square.py
    └── rank_0
        └──kernel_meta
           ├── square_12484080525657478220_2.info
           ├── square_12484080525657478220_2.json
           └── square_12484080525657478220_2.o
```

对于一个算子来说：

`.o`文件即MindSpore在网络执行过程中对该算子生成的可执行文件。

`.info`文件记录了该算子的所有有效信息，包括算子名称、算子属性、输入输出格式、输入输出数据类型等等。`.info`文件用于查找并确定算子的`.o`文件是否可复用。详细内容如下：

```text
{
    "job_content": {
        "SocInfo": {
            "autoTilingMode": "NO_TUNE",
            "coreNum": "",
            "coreType": "",
            "deviceId": "2",
            "l1Fusion": "false",
            "l2Fusion": "false",
            "l2Mode": "2",
            "mdl_bank_path": "",
            "offlineTune": false,
            "op_bank_path": "",
            "op_bank_update": false,
            "op_debug_dir": "./rank_0/",
            "op_debug_level": "0",
            "op_impl_mode": "",
            "op_impl_mode_list": [],
            "socVersion": "Ascend910A",
            "vector_fp_ceiling": ""
        },
        "full_name": "Default/Square-op1",
        "fusion_op_name": "square_12484080525657478220_2",
        "graph_name": "",
        "l1_size": -1,
        "op_list": [
            {
                "name": "x_0",
                "output_desc": [
                    {
                        "L1_addr_offset": 0,
                        "L1_fusion_type": -1,
                        "L1_workspace_size": -1,
                        "addr_type": 0,
                        "data_type": "float32",
                        "dtype": "float32",
                        "format": "ND",
                        "name": "x_0",
                        "ori_format": "NCHW",
                        "ori_shape": [
                            3
                        ],
                        "param_type": "required",
                        "range": [
                            [
                                3,
                                3
                            ]
                        ],
                        "shape": [
                            3
                        ],
                        "slice_offset": [],
                        "split_index": 0,
                        "total_shape": [],
                        "valid": true,
                        "valid_shape": []
                    }
                ],
                "type": "Data"
            },
            {
                "build_type": "accurately_build",
                "dynamic_compile_static": false,
                "func_name": "square",
                "input_desc": [
                    {
                        "L1_addr_offset": 0,
                        "L1_fusion_type": -1,
                        "L1_workspace_size": -1,
                        "addr_type": 0,
                        "data_type": "float32",
                        "dtype": "float32",
                        "format": "ND",
                        "name": "x_0",
                        "ori_format": "NCHW",
                        "ori_shape": [
                            3
                        ],
                        "param_type": "required",
                        "range": [
                            [
                                3,
                                3
                            ]
                        ],
                        "shape": [
                            3
                        ],
                        "slice_offset": [],
                        "split_index": 0,
                        "total_shape": [],
                        "valid": true,
                        "valid_shape": []
                    }
                ],
                "int64mode": false,
                "max_kernel_id": 10,
                "miss_support_info": "",
                "module_name": "impl.square",
                "name": "Default/Square-op1",
                "op_name": "square_12484080525657478220_2",
                "ori_name": [
                    "Default/Square-op1"
                ],
                "output_data_desc": [
                    {
                        "L1_addr_offset": 0,
                        "L1_fusion_type": -1,
                        "L1_workspace_size": -1,
                        "addr_type": 0,
                        "data_type": "float32",
                        "dtype": "float32",
                        "format": "ND",
                        "ori_format": "NCHW",
                        "ori_shape": [
                            3
                        ],
                        "param_type": "required",
                        "range": [
                            [
                                3,
                                3
                            ]
                        ],
                        "shape": [
                            3
                        ],
                        "slice_offset": [],
                        "split_index": 0,
                        "total_shape": [],
                        "valid": true,
                        "valid_shape": []
                    }
                ],
                "output_desc": [
                    {
                        "L1_addr_offset": 0,
                        "L1_fusion_type": -1,
                        "L1_workspace_size": -1,
                        "addr_type": 0,
                        "data_type": "float32",
                        "dtype": "float32",
                        "format": "ND",
                        "name": "y",
                        "ori_format": "NCHW",
                        "ori_shape": [
                            3
                        ],
                        "param_type": "required",
                        "range": [
                            [
                                3,
                                3
                            ]
                        ],
                        "shape": [
                            3
                        ],
                        "slice_offset": [],
                        "split_index": 0,
                        "total_shape": [],
                        "valid": true,
                        "valid_shape": []
                    }
                ],
                "pattern": "Opaque",
                "py_module_path": "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe",
                "type": "Square",
                "unknown_shape": false
            }
        ],
        "scope_id": -1
    },
    "job_id": 1,
    "job_type": "Compile",
    "source_id": 2
}
```

`.json`文件存放了算子编译结果，在运行时将会使用到。详细内容如下：

```text
{
    "batchBindOnly": 1,
    "binFileName": "square_12484080525657478220_2",
    "binFileSuffix": ".o",
    "blockDim": 1,
    "build_result": "",
    "kernelName": "square_12484080525657478220_2__kernel0",
    "magic": "RT_DEV_BINARY_MAGIC_ELF",
    "opParaSize": 0,
    "parameters": [
        0,
        0
    ],
    "sha256": "38ec670e4536958a70a653a0f3bbc7a5aadf66b5fd2b6cfe5379964668929797"
}
```

在生成如上的三种算子缓存文件之后，用户在执行网络模型时即可进行算子增量编译，即仅编译新增或者有改动的算子，大幅提升网络编译性能。

## 常见问题

- 不同场景下缓存文件通常不能共用，例如多卡与单卡、训练与推理等。

- `rank_0`是在环境变量`RANK_ID`为空的情况下的默认值，如果该环境变量的值不为空，则会生成相应`RANK_ID`号的路径。如`RANK_ID=3`，则生成`rank_3/kernel_meta`。

- `kernel_meta`生成的路径可以通过环境变量`MS_COMPILER_CACHE_PATH`指定，例如`export MS_COMPILER_CACHE_PATH=/home/zhang_san/`，`export RANK_ID=2`，则算子编译缓存文件位于`/home/zhang_san/rank_2/kernel_meta/`。

- 在多卡运行时，执行网络模型将会在多个`device`目录下均生成`rank_{ID}/kernel_meta`文件夹（`ID`为环境变量`RANK_ID`的值）。

  请注意，在多卡运行的情况下，如果仅删除部分卡的`rank_{ID}/kernel_meta`下的算子缓存文件后重复执行相同的网络模型，可能会引起不需重新编译算子的部分卡等候超时，导致执行失败。在这种情况下，可以通过设置环境变量`HCCL_CONNECT_TIMEOUT`，即多卡间等待时间来避免失败，但该方式耗时等同于全部删除缓存重新编译（`ID`为环境变量`RANK_ID`的值）。

- 如果在网络编译的过程中打断进程，有概率会导致`rank_0/kernel_meta`中的缓存文件生成错误，并使得后续重新执行的过程失败。此时需要用户去删除`rank_0/kernel_meta`文件夹，重新编译网络。
