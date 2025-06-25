# 自定义算子接入

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/developer_guide/operations/npu_ops.md)

本文档将以 **`adv_step_flash`** 算子的接入为例，讲解如何在 vLLM MindSpore 项目中接入一个新的自定义算子。本文重点在于接入流程，算子的实现参考 MindSpore 官方教程：[动态图自定义算子接入方式](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_customopbuilder.html)。以下章节将介绍文件的组织结构及接入步骤。

实际开发中，可根据项目需求扩展更多功能，算子实现细节可参考 [MindSpore 自定义算子实现方式](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_customopbuilder.html)。

**目前该特性只支持动态图场景。**

## 文件组织结构

接入自定义算子需要在 vLLM MindSpore 项目的 `vllm_mindspore/ops` 目录下添加代码，目录结构如下：

```text
vllm_mindspore/ops/
├── ascendc/
│   ├── adv_step_flash.h      // AscendC AdvStepFlash 算子声明
│   ├── adv_step_flash.c      // AscendC AdvStepFlash 算子实现
│   └── ...
├── module/
│   ├── module.h              // 公共模块注册头文件
│   ├── module.cpp            // 公共模块注册实现文件
│   ├── adv_step_flash.cpp    // 接入层代码，注册 AdvStepFlash 算子的 Python 接口
│   └── ...
```

- **`ops/ascendc/`**：放置 AscendC 自定义算子的实现代码。
- **`ops/module/`**：放置算子接入层代码，包括公共模块注册（`module.h`、`module.cpp`）和算子接入代码（如 `adv_step_flash.cpp`）。

## 接入流程

接入一个自定义算子，在算子实现方面，需在`ops/ascendc/`目录中，创建[算子接口定义](#算子接口声明)，[算子实现](#算子实现)与[算子接入](#算子接入)。在完成自定义算子初步的开发与接入后，可进行[算子编译并测试](#算子编译并测试)。

### 算子接口声明

在 `ops/ascendc/` 目录下，创建头文件（如 `my_custom_op.h`），以声明算子函数及相关接口，内容参考：

```cpp
#ifndef VLLM_MINDSPORE_OPS_ASCENDC_MY_CUSTOM_OP_H
#define VLLM_MINDSPORE_OPS_ASCENDC_MY_CUSTOM_OP_H

extern void MyCustomOpKernelEntry(uint32_t blockDims, void *l2ctrl, void *aclStream,
                                  uint8_t *input, uint8_t *output, int32_t param1, int32_t param2);

#endif  // VLLM_MINDSPORE_OPS_ASCENDC_MY_CUSTOM_OP_H
```

### 算子实现

在 `ops/ascendc/` 目录下创建实现文件（如 `my_custom_op.c`），以实现算子的核心逻辑，内容参考：

```cpp
#include "my_custom_op.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void my_custom_op_impl(GM_ADDR input, GM_ADDR output,
                                                        int32_t param1, int32_t param2) {
  // AscendC operation implement
}

#ifndef __CCE_KT_TEST__
void MyCustomOpKernelEntry(uint32_t blockDims, void *l2ctrl, void *aclStream,
                           uint8_t *input, uint8_t *output, int32_t param1, int32_t param2) {
  my_custom_op_impl<<<blockDims, l2ctrl, aclStream>>>(input, output, param1, param2);
}
#endif
```

### 算子接入

在 `module/` 目录下创建一个新的接入文件（如 `my_custom_op.cpp`），内容参考 `adv_step_flash.cpp`：

```cpp
#include "ms_extension.h"
#include "ascendc/my_custom_op.h"
#include "module/module.h"

using BaseTensorPtr = mindspore::tensor::BaseTensorPtr;

void MyCustomOpPythonInterface(int32_t param1, int32_t param2,
                               BaseTensorPtr input, BaseTensorPtr output) {
  ...
}

MS_EXTENSION_MODULE(my_custom_op) {
  m.def("my_custom_op", &MyCustomOpPythonInterface, "My custom operator",
        pybind11::arg("param1"), pybind11::arg("param2"),
        pybind11::arg("input"), pybind11::arg("output"));
}
```

### 算子编译并测试

1. **代码集成**：将代码集成至 vllm-mindspore 项目。
2. **编译项目**：于vllm-mindspore工程中，执行`pip install .`，编译安装vLLM MindSpore。
3. **测试算子接口**：使用 Python 调用注册的算子接口：

    ```python
    from vllm_mindspore import npu_ops
    import numpy as np
    import mindspore as ms

    input = ms.Tensor(np.array([1, 2, 3], dtype=np.int32))
    output = ms.Tensor(np.zeros_like(input))

    npu_ops.my_custom_op(10, 20, input, output)
    print("Output:", output)
    ```
