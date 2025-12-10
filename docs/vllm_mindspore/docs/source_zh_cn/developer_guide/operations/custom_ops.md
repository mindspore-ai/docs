# 自定义算子接入

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/docs/vllm_mindspore/docs/source_zh_cn/developer_guide/operations/custom_ops.md)

当内置算子不满足需求时，你可以利用MindSpore提供的自定义算子功能接入你的算子。

本文档将以 **`advance_step_flashattn`** 算子为例，讲解如何在vLLM-MindSpore插件项目中接入一个AscendC自定义算子。

本文重点在于介绍把算子集成进vLLM-MindSpore插件的流程。自定义算子的细节请参考 MindSpore 官方教程：[基于CustomOpBuilder的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/r2.7.2/custom_program/operation/op_customopbuilder.html)。AscendC算子的开发流程请参考昇腾官方文档：[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)。

**注：目前vLLM-MindSpore插件的自定义算子仅支持动态图（PyNative Mode）场景。**

## 文件组织结构

接入自定义算子需要在vLLM-MindSpore插件项目的 `csrc` 目录下添加代码，目录结构如下：

```text
vllm-mindspore/
├── csrc/
│   ├── CMakeLists.txt             // 算子编译脚本
│   ├── ascendc/
│   │   ├── CMakeLists.txt         // AscendC 算子编译脚本
│   │   ├── adv_step_flash.h       // AscendC AdvanceStepFlashattn 算子声明
│   │   ├── adv_step_flash.c       // AscendC AdvanceStepFlashattn 算子实现
│   │   └── ...
│   └── module/
│       ├── module.h              // 公共模块注册头文件
│       ├── module.cpp            // 公共模块注册实现文件
│       ├── adv_step_flash.cpp    // 接入层代码，注册 AdvanceStepFlashattn 算子的Python模块接口
│       └── ...
└── vllm_mindspore/
    └── _custom_ops.py            // 封装自定义算子调用接口
```

- **`csrc/ascendc/`**：放置 AscendC 自定义算子的实现代码。
- **`csrc/module/`**：放置算子接入层代码，包括公共模块注册（`module.h`、`module.cpp`）和算子接入代码（如 `adv_step_flash.cpp`）。

## 接入流程

接入一个自定义算子，在算子实现方面，需在`csrc`目录中创建[算子接口定义](#算子接口声明)、[算子实现](#算子实现)与[算子接入](#算子接入)。在完成自定义算子初步的开发与接入后，可添加[算子调用接口](#算子调用接口)并进行[算子编译和测试](#算子编译和测试)。

### 算子接口声明

在 `csrc/ascendc/` 目录下创建头文件（如 `adv_step_flash.h`），以声明算子接口。内容参考[adv_step_flash.h](https://gitee.com/mindspore/vllm-mindspore/blob/r0.4.0/csrc/ascendc/adv_step_flash.h)：

```cpp
#ifndef VLLM_MINDSPORE_CSRC_ASCENDC_ADV_STEP_FLASH_H
#define VLLM_MINDSPORE_CSRC_ASCENDC_ADV_STEP_FLASH_H

extern void AdvStepFlashKernelEntry(uint32_t blockDims, void *l2ctrl, void *aclStream, uint8_t *sampledTokenIds,
                                    uint8_t *blockTables, uint8_t *seqLensInput, uint8_t *inputTokens,
                                    uint8_t *inputPositions, uint8_t *seqLensOut, uint8_t *slotMapping,
                                    int32_t num_seqs, int32_t block_size, int32_t block_tables_stride);

#endif  // VLLM_MINDSPORE_CSRC_ASCENDC_ADV_STEP_FLASH_H
```

### 算子实现

在 `csrc/ascendc/` 目录下创建实现文件（如 `adv_step_flash.c`），以实现算子的核心逻辑。内容参考[adv_step_flash.c](https://gitee.com/mindspore/vllm-mindspore/blob/r0.4.0/csrc/ascendc/adv_step_flash.c)：

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void adv_step_flash_impl(GM_ADDR sampledTokenIds, GM_ADDR blockTables,
                                                          GM_ADDR seqLensInput, GM_ADDR inputTokens,
                                                          GM_ADDR inputPositions, GM_ADDR seqLensOut,
                                                          GM_ADDR slotMapping, int32_t num_seqs, int32_t block_size,
                                                          int32_t block_tables_stride) {
  // AscendC 算子实现
}

#ifndef __CCE_KT_TEST__
void AdvStepFlashKernelEntry(uint32_t blockDims, void *l2ctrl, void *aclStream, uint8_t *sampledTokenIds,
                             uint8_t *blockTables, uint8_t *seqLensInput, uint8_t *inputTokens, uint8_t *inputPositions,
                             uint8_t *seqLensOut, uint8_t *slotMapping, int32_t num_seqs, int32_t block_size,
                             int32_t block_tables_stride) {
  adv_step_flash_impl<<<blockDims, l2ctrl, aclStream>>>(sampledTokenIds, blockTables, seqLensInput, inputTokens,
                                                        inputPositions, seqLensOut, slotMapping, num_seqs, block_size,
                                                        block_tables_stride);
}
#endif
```

### 算子接入

在 `csrc/module/` 目录下创建一个新的接入文件（如 `adv_step_flash.cpp`）。内容参考 [adv_step_flash.cpp](https://gitee.com/mindspore/vllm-mindspore/blob/r0.4.0/csrc/module/adv_step_flash.cpp)：

```cpp
#include "ms_extension/api.h"
#include "ascendc/adv_step_flash.h"
#include "module/module.h"

auto pyboost_adv_step_flash(int32_t num_seqs, int32_t num_queries, int32_t block_size,
                            ms::Tensor input_tokens,
                            ms::Tensor sampled_token_ids,
                            ms::Tensor input_positions,
                            ms::Tensor seq_lens,
                            ms::Tensor slot_mapping,
                            ms::Tensor block_tables) {
  // 使用 ms::PyboostRunner 调用你的算子
}

VLLM_MS_EXTENSION_MODULE(m) {
  m.def("advance_step_flashattn", &pyboost_adv_step_flash, "advance_step_flashattn", pybind11::arg("num_seqs"),
        pybind11::arg("num_queries"), pybind11::arg("block_size"), pybind11::arg("input_tokens"),
        pybind11::arg("sampled_token_ids"), pybind11::arg("input_positions"), pybind11::arg("seq_lens"),
        pybind11::arg("slot_mapping"), pybind11::arg("block_tables"));
}
```

上面`m.def()`接口的第一个参数`"advance_step_flashattn"`就是算子的Python接口名。

`module.h` 和 `module.cpp` 文件的作用是基于pybind11创建算子的Python模块。因为一个动态库内只能有一个 `PYBIND11_MODULE` ，为了让用户可以在一个文件内完成算子接入工作，vLLM-MindSpore插件提供了一个新的注册接口 `VLLM_MS_EXTENSION_MODULE` 宏。自定义算子动态库加载时，所有算子接口都会被自动注册到同一个Python模块中。

### 算子调用接口

vLLM-MindSpore插件的自定义算子被编译到了 `_C_ops.so` 里面。为了方便调用，可以在 `vllm_mindspore/_custom_ops.py` 添加一个调用接口。如果在算子调用前后需要做额外适配，也可以在这接口内实现。

```python
def advance_step_flashattn(num_seqs: int, num_queries: int, block_size: int,
                           input_tokens: torch.Tensor,
                           sampled_token_ids: torch.Tensor,
                           input_positions: torch.Tensor,
                           seq_lens: torch.Tensor,
                           slot_mapping: torch.Tensor,
                           block_tables: torch.Tensor) -> None:
    """Advance a step on Ascend for existing inputs for a multi-step runner"""
    from vllm_mindspore import _C_ops as c_ops
    c_ops.advance_step_flashattn(num_seqs=num_seqs,
                                 num_queries=num_queries,
                                 block_size=block_size,
                                 input_tokens=input_tokens,
                                 sampled_token_ids=sampled_token_ids,
                                 input_positions=input_positions,
                                 seq_lens=seq_lens,
                                 slot_mapping=slot_mapping,
                                 block_tables=block_tables)
```

这里通过 `import _C_ops` 即可导入自定义算子的Python模块。推荐在调用前再导入，未调用时不需要导入。

### 算子编译和测试

1. **代码集成**：将代码集成至vLLM-MindSpore插件项目。
2. **编译项目**：在项目代码根目录下执行 `pip install .` ，编译安装vLLM-MindSpore插件。
3. **测试算子接口**：通过 `_custom_ops` 调用算子接口。可以参考测试用例[test_custom_advstepflash.py](https://gitee.com/mindspore/vllm-mindspore/blob/r0.4.0/tests/st/python/test_custom_advstepflash.py)：

```python
from vllm_mindspore import _custom_ops as custom_ops

custom_ops.advance_step_flashattn(...)
```

## 自定义算子编译工程

当前MindSpore仅提供了一个 [CustomOpBuilder接口](https://www.mindspore.cn/docs/zh-CN/r2.7.2/api_python/ops/mindspore.ops.CustomOpBuilder.html) 用于在线编译自定义算子，接口内置了默认的编译和链接选项。vLLM-MindSpore插件基于MindSpore的自定义算子功能接入算子，并编译成动态库随包发布。下面是编译流程介绍：

### 算子扩展库模块

在 `setup.py` 中，vLLM-MindSpore插件添加了一个 `vllm_mindspore._C_ops` 扩展，并添加了相应的编译模块：

```python
ext_modules = [Extension("vllm_mindspore._C_ops", sources=[])],
cmdclass = {"build_ext": CustomBuildExt},
```

这里不需要指定 `sources` ，是因为vLLM-MindSpore插件通过CMake触发算子编译，自动收集了源文件。

### 算子编译流程

1. `CustomBuildExt` 调用CMake执行 `csrc/CMakeLists.txt` ，传入必要的环境变量，触发算子编译。
2. 通过 `ascendc/CMakeLists.txt` 调用AscendC编译器，编译 `ascendc` 目录内的算子源码。生成静态库 `ascendc_kernels_npu.a` 。
3. 递归收集cpp源文件列表 `SRC_FILES` 。
4. 生成临时脚本 `build_custom_with_ms.py` ，文件内调用 `mindspore.CustomOpBuilder` 编译算子接口。文件里也写入了源文件列表、头文件路径和静态库路径等信息。
5. 通过CMake的 `add_custom_target` 命令调用Python脚本编译自定义算子，生成 `_C_ops.so` 。
6. 将 `_C_ops.so` 重命名成Python包扩展模块的标准名称，如 `_C_ops.cpython-39-aarch64-linux-gnu.so` 。
