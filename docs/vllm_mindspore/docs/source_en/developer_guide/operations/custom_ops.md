# Custom Operator Integration

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/developer_guide/operations/custom_ops.md)

When the built-in operators do not meet your requirements, you can use MindSpore's custom operator functionality to integrate your operators.

This document would introduce how to integrate a new custom operator into the vLLM-MindSpore Plugin project, with the **`advance_step_flashattn`** operator as an example. The focus here is on the integration process into vLLM-MindSpore Plugin. For the details of custom operator development, please refer to the official MindSpore tutorial: [CustomOpBuilder-Based Custom Operators](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/op_customopbuilder.html), and for AscendC operator development, see the official Ascend documentation: [Ascend C Operator Development](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html).

**Note: Currently, custom operators in vLLM-MindSpore Plugin are only supported in PyNative Mode.**

## File Structure

The directory `csrc` contains declaration and implementation of operations:

```text
vllm-mindspore/
├── csrc/
│   ├── CMakeLists.txt             // Operator build script
│   ├── ascendc/
│   │   ├── CMakeLists.txt         // AscendC operator build script
│   │   ├── adv_step_flash.h       // AscendC AdvanceStepFlashattn operator declaration
│   │   ├── adv_step_flash.c       // AscendC AdvanceStepFlashattn operator implementation
│   │   └── ...
│   └── module/
│       ├── module.h              // Common module registration header
│       ├── module.cpp            // Common module registration implementation
│       ├── adv_step_flash.cpp    // Integration layer code (Python interface registration)
│       └── ...
└── vllm_mindspore/
    └── _custom_ops.py            // Wrapper for custom operator call interface
```

- **`csrc/ascendc/`**: Contains AscendC custom operator implementation code.
- **`csrc/module/`**: Contains operator integration layer code, including common module registration (`module.h`, `module.cpp`) and operator-specific integration (e.g., `adv_step_flash.cpp`).

## Integration Process

To integrate a custom operator, users need to create [Operator Interface Declaration](#operator-interface-declaration), [Operator Implementation](#operator-implementation) and [Operator Integration](#operator-integration) in the directory `ops/ascendc/`. After the initial development and integration of the custom operator, user can add [Operator Interface](#operator-interface) and do [Operator Compilation and Testing](#operator-compilation-and-testing) after declaration and implementation.

### Operator Interface Declaration

Create a header file in `csrc/ascendc/` to declare the operator function and related interfaces. Refer to [adv_step_flash.cpp](https://atomgit.com/mindspore/vllm-mindspore/blob/master/csrc/ascendc/adv_step_flash.h):

```cpp
#ifndef VLLM_MINDSPORE_CSRC_ASCENDC_ADV_STEP_FLASH_H
#define VLLM_MINDSPORE_CSRC_ASCENDC_ADV_STEP_FLASH_H

extern void AdvStepFlashKernelEntry(uint32_t blockDims, void *l2ctrl, void *aclStream, uint8_t *sampledTokenIds,
                                    uint8_t *blockTables, uint8_t *seqLensInput, uint8_t *inputTokens,
                                    uint8_t *inputPositions, uint8_t *seqLensOut, uint8_t *slotMapping,
                                    int32_t num_seqs, int32_t block_size, int32_t block_tables_stride);

#endif  // VLLM_MINDSPORE_CSRC_ASCENDC_ADV_STEP_FLASH_H
```

### Operator Implementation

Create an implementation file in `csrc/ascendc/` for the core logic. Refer to [adv_step_flash.cpp](https://atomgit.com/mindspore/vllm-mindspore/blob/master/csrc/ascendc/adv_step_flash.c):

```cpp
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void adv_step_flash_impl(GM_ADDR sampledTokenIds, GM_ADDR blockTables,
                                                          GM_ADDR seqLensInput, GM_ADDR inputTokens,
                                                          GM_ADDR inputPositions, GM_ADDR seqLensOut,
                                                          GM_ADDR slotMapping, int32_t num_seqs, int32_t block_size,
                                                          int32_t block_tables_stride) {
  // AscendC operator implementation
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

### Operator Integration

Create an integration file in `csrc/module/`. Refer to [adv_step_flash.cpp](https://atomgit.com/mindspore/vllm-mindspore/blob/master/csrc/module/adv_step_flash.cpp):

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
  // Use ms::PyboostRunner to call your operator
}

VLLM_MS_EXTENSION_MODULE(m) {
  m.def("advance_step_flashattn", &pyboost_adv_step_flash, "advance_step_flashattn", pybind11::arg("num_seqs"),
        pybind11::arg("num_queries"), pybind11::arg("block_size"), pybind11::arg("input_tokens"),
        pybind11::arg("sampled_token_ids"), pybind11::arg("input_positions"), pybind11::arg("seq_lens"),
        pybind11::arg("slot_mapping"), pybind11::arg("block_tables"));
}
```

In the above, the first parameter `"advance_step_flashattn"` in `m.def()` is the Python interface name for the operator.

The `module.h` and `module.cpp` files create the Python module for the operator based on pybind11. Since only one `PYBIND11_MODULE` is allowed per dynamic library, and to allow users to complete operator integration in a single file, vLLM-MindSpore Plugin provides a new registration macro `VLLM_MS_EXTENSION_MODULE`. When the custom operator dynamic library is loaded, all operator interfaces will be automatically registered into the same Python module.

### Operator Interface

The custom operator in vLLM-MindSpore Plugin is compiled into `_C_ops.so`. For convenient calls, user can add a call interface in `vllm_mindspore/_custom_ops.py`. If extra adaptation is needed before or after the operator call, user can implement it in this interface.

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

Here, importing `_C_ops` allows user to use the Python module for the custom operator. It is recommended to import it right before calling, so it is not imported unnecessarily.

### Operator Compilation and Testing

1. **Code Integration**: Merge the code into the vLLM-MindSpore Plugin project.
2. **Project Compilation**: Run `pip install .` in vllm-mindspore to build and install vLLM-MindSpore Plugin.
3. **Operator Testing**: Call the operator interface via `_custom_ops`. Refer to testcase [test_custom_advstepflash.py](https://atomgit.com/mindspore/vllm-mindspore/blob/master/tests/st/python/test_custom_advstepflash.py):

```python
from vllm_mindspore import _custom_ops as custom_ops

custom_ops.advance_step_flashattn(...)
```

## Custom Operator Compilation Project

Currently, MindSpore provides only a [CustomOpBuilder](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.CustomOpBuilder.html) interface for online compilation of custom operators, with default compilation and linking options built in. vLLM-MindSpore Plugin integrates operators based on MindSpore’s custom operator feature and compiles them into a dynamic library for package release. The following introduces the build process:

### Extension Module

In `setup.py`, vLLM-MindSpore Plugin adds a `vllm_mindspore._C_ops` extension and the corresponding build module:

```python
ext_modules = [Extension("vllm_mindspore._C_ops", sources=[])],
cmdclass = {"build_ext": CustomBuildExt},
```

There is no need to specify `sources` here because vLLM-MindSpore Plugin triggers the operator build via CMake, which automatically collects the source files.

### Building Process

1. `CustomBuildExt` calls CMake to execute `csrc/CMakeLists.txt`, passing necessary environment variables to trigger operator build.
2. Through `ascendc/CMakeLists.txt`, the AscendC compiler is called to compile the source code in the `ascendc` directory. The static library `ascendc_kernels_npu.a` is generated.
3. Recursively collect the list of cpp source files `SRC_FILES`.
4. Generate a temporary script `build_custom_with_ms.py`, which calls `mindspore.CustomOpBuilder` to build the operator interfaces. The script also contains the source file list, header paths, static library paths, etc.
5. Use CMake's `add_custom_target` command to call the Python script to build the custom operator, generating `_C_ops.so`.
6. Rename `_C_ops.so` to the standard Python package extension module name, such as `_C_ops.cpython-39-aarch64-linux-gnu.so`.
