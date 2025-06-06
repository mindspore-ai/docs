# Custom Operator Integration  

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/supported_features/operations/npu_ops.md)  

This document would introduce how to integrate a new custom operator into the vLLM MindSpore project, with the **`adv_step_flash`** operator as an example. The following sections would focus on the integration process, and user can refer to operator implementation introduction in official MindSpore tutorial: [Dynamic Graph Custom Operator Integration](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/op_customopbuilder.html).

For development, additional features can be extended based on project requirements. Implementation details can be referenced from [MindSpore Custom Operator Implementation](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/op_customopbuilder.html).  

## File Structure  

The directory `vllm_mindspore/ops` contains and declaration and implementation of operations:  

```text  
vllm_mindspore/ops/  
├── ascendc/  
│   ├── adv_step_flash.h      // AscendC AdvStepFlash operator declaration  
│   ├── adv_step_flash.c      // AscendC AdvStepFlash operator implementation  
│   └── ...  
├── module/  
│   ├── module.h              // Common module registration header  
│   ├── module.cpp            // Common module registration implementation  
│   ├── adv_step_flash.cpp    // Integration layer code (Python interface registration)  
│   └── ...  
```  

- **`ops/ascendc/`**: Contains AscendC custom operator implementation code.
- **`ops/module/`**: Contains operator integration layer code, including common module registration (`module.h`, `module.cpp`) and operator-specific integration (e.g., `adv_step_flash.cpp`).

## Integration Process  

To integrate a custom operator, user need to create [Operator Interface Declaration](#operator-interface-declaration), [Operator Implementation](#operator-implementation) and [Operator Integration](#operator-integration) in the directory `ops/ascendc/`. And do [Operator Compilation and Testing](#operator-compilation-and-testing) after declaration and implementation.  

### Operator Interface Declaration  

Create a header file (e.g., `my_custom_op.h`) in `ops/ascendc/` to declare the operator function and related interfaces:

```cpp  
#ifndef VLLM_MINDSPORE_OPS_ASCENDC_MY_CUSTOM_OP_H
#define VLLM_MINDSPORE_OPS_ASCENDC_MY_CUSTOM_OP_H

extern void MyCustomOpKernelEntry(uint32_t blockDims, void *l2ctrl, void *aclStream,
                                  uint8_t *input, uint8_t *output, int32_t param1, int32_t param2);

#endif  // VLLM_MINDSPORE_OPS_ASCENDC_MY_CUSTOM_OP_H
```  

### Operator Implementation  

Create an implementation file (e.g., `my_custom_op.c`) in `ops/ascendc/` for the core logic:  

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

### Operator Integration  

Create an integration file (e.g., `my_custom_op.cpp`) in `module/`. User can refer to `adv_step_flash.cpp` for more details about the integration:  

```cpp  
#include "ms_extension.h"
#include "ascendc/my_custom_op.h"
#include "module/module.h"

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

### Operator Compilation and Testing  

1. **Code Integration**: Merge the code into the vLLM MindSpore project.  
2. **Project Compilation**: Build and install the whl package containing the custom operator.  
3. **Operator Testing**: Invoke the operator in Python:

    ```python
    from vllm_mindspore import npu_ops
    import numpy as np
    import mindspore as ms

    input = ms.Tensor(np.array([1, 2, 3], dtype=np.int32))
    output = ms.Tensor(np.zeros_like(input))

    npu_ops.my_custom_op(10, 20, input, output)
    print("Output:", output)
    ```
