# AOT-Type Custom Operators(Ascend)

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/custom_program/operation/op_custom_ascendc.md)

## Overview

Custom operators of the AOT (Ahead-Of-Time) type adopt a pre-compilation approach, requiring network developers to manually implement the corresponding source code files for operator functions based on specific interfaces. The source code files need to be compiled into dynamic link libraries in advance, and then during network runtime, the framework will automatically invoke and execute the functions within these dynamic link libraries. The AOT-type custom operators support the Ascend C programming language on the Ascend platform, an efficient programming language specifically designed for operator development. This guide will start from the user's perspective and provide a detailed introduction to the development and usage process of custom operators based on Ascend C, including the following key steps:

1. **Custom Operator Development**: Using the Ascend C programming language, you can quickly develop custom operators, reducing development costs and improving development efficiency.
2. **Offline Compilation and Deployment**: After completing the operator development, perform offline compilation to ensure that the operator can run efficiently on the Ascend AI processor and deploy it.
3. **Using Custom Operators in MindSpore**: Integrate the compiled Ascend C custom operators into the MindSpore framework to enable their use in actual AI applications.

This chapter aims to help developers fully understand and master the entire lifecycle of Ascend C custom operators, from development to deployment, and to effectively utilize them in MindSpore.

## Custom Operator Development

The Ascend platform provides comprehensive tutorials for Ascend C operator development, helping developers to deeply understand and implement custom operators. The following are key development steps and resource links:

**Basic Tutorial**: Visit [Ascend C Operator Development](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html) and [Ascend C API List](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendcopapi/atlasascendc_api_07_0003.html) to get started.

**Operator Implementation**: Learn about [Custom Operator Development Based on Custom Operator Projects](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) to quickly understand the end-to-end process of custom operator development, with a focus on the implementation on the kernel side and the host side.

**Development Samples**: Ascend Community provides a wealth of [Ascend C Operator Development Samples](https://gitee.com/ascend/samples/tree/master/operator/ascendc), covering various types of operators, helping you quickly understand the practical application of operator development. You can also view the [AddCustom Custom Operator Development Sample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom), which simply shows the core work needed for custom operator development.

## Offline Compilation and Deployment

### Environment Preparation

Make sure you have the following conditions to use MindSpore's Ascend C custom operator offline compilation tool:

- **Ascend C Source Code**: Including the implementation of host-side and kernel-side custom operators.
- **MindSpore Installation**: Ensure that MindSpore version 2.3.0 or above is installed.
- **CMake**: CMake>=3.16.0

### Offline Compilation and Deployment

If you have already completed the compilation and deployment of the custom operator through the CANN custom operator compilation project, you can skip this step. MindSpore also provides a custom compilation tool. After you have developed your custom operator and prepared the kernel side and host side of the custom operator, you can follow the steps below to compile and deploy the custom operator.

1. **Obtain Compilation Tools**:
   Copy the `custom_compiler` tool directory from the MindSpore installation package to your working directory.

   ```shell
   cp -r {LOCATION}/mindspore/lib/plugin/ascend/custom_compiler {your_workspace}
   cd custom_compiler
   ```

2. **Execute Compilation Command**:
   Use the `python setup.py` command with the necessary parameters to compile custom operators.

   ```shell
   python setup.py
     --op_host_path={op_host_path}
     --op_kernel_path={op_kernel_path}
     --vendor_name={your_custom_name}
     --ascend_cann_package_path="/usr/local/Ascend/latest"
   ```

   **Parameter Description**:

   | Parameter               | Description                             | Default Value | Required |
   |-------------------|----------------------------------|--------|----------|
   | `--op_host_path` `-o` | Host-side operator implementation path | None | Yes |
   | `--op_kernel_path` `-k`| Kernel-side operator implementation path | None | Yes |
   | `--vendor_name`   | Custom operator vendor name | "customize" | No |
   | `--ascend_cann_package_path` | CANN software package installation path | None | No |
   | `--install_path`  | Custom operator installation path | None | No |
   | `-i`              | Install the custom operator to the path specified by `--install_path`; if not specified, install to the path designated by the environment variable `ASCEND_OPP_PATH`. | Not set | No |
   | `-c`              | Delete compilation logs and result files | Not set | No |

3. **Install Custom Operators**:
   After compilation, a `CustomProject/build_out` folder containing the compilation results of the custom operators will be generated in the current directory. You can choose to install manually or use the compiled operators by setting environment variables.

   **Manual Installation**:

   ```shell
   bash build_out/*.run
   ```

   **Set Environment Variables**:
   Find the path whose name is specified by `--vendor_name` in the `build_out` directory and add it to `ASCEND_CUSTOM_OPP_PATH`, for example:

   ```shell
   export ASCEND_CUSTOM_OPP_PATH={build_out_path}/build_out/_CPack_Package/Linux/External/custom_opp_euleros_aarch64.run/packages/vendors/{your_custom_name}:$ASCEND_CUSTOM_OPP_PATH
   ```

## Using Custom Operators in MindSpore

MindSpore's custom operator interface is [ops.Custom](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Custom.html). Detailed interface instructions can be found at [ops.Custom](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Custom.html). This article focuses on how to use [ops.Custom](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Custom.html) to access Ascend C custom operators.

### Environment Preparation

Before you start, please ensure that you have completed the development, compilation, and deployment of Ascend C custom operators. You can prepare the environment by installing the custom operator package or setting the environment variable `ASCEND_CUSTOM_OPP_PATH`.

### Parameter Description

```python
ops.Custom(func, bprop=None, out_dtype=None, func_type='aot', out_shape=None, reg_info=None)
```

- `func`(str): Name of the custom operator.
- `out_shape`(Union[function, list, tuple])：Output shape or shape inference function. Default value: `None`.
- `out_dtype` (Union[function, [mindspore.dtype](https://www.mindspore.cn/docs/en/br_base/api_python/mindspore/mindspore.dtype.html#mindspore.dtype), list, tuple])：Output type or type inference function. Default value: `None`.
- `func_type`(str)：Function type of the custom operator. For Ascend C custom operators, specify `func_type="aot"`.
- `bprop`(function)：Backpropagation function for the custom operator. Default value: `None`.
- `reg_info`(Union[str, dict, list, tuple])：Registration information for the custom operator. Default value: `None`. Ascend C custom operators do not need to pass this parameter and can use the default value.

**Scenario Limitations**： Currently, dynamic graphs and static graphs in O2 mode only support input and output of Tensor types. Static graphs in O0/O1 modes have no type restrictions. For dynamic graph scenarios with Ascend C custom operators, it is recommended to use [Custom Operators for Dynamic Graph Scenarios](https://www.mindspore.cn/tutorials/en/br_base/custom_program/operation/op_custom_pyboost.html).

### Simple Example

Through the above parameter description, when using Ascend C custom operators, it is necessary to focus on three core parameters: `func`, `out_shape`, and `out_dtype`. Below is a simple example to help users intuitively understand the usage of Ascend C custom operators in the MindSpore framework.

First, define the custom operator using the `ops.Custom` primitive and pass the required parameters. The operator name `func` is specified as `aclnnCast`, `out_shape` is passed a shape inference function implemented via a lambda function, and the output shape of this operator is the same as the shape of the first input. `out_dtype` is directly specified as MindSpore's built-in data type `mstype.float32`. The implementation of `out_shape` and `out_dtype` will be detailed in subsequent sections. After defining the custom operator, use it by passing all valid inputs to the operator. For example, in the use case's construct, call `self.custom_cast` and pass two parameters: the original data x (Tensor) and the target data type dst_type (mindspore.dtype).

```python
import numpy as np
import mindspore as ms
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context, Tensor, jit
import mindspore.common.dtype as mstype


class CustomNet(Cell):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.custom_cast = ops.Custom(func="aclnnCast", out_shape=lambda x, dst_type: x,
                                      out_dtype=mstype.float32,
                                      func_type="aot",
                                      bprop=None, reg_info=None)

    jit(backend="ms_backend")
    def construct(self, x, dst_type):
        res = self.custom_cast(x, dst_type)
        return res


context.set_context(mode=ms.GRAPH_MODE)

x = np.random.randn(1280, 1280).astype(np.float16)
net = CustomNet()
output = net(ms.Tensor(x), mstype.float32)
assert output.asnumpy().dtype == 'float32'
assert output.asnumpy().shape == (1280, 1280)
```

You can view the [custom operator test cases](https://gitee.com/mindspore/mindspore/tree/br_base/tests/st/graph_kernel/custom/custom_ascendc) in the MindSpore repository to obtain Ascend C custom operator test cases for more data types and usage scenarios. The sample project directory structure is as follows:

```text
.
├── compile_utils.py                // Custom operator compilation utility file
├── infer_file
│   ├── custom_cpp_infer.cc         // C++-side infer shape and infer type file for custom operators
│   └── custom_aot_extra.h          // Header file dependency for custom operator infer shape compilation
├── op_host                         // Custom operator source code op_host
│   ├── add_custom.cpp
│   └── add_custom_tiling.h
├── op_kernel                       // Custom operator source code op_kernel
│   └── add_custom.cpp
├── test_compile_custom.py          // Custom operator compilation test case
├── test_custom_aclnn.py            // Custom operator usage sample
├── test_custom_ascendc.py          // Custom operator startup script, including compilation and execution end-to-end process
├── test_custom_level0.py           // Custom operator combination scenario simple example
├── test_custom_multi_output.py     // Custom operator multi-output scenario usage example
├── test_custom_multi_type.py       // Custom operator different input types usage example, can be used as a reading entry point
├── test_custom_tensor_list.py      // Custom operator dynamic input/output usage example
└── test_custom_utils.py            // Internal test file
```

### Infer Shape/Type

To determine the type and size of the custom operator's output, pass the operator's shape and type through the `out_shape` and `out_dtype` parameters. These two parameters usually need to be inferred. Users can pass in determined shapes and types or use functions to infer the output shape and type. This section mainly explains how to infer the output shape and type through functions.

**Note**

- There are two ways to implement shape and type inference functions: Python-side and C++-side.
- Python-side inference is more user-friendly, but in dynamic graph scenarios, C++-side inference offers higher performance.
- For dynamic shape and value-dependent scenarios, shape inference can only be performed on the C++ side.

#### Python-side Infer Shape/Type

The input to the inference function is the shape or type of the custom operator's input, and the output is the inference result, i.e., the output shape or type. Below are some examples of inference functions.

- Infer function for scenarios where the output type and shape are the same as the input.

   ```python
   from mindspore import ops


   # The Add operator has two inputs, and the output shape is the same as the input shape
   def add_infer_shape(x, _):
       return x


   # The Add operator has two inputs, and the output type is the same as the input type
   def add_infer_type(x, _):
       return x


   # Define the custom operator
   custom_add = ops.Custom(func="aclnnAdd", out_shape=add_infer_shape, out_dtype=add_infer_type, func_type="aot")

   # For simple infer shape or infer type, you can also use a lambda function directly
   custom_add = ops.Custom(func="aclnnAdd", out_shape=lambda x, y: x, out_dtype=lambda x, y: x, func_type="aot")
   ```

- Infer function for scenarios where the output shape is calculated based on the input shape.

   ```python
   from mindspore import ops
   import mindspore.common.dtype as mstype


   # The output shape of the operator is calculated based on the input shape, and the output is of tuple type
   def msda_infer_shape_1(v_s, vss_v, vlsi_s, sl_s, aw_s):
       return [v_s[0], sl_s[1], v_s[2] * v_s[3]]


   # The output shape of the operator is calculated based on the input shape, and the output is of list type
   def msda_infer_shape_2(v_s, vss_v, vlsi_s, sl_s, aw_s):
       return (v_s[0], sl_s[1], v_s[2] * v_s[3])


   # Output shape is inferred through a regular function, and the output type is directly specified
   custom_msda = ops.Custom(func="aclnnMultiScaleDeformableAttn", out_shape=msda_infer_shape_1,
                            out_dtype=mstype.float32, func_type="aot")

   # Output shape and type are inferred through a lambda function
   custom_msda = ops.Custom(func="aclnnMultiScaleDeformableAttn",
                            out_shape=lambda v_s, vss_s, vlsi_s, sl_s, aw_s: (v_s[0], sl_s[1], v_s[2] * v_s[3]),
                            out_dtype=lambda v_s, vss_s, vlsi_s, sl_s, aw_s: v_s, func_type="aot")
   ```

- Infer function for multi-output and dynamic output scenarios.

   ```python
   from mindspore import ops
   import mindspore.dtype as mstype


   def msda_grad_infer_shape_1(v_s, vss_s, vlsi_s, sl_s, aw_s, go_s):
       out1 = v_s
       out2 = sl_s
       out3 = [sl_s[0],
               sl_s[1],
               sl_s[2],
               sl_s[3],
               sl_s[4]]
       return [out1, out2, out3]


   def msda_grad_infer_shape_2(v_s, vss_s, vlsi_s, sl_s, aw_s, go_s):
       out1 = v_s
       out2 = sl_s
       out3 = [sl_s[0],
               sl_s[1],
               sl_s[2],
               sl_s[3],
               sl_s[4]]
       return (out1, out2, out3)


   custom_msda_grad = ops.Custom(
       func="aclnnMultiScaleDeformableAttnGrad", out_shape=msda_grad_infer_shape_1,
       out_dtype=[mstype.float32, mstype.float32, mstype.float32],
       func_type="aot")

   custom_msda_grad = ops.Custom(
       func="aclnnMultiScaleDeformableAttnGrad", out_shape=msda_grad_infer_shape_2,
       out_dtype=(mstype.float32, mstype.float32, mstype.float32),
       func_type="aot")

   ```

**Precautions**

- In the infer shape function, avoid changing the type of the shape value during the calculation process to ensure it remains of type int. For example, division operations may result in float values, which can cause shape conversion failures.
- In multi-output and dynamic output scenarios, if both shape and type are inferred on the Python side, ensure that the return types of both are consistent, either both being lists or both being tuples.

#### C++-side Infer Shape/Type

If using a C++ inference function, set the parameter `func` to the combination of the inference function file path and the operator name, separated by a colon. When defining the operator, set `out_shape` or `out_dtype` to `None`.

```python
# The shape and type inference functions are implemented in the ./infer_file/add_custom_infer.cc file
ops.Custom(func="./infer_file/add_custom_infer.cc:AddCustom", out_shape=None, out_dtype=None, func_type="aot")

# The shape inference function is implemented in the ./infer_file/add_custom_infer.cc file, and the type inference function is implemented via a lambda function on the Python side
ops.Custom(func="./infer_file/add_custom_infer.cc:AddCustom", out_shape=None, out_dtype=lambda x, y: x, func_type="aot")
```

**Infer Shape Function Prototype**

```cpp
extern "C" std::vector<int64_t> FuncNameInferShape(int *ndims, int64_t **shapes, AotExtra *extra)

extern "C" std::vector<std::vector<int64_t>> FuncNameInferShape(int *ndims, int64_t **shapes, AotExtra *extra)
```

Here, the function name `FuncName` is the operator name. For single-output, the return type is `std::vector<int64_t>`. For multi-output or dynamic output, the return type is `std::vector<std::vector<int64_t>>`, which represents the output shape. The parameter list is as follows:

- ndims (int \*): Array of input shape dimensions.
- shapes (int64_t \*\*): Array of input shapes.
- extra (AotExtra \*): Used for extending custom operators with attributes. The `AotExtra` type is defined in the MindSpore-provided header file [custom_aot_extra.h](https://gitee.com/mindspore/mindspore/blob/br_base/tests/st/graph_kernel/custom/aot_test_files/custom_aot_extra.h).

**Infer Type Function Prototype**

```cpp
extern "C" TypeId FuncNameInferType(std::vector<TypeId> type_ids, AotExtra *extra)

extern "C" std::vector<TypeId> FuncNameInferType(std::vector<TypeId> type_ids, AotExtra *extra)
```

Here, the function name `FuncName` is the operator name. For single-output, the return type is `TypeId`. For multi-output and dynamic output, the return type is `std::vector<TypeId>`, which represents the output type. The parameter list is as follows:

- type_ids (std::vector<TypeId>): Array of input TypeId.
- extra (AotExtra \*): Used for extending custom operators with attributes, consistent with the parameters of the shape inference function.

**C++ Inference Function Sample**

- Inference of output shape and type through input shape and type.

   C++ inference function file add_infer.cc

   ```cpp
   #include <vector>
   #include <stdint.h>
   #include "custom_aot_extra.h"
   enum TypeId : int {
   };

   extern "C" std::vector<int64_t> aclnnAddInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
      std::vector<int64_t> output_shape;
      // Get the size of the shape of the 0th input
      auto input0_size = ndims[0];
      // The output shape is the same as the size of the shape of the 0th input
      for (size_t i = 0; i < input0_size; i++) {
      output_shape.push_back(shapes[0][i]);
      }
      return output_shape;
   }

   extern "C" TypeId aclnnAddInferType(std::vector<TypeId> type_ids, AotExtra *extra) {
      // The output type is the same as the type of the 0th input
      return type_ids[0];
   }
   ```

   Custom operator script file custom.py

   ```python
   # Define the custom operator, pass the path of the C++ inference function to func, and set out_shape and out_dtype parameters to None
   custom_add = ops.Custom(func="./add_infer.cc:aclnnAdd", out_shape=None, out_dtype=None, func_type="aot")
   ```

- Scenario where output shape depends on specific values.

   In cases where the output shape depends on specific values rather than just the input shape, the current parameters of both Python-side and C++-side inference interfaces are the input shapes. To obtain specific values, you can use the `add_prim_attr` interface to set the values as attributes of the custom operator's primitive. During C++ inference shape, the value can be obtained through the `extra` parameter. Below is an example of output shape depending on specific values.

   C++ inference function file moe_infer.cc

   ```cpp
   #include <vector>
   #include <stdint.h>
   #include "custom_aot_extra.h"
   extern "C" std::vector<std::vector<int64_t>> MoeSoftMaxTopkInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
      std::vector<std::vector<int64_t>> res_output_shape;
      std::vector<int64_t> out1_shape;
      // The 0th dimension of the output shape is the same as the 0th dimension of the 0th input
      out1_shape.emplace_back(shapes[0][0]);
      // The 1st dimension of the output shape is obtained from the attribute value
      out1_shape.emplace_back(extra->Attr<int64_t>("attr_k"));
      // The operator has two outputs with the same shape
      res_output_shape.emplace_back(out1_shape);
      res_output_shape.emplace_back(out1_shape);
      return res_output_shape;
   }
   ```

   Custom operator script file custom.py

   ```python
   moe_softmax_topk_custom = ops.Custom(func="./infer_file/moe_infer.cc:MoeSoftMaxTopk", out_shape=None,
                                        out_dtype=[mstype.float32, mstype.int32], func_type="aot")
   # Add the dependent value to the attributes, which can be obtained through the attributes during the infer stage
   moe_softmax_topk_custom.add_prim_attr("attr_k", 2)
   ```

- Scenario with dynamic input (TensorList)

   If the input is a TensorList, since the infer interface parameters are Tensor shapes, the framework will expand the TensorList. In the infer shape function, ensure correct indexing. For example, the infer shape function of the Concat operator is as follows.

   C++ inference function file concat_infer.cc

   ```cpp
   #include <vector>
   #include <stdint.h>
   #include "custom_aot_extra.h"
   extern "C" std::vector<int64_t> aclnnCatInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
     std::vector<int64_t> output_shape;
     auto input0_size = ndims[0];
     auto axis = extra->Attr<int64_t>("attr_axis");
     for (size_t i = 0; i < input0_size; i++) {
         if(i==axis){
             output_shape[i] = shapes[0][i] + shapes[1][i];
         }else{
            output_shape.emplace_back(shapes[0][i]);  
         }
     }
     return output_shape;
   }
   ```

  Custom operator script file custom.py

   ```python
   class CustomNetConcat(Cell):
       def __init__(self):
           self.axis = 1
           self.custom_concat = ops.Custom(func="./infer_file/concat_infer.cc:aclnnCat", out_shape=None,
                                           out_dtype=lambda x, _: x[0], func_type="aot")
           self.custom_concat.add_prim_attr("attr_axis", self.axis)

       def construct(self, x1, x2):
           res = self.concat((x1, x2), self.axis)
           return res
   ```

## Common Issues

1. Compilation cannot find the header file `"register/tilingdata_base.h"`

    ```text
    -- The C compiler identification is GNU 7.3.0
    -- The CXX compiler identification is GNU 7.3.0
    -- Check for working C compiler: /usr/bin/cc
    -- Check for working C compiler: /usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Opbuild generating sources
    build ops lib info:
    build ops lib error: In file included from /home/samples/operator/AddCustomSample/FrameworkLaunch/AddCustom/op_host/add_custom.cpp:2:0:
    /home/samples/operator/AddCustomSample/FrameworkLaunch/AddCustom/op_host/add_custom_tiling.h:6:10: fatal error: register/tilingdata_base.h: No such file or directory
     #include "register/tilingdata_base.h"
              ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    compilation terminated.
    CMake Error at cmake/func.cmake:27 (message):
      opbuild run failed!
    Call Stack (most recent call first):
      op_host/CMakeLists.txt:4 (opbuild)
    -- Configuring incomplete, errors occurred!
    See also "/home/samples/operator/AddCustomSample/FrameworkLaunch/AddCustom/build_out/CMakeFiles/CMakeOutput.log".
    gmake: *** No rule to make target 'package'.  Stop.
    ```

    **Solution**: This is usually because the CANN package path is not set correctly, causing the compilation project to not find the dependency files. Check whether the `--cann_package_path` option has been passed and whether the path of this option is correct, and confirm whether the corresponding Ascend software development kit has been correctly installed.

2. Custom operator execution reports the following error:

    ```text
    [INFO] GE(45311,python):2024-05-24-21:17:48.149.016 [ir_data_type_symbol_store.cc:177]45311 SetInputSymbol:Create symbol ge::TensorType::ALL() for Required input x
    [INFO] GE(45311,python):2024-05-24-21:17:48.149.028 [ir_data_type_symbol_store.cc:177]45311 SetInputSymbol:Create symbol ge::TensorType::ALL() for Required input y
    [INFO] GE(45311,python):2024-05-24-21:17:48.149.037 [ir_data_type_symbol_store.cc:223]45311 SetOutputSymbol:Create symbol expression ge::TensorType::ALL() for Required output z
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.068 [ir_definitions_recover.cc:106]45311 AppendIrDefs: ErrorNo: 4294967295(failed) [COMP][PRE_OPT]In the current running version, the order or type of operator[Default/Custom-op0AddCustom][AddCustom] inputs may have changed, ir_def.inputs[0] is [z, 0], ir_inputs_in_node[0] is [output, 0], ir_def.inputs is [[z, 0], ], ir_inputs_in_node is [[output, 0], ]
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.083 [ir_definitions_recover.cc:184]45311 RecoverOpDescIrDefinition: ErrorNo: 4294967295(failed) [COMP][PRE_OPT]recover ir outputs failed.
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.092 [ir_definitions_recover.cc:230]45311 RecoverIrDefinitions: ErrorNo: 4294967295(failed) [COMP][PRE_OPT][Recover][NodeIrDefinitions] failed, node[Default/Custom-op0AddCustom], type[AddCustom]
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.111 [graph_prepare.cc:2282]45311 InferShapeForPreprocess: ErrorNo: 4294967295(failed) [COMP][PRE_OPT][Recover][IrDefinitions] failed, graph[kernel_graph0]
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.129 [graph_prepare.cc:1769]45311 FormatAndShapeProcess: ErrorNo: 1343242270(Prepare Graph infershape failed) [COMP][PRE_OPT][Call][InferShapeForPreprocess] Prepare Graph infershape failed
    [INFO] GE(45311,python):2024-05-24-21:17:48.149.137 [graph_prepare.cc:2008][EVENT]45311 PrepareDynShape:[GEPERFTRACE] The time cost of Prepare::FormatAndShapeProcess is [263] micro second.
    [INFO] GE(45311,python):2024-05-24-21:17:48.149.143 [graph_prepare.cc:2008]45311 PrepareDynShape:[GEPERFTRACE] The time cost of Prepare::FormatAndShapeProcess is [263] micro second.
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.150 [graph_prepare.cc:2008]45311 PrepareDynShape: ErrorNo: 1343242270(Prepare Graph infershape failed) [COMP][PRE_OPT][Process][Prepare_FormatAndShapeProcess] failed
    [INFO] GE(45311,python):2024-05-24-21:17:48.149.158 [graph_manager.cc:1083][EVENT]45311 PreRunOptimizeOriginalGraph:[GEPERFTRACE] The time cost of GraphManager::stages.preparer.PrepareDynShape is [399] micro second.
    [INFO] GE(45311,python):2024-05-24-21:17:48.149.164 [graph_manager.cc:1083]45311 PreRunOptimizeOriginalGraph:[GEPERFTRACE] The time cost of GraphManager::stages.preparer.PrepareDynShape is [399] micro second.
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.170 [graph_manager.cc:1083]45311 PreRunOptimizeOriginalGraph: ErrorNo: 1343242270(Prepare Graph infershape failed) [COMP][PRE_OPT][Process][GraphManager_stages.preparer.PrepareDynShape] failed
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.179 [graph_manager.cc:3817]45311 OptimizeGraph: ErrorNo: 1343242270(Prepare Graph infershape failed) [COMP][PRE_OPT][Run][PreRunOptimizeOriginalGraph] failed for graph:kernel_graph0, session_id:0
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.187 [pne_model_builder.cc:125]45311 OptimizeGraph: ErrorNo: 4294967295(failed) [COMP][PRE_OPT][Optimize][Graph] failed, graph = kernel_graph0, engine = NPU
    [ERROR] GE(45311,python):2024-05-24-21:17:48.149.207 [graph_manager.cc:1286]45311 PreRun: ErrorNo: 4294967295(failed) [COMP][PRE_OPT][Build][Model] failed, session_id:0, graph_id:1.
    [INFO] GE(45311,python):2024-05-24-21:17:48.149.217 [rt_context_util.cc:92]45311 DestroyRtContexts:Destroy 2 rts contexts for graph 1 of session 0.
    [INFO] RUNTIME(45311,python):2024-05-24-21:17:48.149.234 [stream.cc:436] 45311 FreeLogicCq: Return(0), threadIdentifier(281473877712448), devId(64), tsId(0), cqId(65535), isFastCq(0).
    [INFO] RUNTIME(45311,python):2024-05-24-21:17:48.149.244 [stream.cc:682] 45311 FreeStreamId: Free stream_id=1600.
    ```

    **Solution**: The above problem is generally reported in graph mode, and the cause is the inconsistency between the registration information of the custom operator and the prototype definition in the implementation of the custom operator. For example, the prototype definition in the operator implementation is:

    ```cpp
    class AddCustom : public OpDef {
     public:
      explicit AddCustom(const char *name) : OpDef(name) {
        this->Input("x")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32})
          .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
      }
    };
    ```

    And the registration information when using the operator is:

    ```python
    reg_info = CustomRegOp("AddCustom")
                .input(0, "x", "required")
                .input(1, "y", "required")
                .output(0, "output", "required")
                .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default)
                .target("Ascend")
                .get_op_info()
    ```

    The names of the outputs in the two operator information are inconsistent; the operator prototype is named `z`, while in `reg_info` it is named `output`. Pay attention to such small differences that can cause errors.

3. Unsupported operator type

   ```text
   [ERROR] KERNEL(3915621,fffe47fff1e0,python):2024-06-26-16:57:38.219.508 [mindspore/ccsrc/plugin/device/ascend/kernel/acl/acl_kernel/custom_op_kernel_mod.cc:132] Launch] Kernel launch failed, msg:
   Acl compile and execute failed, op_type :aclnnAddCustom
   ----------------------------------------------------------
   Ascend Error Message:
   ----------------------------------------------------------
   EZ3003: 2024-06-26-16:57:38.215.381 No supported Ops kernel and engine are found for [aclnnAddCustom1], optype [aclnnAddCustom].
       Possible Cause: The operator is not supported by the system. Therefore, no hit is found in any operator information library.
       Solution: 1. Check that the OPP component is installed properly. 2. Submit an issue to request for the support of this operator type.
       TraceBack (most recent call last):
       Assert ((SelectEngine(node_ptr, exclude engines, is_check support success, op_info)) == ge::SUCCESS) failed[FUNC:operator()][FILE:engine place.cc][LINE:144]
       build graph failed, graph id:0, ret:-1[FUNC:BuildModelwithGraphId][FILE:ge_generator.cc][LINE:1608]
       [Build][singleOpModeT]call ge interface generator.BuildSingleOpModel failed. ge result = 4294967295[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:161]
       [Build][Op]Fail to build op model[FUNC:ReportInnerError][FILE:log inner.cpp][LINE:145]
       build op model failed, result = 500002[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
   (Please search "CANN Common Error Analysis" at https://www.mindspore.cn for error code description)
   ---------------------------------------------------------
   - C++ Call Stack:(For framework developers)
   ---------------------------------------------------------
   mindspore/ccsrc/transform/acl_ir/acl utils.cc:379 Run
   [ERROR] DEVICE(3915621,fffe47fff1e0,python):2024-06-26-16:57:38.219.637 [mindspore/ccsrc/plugin/device/ascend/hal/hardware/ge kernel executor.cc:1169] LaunchKernel] Launch kernel failed, kernel
   full name: Default/Custom-op0
   Traceback (most recent call last):
   File "/home/jenkins0/dyp/mindspore_custom/tests/st/ops/graph_kernel/custom/custom ascendc/test add.py", Line 58, in <module>
       out = net(Tensor(x), Tensor(y), Tensor(z))
   File "/home/jenkinsO/.conda/envs/dyp_py37_temp/Lib/python3.7/site-packages/mindspore/nn/cell.py", line 721, in _call_
   raise err
       File "/home/jenkinsO/.conda/envs/dyp_py37_temp/lib/python3.7/site-packages/mindspore/nn/cell.py", Line 718, in _call
   pynative_executor.end_graph(self, output, *args, **kwargs)
   File "/home/jenkinsO/.conda/envs/dyp_py37_temp/lib/python3.7/site packages/mindspore/common/api.py", Line 1557, in end_graph
       self._executor.end_graph(obj, output, *args, *(kwargs.values ( ) ) )
   RuntimeError: Launch kernel failed, name:Default/Custom-op0
   ```

   **Solution**: From the error log analysis, the user specified that `AddCustom` should use `aclnn`, but an error occurred in the aclop process, indicating that no corresponding symbol for `aclnn` was found, and the default aclop was used instead. If this happens, please first check whether the environment configuration is correct, including whether the custom operator installation package is correctly installed or the environment variable `ASCEND_CUSTOM_OPP_PATH` for the custom operator is correctly specified. Open the info log, filter the logs of the `op_api_convert.h` file, and check whether the symbols are correctly loaded.