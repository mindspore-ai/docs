
# Ascend C Custom Operator Development and Usage

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/operation/op_custom_ascendc.md)

## Overview

CANN provides AI developers with the Ascend C programming language, which is a highly efficient programming language specifically designed for operator development. This guide will start from the user's perspective and provide a detailed introduction to the development and usage process of custom operators based on Ascend C, including the following key steps:

1. **Custom Operator Development**: Using the Ascend C programming language, you can quickly develop custom operators, reducing development costs and improving development efficiency.
2. **Offline Compilation and Deployment**: After completing the operator development, perform offline compilation to ensure that the operator can run efficiently on the Ascend AI processor and deploy it.
3. **Using Custom Operators in MindSpore**: Integrate the compiled Ascend C custom operators into the MindSpore framework to enable their use in actual AI applications.

This chapter aims to help developers fully understand and master the entire lifecycle of Ascend C custom operators, from development to deployment, and to effectively utilize them in MindSpore.

## Custom Operator Development

The Ascend platform provides comprehensive tutorials for Ascend C operator development, helping developers to deeply understand and implement custom operators. The following are key development steps and resource links:

**Basic Tutorial**: Visit [Ascend C Operator Development](https://www.hiascend.com/document/detail/en/canncommercial/700/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html) to obtain introductory knowledge.

**Operator Implementation**: Focus on learning [kernel-side operator implementation](https://www.hiascend.com/document/detail/en/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0024.html) and [host-side operator implementation](https://www.hiascend.com/document/detail/en/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0024.html), learning the core logic of device-side operator execution and the implementation method of host-side operator operations.

**Development Samples**: Ascend Community provides a wealth of [Ascend C Operator Development Samples](https://gitee.com/ascend/samples/tree/master/operator), covering various types of operators, helping you quickly understand the practical application of operator development. You can also view the [AddCustom Custom Operator Development Sample](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch/AddCustom), which simply shows the core work needed for custom operator development.

## Offline Compilation and Deployment

### Environment Preparation

Make sure you have the following conditions to use MindSpore's Ascend C custom operator offline compilation tool:

- **Ascend C Source Code**: Including the implementation of host-side and kernel-side custom operators.
- **MindSpore Installation**: Ensure that MindSpore version 2.3.0 or above is installed.

### Offline Compilation and Deployment

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
   | `-c`              | Whether to delete compilation logs and result files | Not set | No |

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

### Additional Information

This tool is based on the encapsulation of CANN's `msopgen` tool. You can also choose to use the native `msopgen` tool for compilation. For more information, please refer to [Creating Operator Projects Based on the msopgen Tool](https://www.hiascend.com/document/detail/en/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0023.html) and [Operator Compilation and Deployment](https://www.hiascend.com/document/detail/en/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0031.html).

## Using Custom Operators in MindSpore

### Environment Preparation

Before you begin, please make sure that the development, compilation, and deployment of Ascend C custom operators have been completed. You can prepare the usage environment by installing the custom operator package or setting environment variables.

### Using Custom Operators

MindSpore's custom operator interface is [ops.Custom](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.Custom.html). When using Ascend C custom operators, you need to set the parameter `func_type` to `"aot"` and provide the `func` parameter to specify the operator name. Taking the `AddCustom` operator as an example, there are several ways to use it:

- **aclnn**: Specify that the underlying operator uses the aclnn type. You need to add `aclnn` in front of the operator name, for example: `func="aclnnAddCustom"`.
- **C++ Inference**: If the operator's infer shape is implemented in C++, pass the path of the C++ infer shape file in the `func` and separate the operator name with `:`, for example: `func="add_custom_infer.cc:aclnnAddCustom"`.
- **TBE**: Specify that the underlying operator uses the TBE type, and set `func="AddCustom"`.

> For single-operator execution mode, it is recommended to use aclnn, including in PyNative mode or Graph mode where `jit_config` is set to `O0` or `O1`.

**aclnn Usage Example**:

```python
class AddCustomAclnnNet(Cell):
    def __init__(self, func, out_shape):
        super(AddCustomAclnnNet, self).__init__()
        aclnn_reg_info = CustomRegOp("aclnnAddCustom") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_add = ops.Custom(func=func, out_shape=out_shape, out_dtype=lambda x, _: x, func_type="aot", bprop=None,
                                     reg_info=aclnn_reg_info)

    def construct(self, x, y):
        res = self.custom_add(x, y)
        return res

context.set_context(device_target="Ascend", jit_config={"jit_level": "O0"})
x = np.ones([8, 2048]).astype(np.float16)
y = np.ones([8, 2048]).astype(np.float16)

# Implement the infer shape function through lambda and specify the underlying aclnn operator
net = AddCustomAclnnNet("aclnnAddCustom", lambda x, _: x)

# Use C++ to implement infer shape, pass the path of the infer shape in the func, and specify the underlying aclnn operator
net = AddCustomAclnnNet("./infer_file/add_custom_infer.cc:aclnnAddCustom", None)
```

**TBE Usage Example**

```python
class CustomNet(Cell):
    def __init__(self):
        super(CustomNet, self).__init__()
        aclop_reg_info = CustomRegOp("AddCustom") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_add = ops.Custom(func="AddCustom", out_shape=lambda x, _: x, out_type=lambda x, _: x,
                                     func_type="aot",reg_info=aclop_reg_info)

    def construct(self, x, y):
        res = self.custom_add(x, y)
        return res

x = np.ones([8, 2048]).astype(np.float16)
y = np.ones([8, 2048]).astype(np.float16)
net = CustomNet()
```

**Precautions**

1. **Name Consistency**: The operator name used in the registration information must be exactly the same as the name passed in the `func` parameter of `ops.Custom`. If the operator is specified to use `aclnn`, the operator name in the registration information must also be prefixed with `aclnn`.

2. **Input/Output Name Matching**: The names of the input and output parameters defined in the registration information must be exactly the same as those defined in the source code.

3. **Specification Consistency**: The specifications supported in the registration information must also match those defined in the source code.

4. **Execution Mode Limitations**: `aclnn` can only adopt a single-operator execution mode, set to PyNative mode or specify the Graph mode with `jit_config` set to `O0` or `O1`. For more information on `jit_config` configuration, refer to [set_context](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_context.html).

### Further Reading

- **Custom Operator Registration**: For more information on custom operator registration and the writing of backward functions, please refer to [Custom Operator Registration](https://www.mindspore.cn/tutorials/experts/en/master/operation/op_custom_adv.html).
- **AOT Custom Operators**: For the implementation of C++ shape inference functions and advanced usage of AOT type custom operators, please refer to [Advanced Usage of AOT Type Custom Operators](https://www.mindspore.cn/tutorials/experts/en/master/operation/op_custom_aot.html).
- **Sample Projects**: To learn more about the usage of Ascend C custom operators, you can check out the [Sample Projects](https://gitee.com/mindspore/mindspore/tree/master/tests/st/graph_kernel/custom/custom_ascendc).

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
        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("ascend910b");
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

   **Solution**: From the error log analysis, the user specified that `AddCustom` should use `aclnn`, but an error occurred in the aclop process, indicating that no corresponding symbol for `aclnn` was found, and the default aclop was used instead. In this case, please first check whether the environment configuration is correct, including whether the custom operator installation package is correctly installed or the environment variable `ASCEND_CUSTOM_OPP_PATH` for the custom operator is correctly specified. Open the info log, filter the logs of the `op_api_convert.h` file, and check whether the symbols are correctly loaded.