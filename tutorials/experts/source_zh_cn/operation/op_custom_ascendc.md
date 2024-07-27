# Ascend C自定义算子开发与使用指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3.0/tutorials/experts/source_zh_cn/operation/op_custom_ascendc.md)

## 概述

CANN为AI开发者提供了Ascend C编程语言，这是一款专为算子开发而设计的高效编程语言。本指南将从用户角度出发，详细介绍基于Ascend C的自定义算子开发和使用流程，包括以下关键步骤：

1. **自定义算子开发**：使用Ascend C编程语言，您可以快速开发自定义算子，降低开发成本并提高开发效率。
2. **离线编译与部署**：完成算子开发后，进行离线编译，确保算子可以在Ascend AI处理器上高效运行，并进行部署。
3. **MindSpore使用自定义算子**：将编译后的Ascend C自定义算子集成到MindSpore框架中，实现在实际AI应用中的使用。

本章内容旨在帮助开发者全面了解并掌握Ascend C自定义算子的整个生命周期，从开发到部署，再到在MindSpore中的有效利用。

## 自定义算子开发

昇腾平台提供了全面的Ascend C算子开发教程，帮助开发者深入理解并实现自定义算子。以下是关键的开发步骤和资源链接：

**基础教程**：访问[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/700/operatordev/Ascendcopdevg/atlas_ascendc_10_0001.html) 获取入门知识。

**算子实现**：重点学习[kernel侧算子实现](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0024.html) 和[host侧算子实现](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0024.html) ，学习设备端执行算子的核心逻辑和主机端进行算子操作的实现方法。

**开发样例**：昇腾社区提供了丰富的 [Ascend C算子开发样例](https://gitee.com/ascend/samples/tree/master/operator) ，覆盖了多种类型算子，帮助您快速理解算子开发的实际应用。也可以查看 [AddCustom自定义算子开发样例](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch/AddCustom) ，它简洁展示了一个自定义算子开发需要的核心工作。

## 离线编译与部署

### 环境准备

确保您已具备以下条件以使用MindSpore的Ascend C自定义算子离线编译工具：

- **Ascend C源码**: 包括host侧和kernel侧的自定义算子实现。
- **MindSpore安装**: 确保已安装2.3.0及以上版本的MindSpore。

### 离线编译与部署

1. **获取编译工具**：
   将MindSpore安装包中的`custom_compiler`工具目录拷贝到您的工作目录。

   ```shell
   cp -r {LOCATION}/mindspore/lib/plugin/ascend/custom_compiler {your_workspace}
   cd custom_compiler
   ```

2. **执行编译命令**：
   使用`python setup.py`命令并带上必要的参数来编译自定义算子。

   ```shell
   python setup.py
     --op_host_path={op_host_path}
     --op_kernel_path={op_kernel_path}
     --vendor_name={your_custom_name}
     --ascend_cann_package_path="/usr/local/Ascend/latest"
   ```

   **参数说明**：

   | 参数               | 描述                             | 默认值 | 是否必选 |
   |-------------------|----------------------------------|--------|----------|
   | `--op_host_path` `-o` | host侧算子实现路径               | 无     | 是       |
   | `--op_kernel_path` `-k`| kernel侧算子实现路径            | 无     | 是       |
   | `--vendor_name`   | 自定义算子厂商名称               | "customize" | 否 |
   | `--ascend_cann_package_path` | CANN软件包安装路径 | 无 | 否 |
   | `-c`              | 是否删除编译日志和结果文件       | 不设置 | 否       |

3. **安装自定义算子**：
   编译完成后，当前目录下将生成一个包含自定义算子编译结果的`CustomProject/build_out`文件夹，您可以选择手动安装或通过设置环境变量来使用编译后的算子。

   **手动安装**：

   ```shell
   bash build_out/*.run
   ```

   **设置环境变量**：
   找到`build_out`目录下通过`--vendor_name`指定名字的路径，并添加到`ASCEND_CUSTOM_OPP_PATH`，例如：

   ```shell
   export ASCEND_CUSTOM_OPP_PATH={build_out_path}/build_out/_CPack_Package/Linux/External/custom_opp_euleros_aarch64.run/packages/vendors/{your_custom_name}:$ASCEND_CUSTOM_OPP_PATH
   ```

### 补充说明

本工具基于CANN的`msopgen`工具封装，您也可以选择使用原生`msopgen`工具进行编译。有关更多信息，请参考[基于msopgen工具创建算子工程](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0023.html) 和
[算子编译部署](https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/operatordev/Ascendcopdevg/atlas_ascendc_10_0031.html)。

## MindSpore使用自定义算子

### 环境准备

在开始之前，请确保已完成Ascend C自定义算子的开发、编译和部署。您可以通过安装自定义算子包或设置环境变量来准备使用环境。

### 使用自定义算子

MindSpore的自定义算子接口为[ops.Custom](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/ops/mindspore.ops.Custom.html) ，
使用Ascend C自定义算子时，您需要设置参数`func_type`为`"aot"`，并提供`func`参数来指定算子名字。以`AddCustom`算子为例，存在以下几种使用方式：

- **aclnn**：指定算子底层使用aclnn类型，则需要在算子名字前加上`aclnn`，例如：`func="aclnnAddCustom"`
- **c++ infer**：算子的infer shape通过c++实现，则在func中传入c++的infer shape文件路径并用`:`隔开使用的算子名字，例如：`func="add_custom_infer.cc:aclnnAddCustom`
- **TBE**：指定算子底层使用TBE类型，则设置`func="AddCustom"`

> 单算子执行模式推荐使用aclnn，包括PyNative模式或Graph模式下`jit_config`为`O0`或`O1`。

**aclnn使用样例**：

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

# 通过lambda实现infer shape函数，并指定底层使用aclnn算子
net = AddCustomAclnnNet("aclnnAddCustom", lambda x, _: x)

# 使用c++实现infer shape，在func中传入infer shape的路径，并指定底层使用aclnn算子
net = AddCustomAclnnNet("./infer_file/add_custom_infer.cc:aclnnAddCustom", None)
```

**TBE使用样例**

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

        self.custom_add = ops.Custom(func="AddCustom", out_shape=lambda x, _: x, out_dtype=lambda x, _: x,
                                     func_type="aot",reg_info=aclop_reg_info)

    def construct(self, x, y):
        res = self.custom_add(x, y)
        return res

x = np.ones([8, 2048]).astype(np.float16)
y = np.ones([8, 2048]).astype(np.float16)
net = CustomNet()
```

**注意事项**

1. **名称一致性**：注册信息中使用的算子名称必须与`ops.Custom`中的`func`参数传入的名称完全一致。如果指定算子使用`aclnn`，注册信息中的算子名称前也需要添加`aclnn`前缀。

2. **输入输出名称匹配**：注册信息中定义的输入输出参数名称必须与源代码中定义的名称完全一致。

3. **规格一致性**：注册信息中支持的规格也必须与源代码中定义的规格相匹配。

4. **执行模式限制**：`aclnn`只能采用单算子执行模式，设置为PyNative模式或指定Graph模式下`jit_config`为`O0`或`O1`。`jit_config`配置说明参考[set_context](https://www.mindspore.cn/docs/zh-CN/r2.3.0/api_python/mindspore/mindspore.set_context.html)。

### 进一步阅读

- **自定义算子注册**：更多关于自定义算子的注册信息和反向函数的编写，请参考 [自定义算子注册](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0/operation/op_custom_adv.html) 。
- **AOT自定义算子**：对于C++的shape推导函数实现，以及AOT类型自定义算子的进阶用法，请参考 [aot类型自定义算子进阶用法](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.3.0/operation/op_custom_aot.html)。
- **样例工程**：想要了解更多Ascend C自定义算子的使用方式，可以查看 [样例工程](https://gitee.com/mindspore/mindspore/tree/v2.3.0/tests/st/ops/graph_kernel/custom/custom_ascendc)。

## 常见问题

1. 编译找不到头文件`"register/tilingdata_base.h"`

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

    **解决方案**：通常是因为未正确设置CANN包路径，导致编译工程找不到依赖文件。检查是否已传递`--cann_package_path`选项，以及该选项的路径是否正确，并确认是否已正确安装配套的昇腾软件开发包。

2. 自定义算子执行报下面的错误：

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

    **解决方案**：上述问题一般是图模式下报错，根因是自定义算子使用时的注册信息与自定义算子实现中的原型定义不一致导致的，例如算子的实现中原型定义为

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

    而算子使用时的注册信息时：

    ```python
    reg_info = CustomRegOp("AddCustom") \
                .input(0, "x", "required") \
                .input(1, "y", "required") \
                .output(0, "output", "required") \
                .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
                .target("Ascend") \
                .get_op_info()
    ```

    两个算子信息中output的名字不一致，算子原型中命名为`z`，而reg_info中命名为`output`，注意这种细小差异导致的报错。

3. 报错不支持的算子类型

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

    **解决方案**： 从报错日志分析，用户指定`AddCustom`底层使用aclnn，但是却在aclop流程报错，说明算子选择未找到aclnn对应的符号，而使用了默认的aclop，这种情况请用户首先检查环境配置是否正确，包括是否正确安装自定义算子安装包或正确指定自定义算子的环境变量`ASCEND_CUSTOM_OPP_PATH`，打开info日志，过滤`op_api_convert.h`文件的日志，检查符号是否正确加载。