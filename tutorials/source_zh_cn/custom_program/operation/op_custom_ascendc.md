# Custom原语AOT类型自定义算子（Ascend平台）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/custom_program/operation/op_custom_ascendc.md)

## 概述

AOT类型的自定义算子采用预编译的方式，要求网络开发者基于特定接口，手写算子实现函数对应的源码文件，并提前将源码文件编译为动态链接库，然后在网络运行时框架会自动调用执行动态链接库中的函数。AOT类型的自定义算子支持昇腾平台的Ascend C编程语言，这是一款专为算子开发而设计的高效编程语言。本指南将从用户角度出发，详细介绍基于Ascend C的自定义算子开发和使用流程，包括以下关键步骤：

1. **自定义算子开发**：使用Ascend C编程语言，可以快速开发自定义算子，降低开发成本并提高开发效率。
2. **离线编译与部署**：完成算子开发后，进行离线编译，确保算子可以在Ascend AI处理器上高效运行，并进行部署。
3. **MindSpore使用自定义算子**：将编译后的Ascend C自定义算子集成到MindSpore框架中，实现在实际AI应用中的使用。

本章内容旨在帮助开发者全面了解并掌握Ascend C自定义算子的整个生命周期，从开发到部署，再到在MindSpore中的有效利用。对于其他平台的AOT自定义算子开发，参考[AOT类型自定义算子（CPU/GPU平台）](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/operation/op_custom_aot.html)。

## 自定义算子开发

昇腾平台提供了全面的Ascend C算子开发教程，帮助开发者深入理解并实现自定义算子。以下是关键的开发步骤和资源链接：

**基础教程**：访问[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) 获取入门知识。

**算子实现**：学习[基于自定义算子工程的算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) ，快速了解自定义算子开发的端到端流程，重点关注kernel侧实现和host侧实现。

**开发样例**：昇腾社区提供了丰富的 [Ascend C算子开发样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc) ，覆盖了多种类型算子，帮助您快速理解算子开发的实际应用。也可以查看 [AddCustom自定义算子开发样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom) ，它简洁展示了一个自定义算子开发需要的核心工作。

## 编译与部署方法

### 环境准备

确保已具备以下条件，以使用MindSpore的Ascend C自定义算子离线编译工具：

- **Ascend C源码**: 包括host侧和kernel侧的自定义算子实现。
- **MindSpore安装**: 确保已安装2.3.0及以上版本的MindSpore。
- **CMake**: CMake>=3.16.0。

### 离线编译与部署

若在上述步骤中，已通过CANN的自定义算子编译工程完成编译和部署，则可跳过该步骤。MindSpore同样提供了自定义的编译工具，在开发完自定义算子后，准备好自定义算子的kernel侧和host侧，可按照下述步骤进行自定义算子的编译部署。

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
   | `--install_path`  | 自定义算子安装路径               | 无     | 否       |
   | `-i`              | 安装自定义算子到`--install_path`，否则安装到环境变量`ASCEND_OPP_PATH`指定的路径 | 不设置 | 否       |
   | `-c`              | 删除编译日志和结果文件       | 不设置 | 否       |

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

## MindSpore使用自定义算子

### 环境准备

在开始之前，请确保已完成Ascend C自定义算子的开发、编译和部署。您可以通过安装自定义算子包或设置环境变量来准备使用环境。

### 使用自定义算子

MindSpore的自定义算子接口为[ops.Custom](https://www.mindspore.cn/docs/zh-CN/r2.6.0/api_python/ops/mindspore.ops.Custom.html)，
使用Ascend C自定义算子时，您需要设置参数`func_type`为`"aot"`，并指定`func`参数为算子名字。`func`用来指示该算子在动态库中的入口函数名，根据infer函数的实现方式，存在以下两种使用方式：

- **Python infer**：若算子的infer函数是Python实现，即通过`out_shape`参数传入infer shape函数，`out_dtype`参数传入infer type函数，则指定`func`为算子名，例如`func="CustomName"`
- **C++ infer**：若算子的infer函数通过C++实现，则在func中传入infer实现文件的路径并用`:`隔开算子名字，例如：`func="add_custom_infer.cc:AddCustom`。MindSpore会在后面分别拼接`InferShape`和`InferType`去查找对应的infer函数。

**使用样例**：

```python
class AddCustomNet(Cell):
    def __init__(self, func, out_shape, out_dtype):
        super(AddCustomNet, self).__init__()
        reg_info = CustomRegOp("AddCustom") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_add = ops.Custom(func=func, out_shape=out_shape, out_dtype=out_dtype, func_type="aot", bprop=None,
                                     reg_info=reg_info)

    def construct(self, x, y):
        res = self.custom_add(x, y)
        return res

mindspore.set_context(jit_config={"jit_level": "O0"})
mindspore.set_device("Ascend")
x = np.ones([8, 2048]).astype(np.float16)
y = np.ones([8, 2048]).astype(np.float16)

# 通过lambda实现infer shape函数
net = AddCustomNet("AddCustom", lambda x, _: x, lambda x, _: x)

# 使用C++实现infer shape和infer type，在func中传入infer的路径
net = AddCustomNet("./infer_file/add_custom_infer.cc:AddCustom", None, None)
```

**C++ infer shape和infer type实现示例：**

```cpp
#include <vector>
#include <stdint.h>
#include "custom_aot_extra.h"
enum TypeId : int {};

extern "C" std::vector<int64_t> AddCustomInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
  std::vector<int64_t> output_shape;
  auto input0_size = ndims[0];
  for (size_t i = 0; i < input0_size; i++) {
    output_shape.push_back(shapes[0][i]);
  }
  return output_shape;
}

extern "C" TypeId AddCustomInferType(std::vector<TypeId> type_ids, AotExtra *extra) { return type_ids[0]; }

```

完整Ascend C自定义算子的样例代码，可以查看 [样例工程](https://gitee.com/mindspore/mindspore/tree/v2.6.0/tests/st/graph_kernel/custom/custom_ascendc)。样例工程的目录结构如下：

```text
.
├── compile_utils.py                //自定义算子编译公共文件
├── infer_file
│   ├── custom_cpp_infer.cc         //自定义算子C++侧infer shape和infer type文件
│   └── custom_aot_extra.h          //自定义算子infer shape编译依赖头文件
├── op_host                         //自定义算子源码op_host
│   ├── add_custom.cpp
│   └── add_custom_tiling.h
├── op_kernel                       //自定义算子源码op_kernel
│   └── add_custom.cpp
├── test_compile_custom.py          //自定义算子编译用例
├── test_custom_aclnn.py            //自定义算子使用样例
├── test_custom_aclop.py            //自定义算子走aclop流程使用样例
├── test_custom_ascendc.py          //自定义算子启动脚本，包含编译和执行，端到端流程
└── test_custom_level0.py           //Custom接口使用简单示例，可作为阅读入口
```

**注意事项**

1. **名称一致性**：注册信息中使用的算子名称必须与`ops.Custom`中的`func`参数传入的名称完全一致。

2. **输入输出名称匹配**：注册信息中定义的输入输出参数名称必须与源代码中定义的名称完全一致。

3. **规格一致性**：注册信息中支持的规格也必须与源代码中定义的规格相匹配。

### 进一步阅读

- **自定义算子注册**：更多关于自定义算子的注册信息和反向函数的编写，请参考 [自定义算子注册](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/operation/op_custom_adv.html)。
- **AOT自定义算子**：对于C++的shape和type推导函数实现，以及AOT类型自定义算子的进阶用法，请参考 [AOT类型自定义算子进阶用法](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/operation/op_custom_aot.html)。

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

    **解决方案**：上述问题一般是图模式下报错，根因是自定义算子使用时的注册信息与自定义算子实现中的原型定义不一致导致的，例如算子的实现中原型定义为：

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

    **解决方案**：从报错日志分析，用户指定`AddCustom`底层使用aclnn，但是却在aclop流程报错，说明算子选择未找到aclnn对应的符号，而使用了默认的aclop。若出现这种情况，请用户首先检查环境配置是否正确，包括是否正确安装自定义算子安装包或正确指定自定义算子的环境变量`ASCEND_CUSTOM_OPP_PATH`，打开info日志，过滤`op_api_convert.h`文件的日志，检查符号是否正确加载。