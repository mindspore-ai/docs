# Custom原语AOT类型自定义算子（Ascend平台）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/custom_program/operation/op_custom_ascendc.md)

## 概述

AOT类型的自定义算子采用预编译的方式，要求网络开发者基于特定接口，手写算子实现函数对应的源码文件，并提前将源码文件编译为动态链接库，然后在网络运行时框架会自动调用执行动态链接库中的函数。AOT类型的自定义算子支持昇腾平台的Ascend C编程语言，这是一款专为算子开发而设计的高效编程语言。本指南将从用户角度出发，详细介绍基于Ascend C的自定义算子开发和使用流程，包括以下关键步骤：

1. **自定义算子开发**：使用Ascend C编程语言，可以快速开发自定义算子，降低开发成本并提高开发效率。
2. **离线编译与部署**：完成算子开发后，进行离线编译，确保算子可以在Ascend AI处理器上高效运行，并进行部署。
3. **MindSpore使用自定义算子**：将编译后的Ascend C自定义算子集成到MindSpore框架中，实现在实际AI应用中的使用。

本章内容旨在帮助开发者全面了解并掌握Ascend C自定义算子的整个生命周期，从开发到部署，再到在MindSpore中的有效利用。对于其他平台的AOT自定义算子开发，参考[AOT类型自定义算子（CPU/GPU平台）](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_custom_aot.html)。

## 自定义算子开发

昇腾平台提供了全面的Ascend C算子开发教程，帮助开发者深入理解并实现自定义算子。以下是关键的开发步骤和资源链接：

**基础教程**：访问[Ascend C算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)和[Ascend C API列表](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendcopapi/atlasascendc_api_07_0003.html)获取入门知识。

**算子实现**：学习[基于自定义算子工程的算子开发](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/developmentguide/opdevg/Ascendcopdevg/atlas_ascendc_10_0006.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)，快速了解自定义算子开发的端到端流程，重点关注kernel侧实现和host侧实现。

**开发样例**：昇腾社区提供了丰富的 [Ascend C算子开发样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc)，覆盖了多种类型算子，帮助您快速理解算子开发的实际应用。也可以查看 [AddCustom自定义算子开发样例](https://gitee.com/ascend/samples/tree/master/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom)，它简洁展示了一个自定义算子开发需要的核心工作。

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

MindSpore自定义算子接口为[ops.Custom](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Custom.html)，
详细的接口说明可以参看[ops.Custom](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Custom.html)
，本文侧重说明如何使用[ops.Custom](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.Custom.html)
原语接入Ascend C自定义算子。

### 环境准备

在开始之前，请确保已完成Ascend C自定义算子的开发、编译和部署。您可以通过安装自定义算子包或设置环境变量`ASCEND_CUSTOM_OPP_PATH`来准备使用环境。

### 参数说明

```python
ops.Custom(func, bprop=None, out_dtype=None, func_type='aot', out_shape=None, reg_info=None)
```

- `func`(str)： 自定义算子名字。
- `out_shape`(Union[function, list, tuple]): 输出shape或输出shape的推导函数。默认值： `None`。
- `out_dtype` (Union[
  function, [mindspore.dtype](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype)
  ,list, tuple]) :
  输出type或输出type的推导函数。默认值： `None`。
- `func_type`(str): 自定义算子的函数类型， Ascend C自定义算子指定`func_type="aot"`。
- `bprop`(function): 自定义算子的反向函数。默认值： `None`。
- `reg_info`(Union[str, dict, list, tuple]): 自定义算子的算子注册信息。默认值： `None`。Ascend C自定义算子无需传入该参数，使用默认值。

**场景限制**： 当前动态图和静态图O2模式只支持输入输出为Tensor类型，静态图O0/O1模式无限制类型。Ascend
C自定义算子动态图场景推荐使用[基于CustomOpBuilder的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_customopbuilder.html)。

### 简单示例

通过上述参数说明，使用Ascend C自定义算子时，需重点关注`func`、`out_shape`、`out_dtype`三个核心入参。下面是一个简单示例，帮助用户直观理解Ascend C自定义算子在MindSpore框架中的使用方法。

首先，使用`ops.Custom`原语定义自定义算子，并传入必要参数。算子名字`func`指定为`aclnnCast`，`out_shape`
传入通过lambda函数实现的推导shape函数，该算子的输出shape
与第一个输入的shape相同，`out_dtype`直接指定为MindSpore的内置数据类型`mstype.float32`。关于`out_shape`和`out_dtype`
的实现，将在后续部分进行详细介绍。
定义好自定义算子后，通过传入该算子的所有合法输入来使用该算子，例如，在用例的construct中调用`self.custom_cast`
，传入两个参数，分别是原始数据x（Tensor）和目标的数据类型dst_type（mindspore.dtype）。

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

您可以查看MindSpore仓中的 [自定义算子测试用例](https://gitee.com/mindspore/mindspore/tree/master/tests/st/graph_kernel/custom/custom_ascendc)
获取更多数据类型和使用场景的Ascend C自定义算子用例。
样例工程的目录结构如下：

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
├── test_custom_ascendc.py          //自定义算子启动脚本，包含编译和执行，端到端流程
├── test_custom_level0.py           //自定义算子组合场景简单示例
├── test_custom_multi_output.py     //自定义算子多输出场景使用样例
├── test_custom_multi_type.py       //自定义算子不同输入类型使用样例，可作为阅读入口
├── test_custom_tensor_list.py      //自定义算子动态输入输出使用样例
└── test_custom_utils.py            //内部测试文件
```

### Infer Shape/Type

为了确定自定义算子输出的类型和大小，通过`out_shape`和`out_dtype`参数传入算子的shape和type。这两个参数通常需要通过推导才能确定。用户可以传入确定的shape和type，
也可以通过函数推导输出shape和type。本节主要说明如何通过函数推导输出shape和type。

**说明**

- shape和type的推导函数实现有Python侧和C++侧两种方式。
- Python侧infer易用性更高，但是在动态图场景，C++侧infer性能更高。
- 动态shape和值依赖的场景只能在C++侧infer shape。

#### Python侧Infer Shape/Type

推导函数的输入为自定义算子输入的shape或type，输出为推导结果，即输出shape或type。以下是几个推导函数的例子。

- 输出type和shape与输入相同场景的infer函数

   ```python
   from mindspore import ops


   # Add算子有两个输入，输出的shape与输入的shape相同
   def add_infer_shape(x, _):
       return x


   # Add算子有两个输入，输出type与输入的type相同
   def add_infer_type(x, _):
       return x


   # 定义自定义算子
   custom_add = ops.Custom(func="aclnnAdd", out_shape=add_infer_shape, out_dtype=add_infer_type, func_type="aot")

   # 对于简单的infer shape或infer type， 也可以直接使用lambda函数
   custom_add = ops.Custom(func="aclnnAdd", out_shape=lambda x, y: x, out_dtype=lambda x, y: x, func_type="aot")
   ```

- 输出shape通过输入shape计算场景的infer函数

   ```python
   from mindspore import ops
   import mindspore.common.dtype as mstype


   # 算子的输出shape由输入的shape计算得到，输出为tuple类型
   def msda_infer_shape_1(v_s, vss_v, vlsi_s, sl_s, aw_s):
       return [v_s[0], sl_s[1], v_s[2] * v_s[3]]


   # 算子的输出shape由输入的shape计算得到，输出为list类型
   def msda_infer_shape_2(v_s, vss_v, vlsi_s, sl_s, aw_s):
       return (v_s[0], sl_s[1], v_s[2] * v_s[3])


   # 输出shape通过普通函数推导，输出type直接指定输出类型
   custom_msda = ops.Custom(func="aclnnMultiScaleDeformableAttn", out_shape=msda_infer_shape_1,
                            out_dtype=mstype.float32, func_type="aot")

   # 输出shape和type通过lambda函数推导
   custom_msda = ops.Custom(func="aclnnMultiScaleDeformableAttn",
                            out_shape=lambda v_s, vss_s, vlsi_s, sl_s, aw_s: (v_s[0], sl_s[1], v_s[2] * v_s[3]),
                            out_dtype=lambda v_s, vss_s, vlsi_s, sl_s, aw_s: v_s, func_type="aot")
   ```

- 多输出和动态输出场景的infer函数

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

**注意事项**

- 在infer shape函数中，计算过程应避免改变shape值的类型，确保其为int类型。例如，除法操作可能导致计算结果为float类型，从而引发shape转换失败。
- 在多输出和动态输出场景下，若shape和type都在Python侧推导，则需保持两者的返回值类型一致，都为list或都为tuple。

#### C++侧Infer Shape/Type

若使用C++推导函数，则需设置参数`func`，其值为推导函数文件路径与算子名称的组合，两者以`:`
分隔。同时，在定义算子时，应将`out_shape`或`out_dtype`设为`None`。

```python
# shape和type的推导函数在./infer_file/add_custom_infer.cc文件中实现
ops.Custom(func="./infer_file/add_custom_infer.cc:AddCustom", out_shape=None, out_dtype=None, func_type="aot")

# shape的推导函数在./infer_file/add_custom_infer.cc文件中实现，type推导函数在Python侧通过lambda函数实现
ops.Custom(func="./infer_file/add_custom_infer.cc:AddCustom", out_shape=None, out_dtype=lambda x, y: x, func_type="aot")
```

**Infer Shape函数原型**

```cpp
extern "C" std::vector<int64_t> FuncNameInferShape(int *ndims, int64_t **shapes, AotExtra *extra)

extern "C" std::vector<std::vector<int64_t>> FuncNameInferShape(int *ndims, int64_t **shapes, AotExtra *extra)
```

其中，函数名`FuncName`为算子名字，单输出返回值为`std::vector<int64_t>`
类型，多输出或动态输出返回值为`std::vector<std::vector<int64_t>>`
类型，值为输出的shape。参数列表的含义如下：

- ndims (int \*): 输入shape维度数组。
- shapes (int64_t \*\*): 输入shape数组。
- extra (AotExtra \*): 用于带属性的自定义算子扩展。其中`AotExtra`
  类型定义在MindSpore提供的头文件[custom_aot_extra.h](https://gitee.com/mindspore/mindspore/blob/master/tests/st/graph_kernel/custom/aot_test_files/custom_aot_extra.h)。

**Infer Type函数原型**

```cpp
extern "C" TypeId FuncNameInferType(std::vector<TypeId> type_ids, AotExtra *extra)

extern "C" std::vector<TypeId> FuncNameInferType(std::vector<TypeId> type_ids, AotExtra *extra)
```

其中，函数名`FuncName`为算子算子的名字。单输出返回值为`TypeId`类型，多输出和动态输出返回值为`std::vector<TypeId>`
类型，值为输出的type。参数列表的含义如下：

- type_ids (std::vector<TypeId>): 输入的TypeId数组。
- extra (AotExtra \*): 用于带属性的自定义算子扩展，与shape推导函数的入参一致。

**C++侧推导函数样例**

- 输出shape和type通过输入shape和type推导计算

  C++推导函数文件 add_infer.cc

   ```cpp
   #include <vector>
   #include <stdint.h>
   #include "custom_aot_extra.h"
   enum TypeId : int {
   };

   extern "C" std::vector<int64_t> aclnnAddInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
      std::vector<int64_t> output_shape;
      // 获取第0个输入的shape大小
      auto input0_size = ndims[0];
      // 输出shape与第0个输入的shape大小相同
      for (size_t i = 0; i < input0_size; i++) {
      output_shape.push_back(shapes[0][i]);
      }
      return output_shape;
   }

   extern "C" TypeId aclnnAddInferType(std::vector<TypeId> type_ids, AotExtra *extra) {
      // 输出type与第0个输入的type形同
      return type_ids[0];
   }
   ```

  自定义算子脚本文件 custom.py

   ```python
   # 定义自定义算子，func传入C++infer的路径，同时out_shape和out_dtype参数设为None
   custom_add = ops.Custom(func="./add_infer.cc:aclnnAdd", out_shape=None, out_dtype=None, func_type="aot")
   ```

- 输出shape值依赖场景

   在infer shape中存在输出shape依赖于具体值而不只时输入的shape的场景，当前无论是Python侧还是C++侧infer接口入参都为输入的shape，要想获取具体值，需要通过在`add_prim_attr`
接口，把值以属性的形式设置给自定义算子的原语，在C++ infer shape时通过`extra`参数获取该值。下面是一个输出shape值依赖的例子。

   C++推导函数文件 moe_infer.cc

   ```cpp
   #include <vector>
   #include <stdint.h>
   #include "custom_aot_extra.h"
   extern "C" std::vector<std::vector<int64_t>> MoeSoftMaxTopkInferShape(int *ndims, int64_t **shapes, AotExtra *extra) {
      std::vector<std::vector<int64_t>> res_output_shape;
      std::vector<int64_t> out1_shape;
      // 输出shape的第0维与第0个输入的0维相同
      out1_shape.emplace_back(shapes[0][0]);
      // 输出shape的1为从属性值中获取
      out1_shape.emplace_back(extra->Attr<int64_t>("attr_k"));
      // 算子由两个输出，且两个输出的shape相同
      res_output_shape.emplace_back(out1_shape);
      res_output_shape.emplace_back(out1_shape);
      return res_output_shape;
   }
   ```

   自定义算子脚本文件 custom.py

   ```python
   moe_softmax_topk_custom = ops.Custom(func="./infer_file/moe_infer.cc:MoeSoftMaxTopk", out_shape=None,
                                        out_dtype=[mstype.float32, mstype.int32], func_type="aot")
   # 将依赖值添加到属性中，在infer阶段可通过属性获取该值
   moe_softmax_topk_custom.add_prim_attr("attr_k", 2)
   ```

- 输入为动态输入（TensorList）场景

   如果输入为TensorList，由于infer接口入参是Tensor shape，因此框架会对TensorList展开，在infer shape函数中，注意索引正确。例如如下Concat算子的infer shape函数。

   C++推导函数文件 concat_infer.cc

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

   自定义算子脚本文件 custom.py

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

   **解决方案**：通常是因为未正确设置CANN包路径，导致编译工程找不到依赖文件。检查是否已传递`--cann_package_path`
   选项，以及该选项的路径是否正确，并确认是否已正确安装配套的昇腾软件开发包。

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

   **解决方案**：从报错日志分析，用户指定`AddCustom`
   底层使用aclnn，但是却在aclop流程报错，说明算子选择未找到aclnn对应的符号，而使用了默认的aclop。若出现这种情况，请用户首先检查环境配置是否正确，包括是否正确安装自定义算子安装包或正确指定自定义算子的环境变量`ASCEND_CUSTOM_OPP_PATH`
   ，打开info日志，过滤`op_api_convert.h`文件的日志，检查符号是否正确加载。