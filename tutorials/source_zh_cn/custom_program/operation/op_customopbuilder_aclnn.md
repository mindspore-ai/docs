# CustomOpBuilder 通过 AclnnOpRunner 接入 ACLNN 算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/tutorials/source_zh_cn/custom_program/operation/op_customopbuilder_aclnn.md)

## 概述

CANN中的[AOL算子加速库](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/operatorlist_00001.html)提供了大量深度优化、硬件亲和的高性能算子。若MindSpore尚未封装[aclnn](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/context/common/aclnn_domains.md)算子的Python接口，或你已基于[Ascend C](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html)自研开发了算子，只需**CustomOpBuilder + AclnnOpRunner** 即可在 **动态图（PyNative）模式**下无感接入，无需关心内存、stream、workspace等底层细节。

aclnn算子的调用方式，通常是基于“两段式”接口，样式形如：

```c++
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor * src, ..., aclTensor * out, ..., uint64_t * workspaceSize, aclOpExecutor ** executor);
aclnnStatus aclxxXxx(void * workspace, uint64_t workspaceSize, aclOpExecutor * executor, aclrtStream stream);
```

必须先调用第一段接口`aclxxXxxGetWorkspaceSize`，用于计算本次API调用过程中需要多少workspace内存，获取到计算所需的workspace size后，按照workspace size申请NPU内存，然后调用第二段接口`aclxxXxx`执行计算。

在 [基于CustomOpBuilder的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_customopbuilder.html)中，MindSpore提供了 `PyboostRunner` 方便用户在动态图接入自定义算子。为了简化调用流程，隐藏接口数据类型转换操作，MindSpore针对`aclnn`算子提供了统一的执行入口`ms::pynative::AclnnOpRunner`，其支持PyBoost多级流水和MindSpore框架的算子缓存能力，提高算子和网络执行效率。

本教程以 `ArgMin`为例，展示完整接入流程。完整代码可参考 [MindSpore 代码仓](https://atomgit.com/mindspore/mindspore/tree/master/tests/st/custom/custom_op_builder/pyboost/)。

## 安装 ACLNN 开发环境

1. CANN中算子

    若算子为CANN包中算子，则用户无需额外配置环境，只需按照[MindSpore安装指南](https://www.mindspore.cn/install/#guide)配置MindSpore使用环境即可。

2. Ascend C自定义算子

    若算子是用户基于Ascend C开发的自定义算子，则需将算子编译结果路径添加环境变量`ASCEND_CSUTOM_OPP_PATH`中，例如：

    ```shell
    export ASCEND_CUSTOM_OPP_PATH={build_out_path}/build_out/_CPack_Package/Linux/External/custom_opp_euleros_aarch64.run/packages/vendors/{your_custom_name}:$ASCEND_CUSTOM_OPP_PATH
    ```

## ArgMin算子接入

下面给出示例完整代码。

```cpp
#include <set>
#include <optional>
#include "ms_extension/all.h"

namespace custom {

/* 1. 推导输出shape */
static ShapeVector InferArgMinShape(const ShapeVector &in_shape, int64_t dim, bool keep_dims) {
  const int64_t rank = static_cast<int64_t>(in_shape.size());
  if (rank == 0) {
    return in_shape;
  }

  int64_t axis = (dim < 0) ? (dim + rank) : dim;
  if (axis < 0 || axis >= rank) {
    MS_LOG(EXCEPTION) << "Infer shape failed";
  }

  ShapeVector out_shape;
  out_shape.reserve(keep_dims ? rank : rank - 1);

  for (int64_t i = 0; i < rank; ++i) {
    if (i == axis) {
      if (keep_dims) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(in_shape[i]);
    }
  }

  return out_shape;
}

/* 2. 构造空输出Tensor */
ms::Tensor GenResultTensor(const ms::Tensor &t, int64_t dim, bool keep_dim, ms::TypeId type_id) {
  ShapeVector in_shape = t.shape();
  ShapeVector out_shape = InferArgMinShape(in_shape, dim, keep_dim);
  return ms::Tensor(type_id, out_shape);
}

/* 3. 算子入口：被Python直接调用 */
ms::Tensor npu_arg_min(const ms::Tensor &x, int64_t dim, bool keep_dim) {
  auto result = GenResultTensor(x, dim, keep_dim, ms::TypeId::kNumberTypeInt64);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ArgMin");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnArgMin, x, dim, keep_dim, result));
  runner->Run({x}, {result});
  return result;
}
}  // namespace custom

/* 4. PYBIND11接口定义 */
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_arg_min", PYBOOST_CALLER(1, custom::npu_arg_min)); }
```

### 1. 推导算子输出信息

```cpp
auto y = GenResultTensor(x, axis, keep_dims);
```

这一步骤是基于算子的计算逻辑，使用`shape`和`type`创建输出Tensor。例如，`aclnnArgMin`根据 `axis` 与 `keep_dims`提前计算出输出`shape`和`type`并使用`ms::Tensor(dtype, shape)`构造空Tensor。该Tensor只分配元信息，不分配device内存，`AclnnOpRunner::Run`会在内部分配device内存。

### 2. 创建AclnnOpRunner

在[基于CustomOpBuilder的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_customopbuilder.html)中，MindSpore提供了通用的接入自定义算子类`PyboostRunner`。针对aclnn算子，用户可直接使用`AclnnOpRunner`类创建对象。

```c++
auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ArgMin");
```

### 3. 调用接口执行算子

```cpp
runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnArgMin, x, axis, keep_dims, y));
runner->Run({x}, {y});
```

在`LAUNCH_ACLNN_FUNC`中依次传入算子名字、输入、输出，并调用`SetLaunchFunc`方法将launch函数设置到runner中。调用`Run`方法，入参分别是算子`ms::Tensor`类型的输入和输出。

### 4. 通过pybind11将C++函数绑定一个Python函数

```c++
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_arg_min", PYBOOST_CALLER(1, custom::npu_arg_min)); }
```

`npu_arg_min`: 前端接口名称。
`custom::npu_arg_min`: 实际调用的后端接口。
`PYBOOST_CALLER`: 入参为输出个数和后端接口。

### 5. 使用CustomOpBuilder编译自定义算子

将上述C++代码保存成文件`argmin.cpp`，然后使用Python接口CustomOpBuilder编译。

```python
import mindspore as ms
import numpy as np

my_ops = CustomOpBuilder("my_custom", 'argmin.cpp', backend="Ascend").load()
x = np.random.randn(2, 3, 4, 5).astype(np.float32)
output = my_ops.npu_arg_min(ms.Tensor(x), 0, False)
```
