# CustomOpBuilder通过AtbOpRunner接入ATB算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/custom_program/operation/op_customopbuilder_atb.md)

## 概述

[Ascend Transformer Boost (ATB) 算子加速库](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/acce/ascendtb/ascendtb_0001.html) 是基于华为Ascend AI处理器，专门为Transformer模型的训练和推理而设计的算子库。

当用户需要使用ATB加速库的算子，而MindSpore未提供相应算子接口时，用户可以使用自定义算子的方法快速接入使用。

在 [基于CustomOpBuilder的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/op_customopbuilder.html) 中，MindSpore提供了 `PyboostRunner` 方便用户在动态图接入自定义算子。现在针对ATB算子，MindSpore又额外提供了一套`AtbOpRunner`用于把ATB算子的调用流程和动态图多级流水封装到一起。

在完整的[ATB算子的调用流程](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/acce/ascendtb/ascendtb_0037.html)中，用户需要执行 构造`Param`、创建`Operation`和`Context`、设置`variantPack`（算子输入输出张量）、调用`Setup`、调用`Execute`、销毁`Context`和`Operation` 等流程。但是对于一个算子来说，其`Operation`仅依赖于算子属性（`Param`），其`Context`仅依赖于流（stream），且都是可以复用的，因此MindSpore提供了一个缓存，将这些数据结构放在缓存中，避免多次创建和销毁带来不必要的时间消耗。

用户基于 [AtbOpRunner类](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/cpp_api_for_custom_ops.html#class-AtbOpRunner) 对接ATB算子时，仅需要提供相应`Param`的哈希函数（作为缓存`Operation`的键值），并调用`Init`接口初始化（即构造`Operation`），再调用`Run`接口即可执行ATB算子。还可以直接调用 [RunAtbOp](https://www.mindspore.cn/tutorials/zh-CN/master/custom_program/operation/cpp_api_for_custom_ops.html#function-runatbop)函数一键执行（函数内包含了`Init`和`Run`接口的调用）。

本指南以一个`SwiGLU`为例，展示ATB算子的接入流程。完整代码请参阅[代码仓库](https://gitee.com/mindspore/mindspore/blob/master/tests/st/graph_kernel/custom/jit_test_files/atb_swiglu.cpp)。

## 安装ATB加速库

[点这里查看安装教程](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/acce/ascendtb/ascendtb_0034.html)

由于MindSpore在构建时采用的是“ABI=0”的标准，所以在设置ATB的`set_env.sh`脚本时也需要加上“ABI=0”的配置，例如：

```sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 &> /dev/null
```

## SwiGLU算子接入

这里使用`ms::pynative::RunAtbOp`接入算子，并通过`ms::pynative::PyboostRunner::Call`调用函数接口：

```cpp
#include "ms_extension/api.h"

namespace atb {
template <>
struct HashOpParam<atb::infer::ActivationParam> {
  void operator()(const atb::infer::ActivationParam &param) const {
    add_param_to_buf("activationType", param.activationType);
    add_param_to_buf("scale", param.scale);
    add_param_to_buf("dim", param.dim);
    add_param_to_buf("geluMode", param.geluMode);
  }
};
}  // namespace atb

ms::Tensor InferSwigluForward(const ms::Tensor &x, int32_t dim) {
  ShapeVector out_tensor_shape(x.shape());
  int64_t split_dim = dim;
  if (split_dim < 0) {
    split_dim += out_tensor_shape.size();
  }
  const int64_t split_num = 2;
  out_tensor_shape[split_dim] /= split_num;
  return ms::Tensor(x.data_type(), out_tensor_shape);
}

ms::Tensor npu_swiglu(const ms::Tensor &x, int32_t dim) {
  auto y = InferSwigluForward(x, dim);

  atb::infer::ActivationParam param;
  param.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
  param.dim = dim;

  ms::pynative::RunAtbOp("SwiGLU", param, {x}, {y});
  return y;
}

auto pyboost_npu_swiglu(const ms::Tensor &x, int32_t dim) {
  return ms::pynative::PyboostRunner::Call<1>(npu_swiglu, x, dim);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_swiglu", &pyboost_npu_swiglu, "swiglu realization", pybind11::arg("x"), pybind11::arg("dim") = -1);
}
```

### 1. 提供Param的哈希函数

```cpp
namespace atb {
template <>
struct HashOpParam<atb::infer::ActivationParam> {
  void operator()(const atb::infer::ActivationParam &param) const {
    add_param_to_buf("activationType", param.activationType);
    add_param_to_buf("scale", param.scale);
    add_param_to_buf("dim", param.dim);
    add_param_to_buf("geluMode", param.geluMode);
  }
};
}  // namespace atb
```

通过查看[ATB加速库的API文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendtbapi/ascendtb_01_0005.html)可以知道ATB的`SwiGLU`算子使用的是`atb::infer::ActivationParam`参数。

哈希函数被定义成一个`HashOpParam`模板类里的`operator()`函数。用户通过实际`Param`特例化此类，并需要放在`namespace atb`内。在哈希函数里仅需使用`add_param_to_buf`接口依次添加Param的各个成员变量即可，框架在调用时会根据缓存区内的值计算得到一个整数哈希值。

> 一般情况下，如果算子参数的某个值是未使用的或者固定值的，那可以不把它加进哈希函数内，因为哈希函数的目的是对于相同的Param仅创建一次Operation。但是为了可维护性和可扩展性，防止以后在扩展算子功能时，因为疏忽而漏了计算某个成员变量的哈希值，导致出现难以定位的精度问题，可以在一开始就把Param的成员变量都添加上去。

### 2. 推导算子的输出信息

```cpp
ms::Tensor InferSwigluForward(const ms::Tensor &x, int32_t dim) {
  ShapeVector out_tensor_shape(x.shape());
  int64_t split_dim = dim;
  if (split_dim < 0) {
    split_dim += out_tensor_shape.size();
  }
  const int64_t split_num = 2;
  out_tensor_shape[split_dim] /= split_num;
  return ms::Tensor(x.data_type(), out_tensor_shape);
}
```

对于`SwiGLU`算子，它输出张量的数据类型和输入的一样，形状仅有`dim`维度长度是输入维度的一半长度，其它维度与输入维度一样。推导出输出形状之后，通过`ms::Tensor`构造函数构造一个空的张量。

这里定义输出张量为 `y`：

```cpp
auto y = InferSwigluForward(x, dim);
```

### 3. 创建并设置算子属性结构体

```cpp
atb::infer::ActivationParam param;
param.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
param.dim = dim;
```

### 4. 调用RunAtbOp接口执行算子

```cpp
ms::pynative::RunAtbOp("SwiGLU", param, {x}, {y});
```

这是一个模板接口，其等效于：

```cpp
auto runner = std::make_shared<AtbOpRunner>("SwiGLU");
runner->Init(param);
runner->Run({x}, {y});
```

传入算子名、属性、输入张量列表、输出张量列表几个信息，即可调用相应的ATB算子。此接口支持了动态图的多级流水执行流程。

### 5. 通过pybind11将C++函数绑定一个Python函数

```cpp
auto pyboost_npu_swiglu(const ms::Tensor &x, int32_t dim) {
  return ms::pynative::PyboostRunner::Call<1>(npu_swiglu, x, dim);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_swiglu", &pyboost_npu_swiglu, "swiglu realization", pybind11::arg("x"), pybind11::arg("dim") = -1);
}
```

### 6. 使用CustomOpBuilder编译自定义算子

将上述C++代码保存成文件`atb_activation.cpp`，然后使用Python接口`CustomOpBuilder`编译。

```python
x = mindspore.Tensor(np.random.rand(2, 32).astype(np.float16))
my_ops = CustomOpBuilder("atb_activation", "atb_activation.cpp", enable_atb=True).load()
y = my_ops.swiglu(x, -1)
print(y)
```

这里向`CustomOpBuilder`传入了`enable_atb=True`的参数，MindSpore会自动添加与ATB加速库有关的编译和链接选项。用户续保证正确执行了ATB库的`set_env.sh`脚本，环境中有了`ATB_HOME_PATH`环境变量。
