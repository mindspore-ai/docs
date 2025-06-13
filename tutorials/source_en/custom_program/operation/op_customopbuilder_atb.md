# CustomOpBuilder: Integrating ATB Operators Using AtbOpRunner

[![View Source File](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/custom_program/operation/op_customopbuilder_atb.md)

## Overview

[Ascend Transformer Boost (ATB) Operator Acceleration Library](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/acce/ascendtb/ascendtb_0001.html) is an operator library specifically designed for training and inference of Transformer models, based on Huawei's Ascend AI processors.

When users need to use operators from the ATB acceleration library that are not provided by MindSpore, they can quickly integrate and use them through custom operators.

In [Custom Operators Based on CustomOpBuilder](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/op_customopbuilder.html), MindSpore provides the `PyboostRunner` tool to allow users to integrate custom operators in dynamic graphs. Now, for ATB operators, MindSpore additionally provides the `AtbOpRunner` tool to encapsulate the ATB operator's workflow and the dynamic graph's multi-stage pipeline.

In the complete [ATB operator workflow](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/acce/ascendtb/ascendtb_0037.html), users need to execute steps such as constructing `Param`, creating `Operation` and `Context`, setting `variantPack` (operator input-output tensors), calling `Setup`, calling `Execute`, and destroying `Context` and `Operation`. However, for a single operator, its `Operation` only depends on operator attributes (`Param`), and its `Context` only depends on the stream, both of which can be reused. Therefore, MindSpore provides a cache to store these data structures, avoiding unnecessary time consumption caused by repeated creation and destruction.

When integrating ATB operators using the [AtbOpRunner class](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/cpp_api_for_custom_ops.html#class-AtbOpRunner), users only need to provide a corresponding hash function for `Param` (used as the key for caching `Operation`) and call the `Init` interface for initialization (constructing `Operation`), followed by the `Run` interface to execute the ATB operator. Additionally, users can directly call the [RunAtbOp](https://www.mindspore.cn/tutorials/en/master/custom_program/operation/cpp_api_for_custom_ops.html#function-runatbop) function for one-click execution (the function internally includes calls to both `Init` and `Run` interfaces).

This guide uses `SwiGLU` as an example to demonstrate the ATB operator integration process. The complete code can be found in the [code repository](https://gitee.com/mindspore/mindspore/blob/master/tests/st/graph_kernel/custom/jit_test_files/atb_swiglu.cpp).

## Installing the ATB Acceleration Library

[Click here for installation tutorial](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/developmentguide/acce/ascendtb/ascendtb_0034.html)

Since MindSpore uses the "ABI=0" standard during construction, the `set_env.sh` script for ATB also requires the "ABI=0" configuration. For example:

```sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=0 &> /dev/null
```

## Integrating the SwiGLU Operator

Here we use `ms::pynative::RunAtbOp` to integrate the operator and call the function interface through `ms::pynative::PyboostRunner::Call`:

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

### 1. Provide the Hash Function for Param

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

As described in the [ATB Acceleration Library API documentation](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendtbapi/ascendtb_01_0005.html), the ATB `SwiGLU` operator uses the `atb::infer::ActivationParam` parameter.

The hash function is defined as an `operator()` function within the `HashOpParam` template class. Users can specialize this class with an actual `Param` type and must place it within the `namespace atb`. Within the hash function, the `add_param_to_buf` interface is used to sequentially add the member variables of `Param`, and the framework calculates an integer hash value based on the values in the buffer.

> In general, if a specific value of an operator parameter is unused or fixed, it can be excluded from the hash function. However, for maintainability and extensibility, it is recommended to include all member variables of `Param` in the hash function to avoid potential precision issues caused by missing hash values when extending operator functionality in the future.

### 2. Infer the Output Information of the Operator

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

For the `SwiGLU` operator, the output tensor has the same data type as the input tensor. Its shape only differs in the `dim` dimension, which has a length half that of the input dimension, while other dimensions remain the same. After inferring the output shape, an empty tensor is constructed using the `ms::Tensor` constructor.

Here, the output tensor is defined as `y`:

```cpp
auto y = InferSwigluForward(x, dim);
```

### 3. Create and Set the Operator Attribute Structure

```cpp
atb::infer::ActivationParam param;
param.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
param.dim = dim;
```

### 4. Execute the Operator via the RunAtbOp Interface

```cpp
ms::pynative::RunAtbOp("SwiGLU", param, {x}, {y});
```

This is a template interface, equivalent to:

```cpp
auto runner = std::make_shared<AtbOpRunner>("SwiGLU");
runner->Init(param);
runner->Run({x}, {y});
```

By passing in the operator name, attributes, input tensor list, and output tensor list, the corresponding ATB operator can be invoked. This interface supports multi-stage pipeline execution in dynamic graphs.

### 5. Bind the C++ Function to a Python Function via pybind11

```cpp
auto pyboost_npu_swiglu(const ms::Tensor &x, int32_t dim) {
  return ms::pynative::PyboostRunner::Call<1>(npu_swiglu, x, dim);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_swiglu", &pyboost_npu_swiglu, "swiglu realization", pybind11::arg("x"), pybind11::arg("dim") = -1);
}
```

### 6. Compile the Custom Operator Using CustomOpBuilder

Save the above C++ code as a file named `atb_activation.cpp`, and then compile it using the Python interface `CustomOpBuilder`.

```python
x = mindspore.Tensor(np.random.rand(2, 32).astype(np.float16))
my_ops = CustomOpBuilder("atb_activation", "atb_activation.cpp", enable_atb=True).load()
y = my_ops.swiglu(x, -1)
print(y)
```

Here, the parameter `enable_atb=True` is passed into `CustomOpBuilder`, and MindSpore will automatically add compilation and linking options related to the ATB acceleration library. Users only need to ensure that the `set_env.sh` script for the ATB library has been correctly executed, and the environment contains the `ATB_HOME_PATH` variable.
