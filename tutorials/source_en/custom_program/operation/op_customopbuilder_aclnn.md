# CustomOpBuilder Integrates ACLNN Operators via AclnnOpRunner

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.2/tutorials/source_en/custom_program/operation/op_customopbuilder_aclnn.md)

## Overview

The [Operator Acceleration Library (AOL)](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/operatorlist_00001.html) in CANN provides a large number of deeply optimized and hardware-friendly high-performance operators. If MindSpore has not yet wrapped the [aclnn](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/context/common/aclnn_domains.md) operator's Python interface, or if you have developed your own operator based on [Ascend C](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html), you can seamlessly integrate it in **dynamic graph (PyNative) mode** using **CustomOpBuilder + AclnnOpRunner**, without worrying about low-level details such as memory, stream, or workspace.

The typical calling convention for aclnn operators is based on a "two-stage" interface, like this:

```c++
aclnnStatus aclxxXxxGetWorkspaceSize(const aclTensor * src, ..., aclTensor * out, ..., uint64_t * workspaceSize, aclOpExecutor ** executor);
aclnnStatus aclxxXxx(void * workspace, uint64_t workspaceSize, aclOpExecutor * executor, aclrtStream stream);
```

You must first call the first-stage interface `aclxxXxxGetWorkspaceSize` to calculate how much workspace memory is required for this API call. After obtaining the required workspace size, allocate NPU memory accordingly, and then call the second-stage interface `aclxxXxx` to perform the computation.

In [Custom Operator Based on CustomOpBuilder](https://www.mindspore.cn/tutorials/en/r2.7.2/custom_program/operation/op_customopbuilder.html), MindSpore provides `PyboostRunner` to help users integrate custom operators in dynamic graph mode. To simplify the calling process and hide interface data type conversion operations, MindSpore provides a unified execution entry `ms::pynative::AclnnOpRunner` for `aclnn` operators. It supports PyBoost multi-level pipeline and MindSpore's operator caching capabilities, improving operator and network execution efficiency.

This tutorial uses `ArgMin` as an example to demonstrate the full integration process. The complete code can be found in the [MindSpore repository](https://gitee.com/mindspore/mindspore/tree/v2.7.2/tests/st/graph_kernel/custom/jit_test_files/).

## Installing ACLNN Development Environment

1. **Operators in CANN**

    If the operator is already included in the CANN package, no additional environment configuration is required. Just follow the [MindSpore Installation Guide](https://www.mindspore.cn/install/en#guide) to set up the MindSpore environment.

2. **Custom Operators Based on Ascend C**

    If the operator is a custom one developed by the user based on Ascend C, you need to add the compiled operator path to the environment variable `ASCEND_CUSTOM_OPP_PATH`, for example:

    ```shell
    export ASCEND_CUSTOM_OPP_PATH={build_out_path}/build_out/_CPack_Package/Linux/External/custom_opp_euleros_aarch64.run/packages/vendors/{your_custom_name}:$ASCEND_CUSTOM_OPP_PATH
    ```

## ArgMin Operator Integration Example

Below is the complete example code.

```cpp
#include <set>
#include <optional>
#include "ms_extension/all.h"

namespace custom {

/* 1. Infer output shape */
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

/* 2. Construct empty output tensor */
ms::Tensor GenResultTensor(const ms::Tensor &t, int64_t dim, bool keep_dim, ms::TypeId type_id) {
  ShapeVector in_shape = t.shape();
  ShapeVector out_shape = InferArgMinShape(in_shape, dim, keep_dim);
  return ms::Tensor(type_id, out_shape);
}

/* 3. Operator entry: called directly from Python */
ms::Tensor npu_arg_min(const ms::Tensor &x, int64_t dim, bool keep_dim) {
  auto result = GenResultTensor(x, dim, keep_dim, ms::TypeId::kNumberTypeInt64);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ArgMin");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnArgMin, x, dim, keep_dim, result));
  runner->Run({x}, {result});
  return result;
}
}  // namespace custom

/* 4. PYBIND11 interface definition */
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_arg_min", PYBOOST_CALLER(1, custom::npu_arg_min)); }
```

### 1. Infer Operator Output Info

```cpp
auto y = GenResultTensor(x, axis, keep_dims);
```

This step creates the output tensor based on the operator's logic, using `shape` and `type`. For example, `aclnnArgMin` precomputes the output `shape` and `type` based on `axis` and `keep_dims`, and constructs an empty Tensor using `ms::Tensor(dtype, shape)`. This tensor only allocates metadata and does not allocate device memory. `AclnnOpRunner::Run` will allocate device memory internally.

### 2. Create AclnnOpRunner

In [Custom Operator Based on CustomOpBuilder](https://www.mindspore.cn/tutorials/en/r2.7.2/custom_program/operation/op_customopbuilder.html), MindSpore provides the general custom operator integration class `PyboostRunner`. For aclnn operators, users can directly use the `AclnnOpRunner` class to create an object.

```cpp
auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ArgMin");
```

### 3. Call Interface to Execute Operator

```cpp
runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnArgMin, x, axis, keep_dims, y));
runner->Run({x}, {y});
```

In `LAUNCH_ACLNN_FUNC`, pass the operator name, inputs, and outputs in order, and use `SetLaunchFunc` to set the launch function to the runner. Call the `Run` method, with inputs and outputs of type `ms::Tensor`.

### 4. Bind C++ Function to Python via pybind11

```cpp
PYBIND11_MODULE(MS_EXTENSION_NAME, m) { m.def("npu_arg_min", PYBOOST_CALLER(1, custom::npu_arg_min)); }
```

- `npu_arg_min`: Frontend interface name.
- `custom::npu_arg_min`: Actual backend interface being called.
- `PYBOOST_CALLER`: Takes the number of outputs and the backend interface.

### 5. Compile Custom Operator Using CustomOpBuilder

Save the above C++ code as `argmin.cpp`, then compile it using the Python `CustomOpBuilder` interface.

```python
import mindspore as ms
import numpy as np

my_ops = CustomOpBuilder("my_custom", 'argmin.cpp', backend="Ascend").load()
x = np.random.randn(2, 3, 4, 5).astype(np.float32)
output = my_ops.npu_arg_min(ms.Tensor(x), 0, False)
```
