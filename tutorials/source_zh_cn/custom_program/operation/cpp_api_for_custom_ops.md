# 自定义算子的C++接口说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0rc1/tutorials/source_zh_cn/custom_program/operation/cpp_api_for_custom_ops.md)

## 概述

MindSpore自定义算子的C++接口分为两类：

1. **API 接口**：
   标记为 【API】 的接口是稳定的公开接口，供用户直接使用。这些接口经过充分测试，功能明确，向后兼容性较高。

2. **实验性接口**：
   未标记为 【API】 的接口为实验性接口。这些接口可能会在未来版本中发生变动或被移除，使用时需谨慎。

自定义算子开发时，通过 `#include "ms_extension/api.h"` 即可引用下方接口涉及的头文件，无需关注每个接口的具体位置。

## namespace ms

### enum TypeId

`TypeId` 枚举类型定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/v2.7.0-rc1/mindspore/core/include/mindapi/base/type_id.h)头文件中，定义了 MindSpore 中支持的张量数据类型，包括布尔值、整数类型、浮点数类型、复数类型等。

此接口也被包含在 `namespace ms`中，通过`ms::TypeId`也可以访问。

```cpp
kNumberTypeBegin,       // Number 类型起始值
kNumberTypeBool,        // 布尔类型
kNumberTypeInt,         // 默认整数类型
kNumberTypeInt8,        // 8 位有符号整数
kNumberTypeInt16,       // 16 位有符号整数
kNumberTypeInt32,       // 32 位有符号整数
kNumberTypeInt64,       // 64 位有符号整数
kNumberTypeUInt,        // 默认无符号整数类型
kNumberTypeUInt8,       // 8 位无符号整数
kNumberTypeUInt16,      // 16 位无符号整数
kNumberTypeUInt32,      // 32 位无符号整数
kNumberTypeUInt64,      // 64 位无符号整数
kNumberTypeFloat,       // 默认浮点数类型
kNumberTypeFloat16,     // 16 位半精度浮点数
kNumberTypeFloat32,     // 32 位单精度浮点数
kNumberTypeFloat64,     // 64 位双精度浮点数
kNumberTypeBFloat16,    // 16 位脑浮点数
kNumberTypeDouble,      // 双精度浮点数（等价于 kNumberTypeFloat64 ）
kNumberTypeComplex,     // 默认复数类型
kNumberTypeComplex64,   // 64 位复数（由2 个 32 位浮点数组成）
kNumberTypeComplex128,  // 128 位复数（由2 个 64 位浮点数组成）
kNumberTypeInt4,        // 4 位有符号整数
kNumberTypeGLUInt,      // OpenGL 无符号整数类型
kNumberTypeEnd,         // Number 类型结束值
```

### class Tensor

张量类定义在[tensor.h](https://gitee.com/mindspore/mindspore/blob/v2.7.0-rc1/mindspore/ccsrc/ms_extension/common/tensor.h)头文件中，表示 MindSpore 的张量对象，提供操作和查询张量属性的方法。

#### 构造函数

- **Tensor()**
    - **描述**：【API】 构造一个未定义的占位符张量。

- **Tensor(TypeId, const ShapeVector &)**

  ```cpp
  Tensor(TypeId type_id, const ShapeVector &shape)
  ```

    - **描述**：【API】 根据指定的数据类型和形状构造一个张量。
    - **参数**：
        - `type_id`：张量的数据类型。
        - `shape`：张量的形状，表示为整数向量。

- **Tensor(const mindspore::ValuePtr &)**

  ```cpp
  Tensor(const mindspore::ValuePtr &value)
  ```

    - **描述**：从给定的 `ValuePtr` 构造一个张量对象。
    - **参数**：
        - `value`：指向 MindSpore `Value` 对象的智能指针。如果值为 `nullptr`，则构造未定义的张量。

#### 公共方法（属性及配置）

- **is_defined()**

  ```cpp
  bool is_defined() const
  ```

    - **描述**：【API】 检查张量是否已定义。
    - **返回值**：如果张量已定义返回 `true`，否则返回 `false`。

- **data_type()**

    ```cpp
    TypeId data_type() const
    ```

    - **描述**：【API】 获取张量的数据类型。
    - **返回值**：张量的数据类型。

- **shape()**

  ```cpp
  const ShapeVector &shape() const
  ```

    - **描述**：【API】 获取张量的形状。
    - **返回值**：张量形状的引用（ `ShapeVector` , 即 `std::vector<int64_t>` ）。

- **numel()**

  ```cpp
  size_t numel() const
  ```

    - **描述**：【API】 返回张量中的元素总数。
    - **返回值**：元素的总数。

- **stride()**

  ```cpp
  std::vector<int64_t> stride() const
  ```

    - **描述**：【API】 计算张量的步幅。
    - **返回值**：表示张量每个维度步幅的向量。

- **storage_offset()**

  ```cpp
  int64_t storage_offset() const
  ```

    - **描述**：【API】 获取张量的存储偏移量。
    - **返回值**：从存储开始的偏移量（以元素为单位）。

- **is_contiguous()**

  ```cpp
  bool is_contiguous() const
  ```

    - **描述**：【API】 检查张量在内存中的存储是否连续。
    - **返回值**：如果张量存储连续返回 `true`，否则返回 `false`。

- **SetNeedContiguous(bool)**

  ```cpp
  void SetNeedContiguous(bool flag) const
  ```

    - **描述**：【API】 设置张量是否需要连续存储空间。
    - **参数**：
        - `flag`：一个布尔值，表示张量是否需要连续存储。

- **GetDataPtr()**

  ```cpp
  void *GetDataPtr() const
  ```

    - **描述**：【API】 获取指向张量数据的指针。
    - **返回值**：指向张量数据的 `void` 指针。
    - **注意**：返回的指针已经包含了 `storage_offset()` 接口指示的偏移量。

#### 公共方法（内部流程）

以下方法非API，仅在内部模块流程中使用，但由于语法限制，需要设置成公共方法。不推荐用户直接使用。

- **need_contiguous()**

  ```cpp
  bool need_contiguous() const
  ```

    - **描述**：检查张量是否需要连续存储空间。
    - **返回值**：如果张量需要连续存储返回 `true`，否则返回 `false`。

- **stub_node()**

  ```cpp
  const mindspore::ValuePtr &stub_node() const
  ```

    - **描述**：获取与张量关联的存根节点。
    - **返回值**：指向存根节点（`ValuePtr`）的智能指针。

- **tensor()**

  ```cpp
  const mindspore::tensor::TensorPtr &tensor() const
  ```

    - **描述**：获取底层张量对象。
    - **返回值**：指向 `TensorPtr` 对象的智能指针。

- **ConvertStubNodeToTensor()**

  ```cpp
  void ConvertStubNodeToTensor() const
  ```

    - **描述**：将存根节点转换为张量对象。
    - **行为**：确保张量从其存根表示中完全实现。转换完成后，存根节点被释放。

## namespace ms::pynative

### class PyboostRunner

PyNative 流程的运行器类，定义在[pyboost_extension.h](https://gitee.com/mindspore/mindspore/blob/v2.7.0-rc1/mindspore/ccsrc/ms_extension/pynative/pyboost_extension.h)头文件中，为管理执行、内存分配和内核启动提供方法。

`PyboostRunner` 是 `std::enable_shared_from_this` 的子类，需要使用智能指针 `std::shared_ptr` 管理其对象。

#### 构造函数

- **PyboostRunner(const std::string &)**

  ```cpp
  PyboostRunner(const std::string &op_name)
  ```

    - **描述**：【API】 构造一个 `PyboostRunner`。
    - **参数**：
        - `op_name`：算子名。

#### 静态公共方法

- **static Call(FuncType, Args &&...)**

  ```cpp
  template <int OUT_NUM, typename FuncType, typename... Args>
  static py::object Call(FuncType func, Args &&... args)
  ```

    - **描述**：【API】 执行给定函数并将其输出转成Python对象。
    - **模板参数**：
        - `OUT_NUM`：算子输出个数，需与 `func` 返回的tensor列表长度一致。暂不支持可变输出个数的场景。
        - `FuncType`：算子入口函数原型，可通过函数入参自动识别。
        - `Args`：算子入参类型，可通过函数入参自动识别。传参顺序需与 `func` 的参数顺序一致。
    - **参数**：
        - `func`：要执行的函数。
        - `args`：函数执行所需的参数。
    - **返回值**：表示算子输出的 Python 对象。

#### 公共方法

- **Run(const std::vector<Tensor> &, const std::vector<Tensor> &)**

  ```cpp
  void Run(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs)
  ```

    - **描述**：【API】 使用指定的输入和输出运行算子。
    - **参数**：
        - `inputs`：输入张量列表。
        - `outputs`：输出张量列表。

- **CalcWorkspace()**

  ```cpp
  virtual size_t CalcWorkspace()
  ```

    - **描述**：【API】 计算算子所需的工作区大小。
    - **返回值**：工作区大小（字节）。默认值为 0。

- **LaunchKernel()**

  ```cpp
  virtual void LaunchKernel() = 0;
  ```

    - **描述**：【API】 启动算子的内核函数。

- **op_name()**

  ```cpp
  const std::string &op_name() const
  ```

    - **描述**：【API】 获取与运行器关联的算子名称。
    - **返回值**：算子名称字符串。

- **inputs()**

  ```cpp
  const std::vector<ms::Tensor> &inputs() const
  ```

    - **描述**：【API】 获取输入张量列表。
    - **返回值**：输入张量的引用。

- **outputs()**

  ```cpp
  const std::vector<ms::Tensor> &outputs() const
  ```

    - **描述**：【API】 获取输出张量列表。
    - **返回值**：输出张量的引用。

- **stream_id()**

  ```cpp
  uint32_t stream_id() const
  ```

    - **描述**：【API】 获取与运行器关联的流ID。
    - **返回值**：流ID。

- **stream()**

  ```cpp
  void *stream()
  ```

    - **描述**：【API】 获取与运行器关联的流指针。
    - **返回值**：流指针。

- **workspace_ptr()**

  ```cpp
  void *workspace_ptr()
  ```

    - **描述**：【API】 获取算子的工作区指针。
    - **返回值**：工作区内存的指针。

### class AtbOpRunner

用于执行 Ascend Transformer Boost (ATB) 算子的运行器类，定义在[atb_common.h](https://gitee.com/mindspore/mindspore/blob/v2.7.0-rc1/mindspore/ccsrc/ms_extension/ascend/atb/atb_common.h)头文件中。

此类继承自 `PyboostRunner`，并封装了 ATB 算子的调用流程，包括初始化和运行 ATB 算子、管理输入输出 Tensor、内存分配及内核调度。

可以查看教程 [CustomOpBuilder通过AtbOpRunner接入ATB算子](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/custom_program/operation/op_customopbuilder_atb.html) 获取使用方法。

#### 构造函数

- **AtbOpRunner**

  ```cpp
  using PyboostRunner::PyboostRunner;
  ```

  继承自 `PyboostRunner` 的构造函数。

#### 公共方法

- **Init(const ParamType&)**

  ```cpp
  template <typename ParamType>
  void Init(const ParamType &param)
  ```

    - **描述**： 【API】 使用给定参数初始化 ATB 算子。此方法通过 `atb::CreateOperation` 创建对应算子的 `atb::Operation` 实例，并将其放入缓存中。对于`param`哈希值相同的算子，只会创建一份 `atb::Operation` 实例。
    - **参数**
        - `param`：用于配置 ATB 算子的参数。
    - **注意**： 对于传入的ParamType类型，需提前特例化 `template <> struct HashOpParam<ParamType>::operator()` 实例函数。

### function RunAtbOp

动态图执行ATB算子的接口，定义在[atb_common.h](https://gitee.com/mindspore/mindspore/blob/v2.7.0-rc1/mindspore/ccsrc/ms_extension/ascend/atb/atb_common.h)头文件中。

```cpp
template <typename ParamType>
void RunAtbOp(const std::string &op_name, const ParamType &param, const std::vector<Tensor> &inputs,
              const std::vector<Tensor> &outputs)
```

【API】 使用提供的参数、输入和输出执行一个 ATB 算子。此函数是对 `AtbOpRunner` 的一层封装。

- **参数**
    - `op_name`：要执行的 ATB 算子名称。
    - `param`：初始化 ATB 算子所需的参数。
    - `inputs`：算子的输入 Tensor 列表。
    - `outputs`：算子的输出 Tensor 列表。
