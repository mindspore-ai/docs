# Template Class DelegateModel

\#include &lt;[delegate.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.9/include/api/delegate.h)&gt;

`DelegateModel`定义了MindSpore Lite Delegate机制操作的模型对象。

## 构造函数

```cpp
DelegateModel(std::vector<kernel::Kernel *> *kernels, const std::vector<MSTensor> &inputs,
              const std::vector<MSTensor> &outputs,
              const std::map<kernel::Kernel *, const schema::Primitive *> &primitives, SchemaVersion version)
```

## 析构函数

```cpp
~DelegateModel() = default
```

## 保护成员

### kernels_

```cpp
std::vector<kernel::Kernel *> *kernels_
```

[**Kernel**](https://www.mindspore.cn/lite/api/zh-CN/r2.9.0/api_cpp/mindspore_kernel.html#kernel)的列表，保存模型的所有算子。

### inputs_

```cpp
const std::vector<mindspore::MSTensor> &inputs_
```

[**MSTensor**](./classmindspore_MSTensor.md)的列表，保存这个算子的输入tensor。

### outputs_

```cpp
const std::vector<mindspore::MSTensor> &outputs
```

[**MSTensor**](./classmindspore_MSTensor.md)的列表，保存这个算子的输出tensor。

### primitives_

```cpp
const std::map<kernel::Kernel *, const schema::Primitive *> &primitives_
```

[**Kernel**](https://www.mindspore.cn/lite/api/zh-CN/r2.9.0/api_cpp/mindspore_kernel.html#kernel)和**schema::Primitive**的Map，保存所有算子的属性。

### version_

```cpp
SchemaVersion version_
```

**enum**值，当前执行推理的模型的版本[SchemaVersion](./enum_mindspore_SchemaVersion-1.md)。

## 公有成员函数

### GetPrimitive

```cpp
const schema::Primitive *GetPrimitive(kernel::Kernel *kernel) const
```

获取一个Kernel的属性值。

- 参数

    - `kernel`: 指向Kernel的指针。

- 返回值

  const schema::Primitive *，输入参数Kernel对应的该算子的属性值。

### BeginKernelIterator

```cpp
KernelIter BeginKernelIterator()
```

返回DelegateModel Kernel列表起始元素的迭代器。

- 返回值

  **KernelIter**，指向DelegateModel Kernel列表起始元素的迭代器。

### EndKernelIterator

```cpp
KernelIter EndKernelIterator()
```

返回DelegateModel Kernel列表末尾元素的迭代器。

- 返回值

  **KernelIter**，指向DelegateModel Kernel列表末尾元素的迭代器。

### Replace

```cpp
KernelIter Replace(KernelIter from, KernelIter end, kernel::Kernel *graph_kernel)
```

用Delegate子图Kernel替换Delegate支持的连续Kernel列表。

- 参数

    - `from`: Delegate支持的连续Kernel列表的起始元素迭代器。
    - `end`: Delegate支持的连续Kernel列表的末尾元素迭代器。
    - `graph_kernel`: 指向Delegate子图Kernel实例的指针。

- 返回值

  **KernelIter**，用Delegate子图Kernel替换之后，子图Kernel下一个元素的迭代器，指向下一个未被访问的Kernel。

### nodes

```cpp
std::vector<kernel::Kernel *> *nodes()
```

获取nodes。

- 返回值

  Kernel组成的vector。

### inputs

```cpp
const std::vector<mindspore::MSTensor> &inputs()
```

返回DelegateModel输入tensor列表。

- 返回值

  [**MSTensor**](./classmindspore_MSTensor.md)的列表。

### outputs

```cpp
const std::vector<mindspore::MSTensor> &outputs()
```

返回DelegateModel输出tensor列表。

- 返回值

  [**MSTensor**](./classmindspore_MSTensor.md)的列表。

### GetVersion

```cpp
const SchemaVersion GetVersion()
```

返回当前执行推理的模型文件的版本。

- 返回值

  **enum**值，0: r1.2及r1.2之后的版本，1: r1.1及r1.1之前的版本，-1: 无效版本。
