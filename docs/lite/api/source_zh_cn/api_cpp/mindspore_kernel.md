# mindspore::kernel

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_cpp/mindspore_kernel.md)

## 接口汇总

| 类名 | 描述 |
| --- | --- |
| [Kernel](#kernel) | 算子基类。|
| [KernelInterface](#kernelinterface) | 算子扩展能力基类。|
| [MSKernel](#mskernel) | Mindspore Kernel Mindspore算子基类。 |
| [IKernel](#ikernel) | IKernel 算子模板类。 |

## Kernel

\#include <[kernel.h](https://gitee.com/mindspore/mindspore/blob/master/include/api/kernel.h)>

Kernel是算子实现的基类，定义了几个必须实现的接口。继承自IKernel。

### 构造函数

#### Kernel

``` c++
Kernel()

Kernel(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
       const schema::Primitive *primitive, const mindspore::Context *ctx)
```

Kernel的默认与带参构造函数，构造Kernel实例。

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经由flatbuffers反序化为Primitive后的结果。

    - `ctx`: 算子的上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)。

### 析构函数

#### ~Kernel

``` c++
virtual ~Kernel()
```

Kernel的析构函数。

### 公有成员函数

#### InferShape

``` c++
virtual int InferShape()
```

在用户调用`Model::Build`接口时，或是模型推理中需要推理算子形状时，会调用到该接口。
在自定义算子场景中，用户可以覆写该接口，实现自定义算子的形状推理逻辑。详见[自定义算子章节](https://www.mindspore.cn/lite/docs/zh-CN/master/advanced/third_party/register_kernel.html)。
在`InferShape`函数中，一般需要实现算子的形状、数据类型和数据排布的推理逻辑。

- 返回值

    是否成功。

#### type

``` c++
virtual schema::PrimitiveType type()
```

返回算子的类型。

#### quant_type

``` c++
virtual schema::QuantType quant_type()
```

返回算子的量化类型。

## KernelInterface

\#include <[kernel_interface.h](https://gitee.com/mindspore/mindspore-lite/blob/master/mindspore-lite/include/kernel_interface.h)>

算子扩展能力基类。

### ~KernelInterface

``` c++
virtual ~KernelInterface() = default;
```

析构函数。

### KernelInterfaceCreator

``` c++
using KernelInterfaceCreator = std::function<std::shared_ptr<KernelInterface>()>
```

创建[KernelInterface](#kernelinterface)的函数原型声明。

### 公有成员函数

#### Infer

算子用于推测输出shape的方法。

``` c++
virtual Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                       const schema::Primitive *primitive)
```

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经过flatbuffers反序化后的结果，存储算子属性。

- 返回值

    状态码。

#### Infer

``` c++
virtual Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                       const schema::Primitive *primitive, const Kernel *kernel)
```

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经过flatbuffers反序化后的结果，存储算子属性。

    - `kernel`: 对应的算子。

推测自定义算子的输出shape。

## MSKernel

\#include <[kernel.h](https://gitee.com/mindspore/mindspore/blob/master/include/api/kernel_api.h)>

Mindspore Kernel 算子类。是IKernel的父类。

### 构造函数

```cpp
MSKernel() = default;
MSKernel(std::vector<mindspore::MSTensor> inputs, std::vector<mindspore::MSTensor> outputs,
           const mindspore::Context *ctx)

      : context_(ctx), inputs_(std::move(inputs)), outputs_(std::move(outputs)) {}
```

- 参数

    `inputs`: 输入。

    `output`: 输出。

    `ctx`: 算子对应Context。

### 析构函数

```cpp
virtual ~MSKernel() = default;
```

### 公有成员函数

#### InferShape

``` c++
virtual int InferShape()
```

在用户调用`Model::Build`接口时，或是模型推理中需要推理算子形状时，会调用到该接口。
在自定义算子场景中，用户可以覆写该接口，实现自定义算子的形状推理逻辑。详见[自定义算子章节](https://www.mindspore.cn/lite/docs/zh-CN/master/advanced/third_party/register_kernel.html)。
在`InferShape`函数中，一般需要实现算子的形状、数据类型和数据排布的推理逻辑。

- 返回值

    状态码。

#### Prepare

``` c++
virtual int Prepare() = 0;
```

进行算子运行前相关的准备工作，MindSpore Lite 框架运行时会对所有算子执行一遍Prepare后再执行Execute。

- 返回值

    状态码。

#### Execute

``` c++
virtual int Execute() = 0;
```

运行算子。

- 返回值

    状态码。

#### ReSize

``` c++
virtual int ReSize() = 0;
```

在用户调用`Model::Resize`接口时，或是模型推理中需要重新推理算子形状时，会调用到该接口。
在`ReSize`函数中，若有必要，根据输入的形状态重新推理输出形状，并分配算子运算中需要的内存。

- 返回值

    状态码。

#### set_inputs

``` c++
virtual void set_inputs(const std::vector<mindspore::MSTensor> &in_tensors) { this->inputs_ = in_tensors; }
```

设置算子的输入列表。

- 参数

    - `in_tensors`: 算子的所有输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)列表。

#### set_input

``` c++
virtual void set_input(mindspore::MSTensor in_tensor, int index) { this->inputs_[index] = in_tensor; }
```

设置算子指定位置的输入。

- 参数

    - `in_tensor`: 算子的输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `index`: 算子输入在所有输入中的下标，从0开始计数。

#### set_outputs

``` c++
virtual void set_outputs(const std::vector<mindspore::MSTensor> &out_tensors) { this->outputs_ = out_tensors; }
```

设置算子的输出列表。

- 参数

    - `out_tensor`: 算子的所有输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)列表。

#### set_output

``` c++
virtual void set_output(mindspore::MSTensor out_tensor, int index) { this->outputs_[index] = out_tensor; }
```

设置算子指定位置的输出。

- 参数

    - `out_tensor`: 算子的输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `index`: 算子输出在所有输出中的下标，从0开始计数。

#### inputs

``` c++
virtual const std::vector<mindspore::MSTensor *> &inputs()
```

返回算子的所有输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)列表。

- 返回值

    算子的所有输入。

#### outputs

``` c++
virtual const std::vector<mindspore::MSTensor *> &outputs()
```

返回算子的所有输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)列表。

- 返回值

    算子的所有输出。

#### name

``` c++
std::string name() const
```

返回算子的名称。

- 返回值

    算子的名称。

#### set_name

``` c++
void set_name(const std::string &name)
```

设置算子的名称。

- 参数

    - `name`: 算子名称。

#### context

``` c++
const lite::Context *context() const
```

返回算子对应的[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)。

- 返回值

    算子的Context。

#### GetAttr

``` c++
std::string GetAttr(const std::string &key) const
```

获取指定配置名对应的配置。

- 参数

    - `key`: 配置名。

#### SetConfig

``` c++
void SetConfig(const std::map<std::string, std::map<std::string, std::string>> *config)
```

保存配置内容的常量指针到kernel里，该接口当前是由框架在加载配置文件时自动触发调用的，不建议用户使用。

- 参数

    - `config`: 配置的常量指针。

#### GetConfig

``` c++
std::map<std::string, std::string> GetConfig(const std::string &section) const
```

获取指定章节名对应的配置。

- 参数

    - `section`: 配置的章节名称。

### 保护成员变量

```cpp
std::string name_;
const mindspore::Context *context_ = nullptr;
std::vector<mindspore::MSTensor> inputs_;
std::vector<mindspore::MSTensor> outputs_;
std::map<std::string, std::string> attrs_;
const std::map<std::string, std::map<std::string, std::string>> *config_ = nullptr;
```

name_: 名字

context_: 训练context。

inputs_: 输入。

outputs_: 输出。

attrs_: 属性。

config_: 配置。

### 保护成员函数

```cpp
void SetAttr(const std::string &key, const std::string &value) { attrs_[key] = value; }
```

设置算子的属性。

- 参数

    - `key`: 属性键。

    - `value`: 属性值。

## IKernel

继承自[MSKernel](#mskernel)

### 构造函数

```cpp
IKernel() = default;
IKernel(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
          const Primitive *primitive, const mindspore::Context *ctx)
      : MSKernel(inputs, outputs, ctx), primitive_(primitive) {}
```

- 参数

    - `inputs`: 输入Tensor。

    - `outputs`: 输出Tensor。

    - `primitive`: 算子经过flatbuffers反序化后的结果，存储算子属性。

    - `ctx`: 计算context。

### 析构函数

```cpp
~IKernel() override = default;
```

### 公有成员函数

#### Primitive

```cpp
const Primitive *primitive() const { return this->primitive_; }
```

获取IKernel算子的primitive。

- 返回值

    IKernel算子经过flatbuffers反序化后的结果。

### 保护成员函数

```cpp
const Primitive *primitive_ = nullptr;
```

IKernel算子经过flatbuffers反序化后的结果。