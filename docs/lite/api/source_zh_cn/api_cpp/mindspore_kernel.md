# mindspore::kernel

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_cpp/mindspore_kernel.md)

## 接口汇总

| 类名 | 描述 |
| --- | --- |
| [Kernel](#kernel) | 算子基类。|
| [KernelInterface](#kernelinterface) | 算子扩展能力基类。|

## Kernel

\#include <[kernel.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/include/api/kernel.h)>

Kernel是算子实现的基类，定义了几个必须实现的接口。

## 构造函数

### Kernel

``` c++
Kernel()

Kernel(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
       const schema::Primitive *primitive, const mindspore::Context *ctx)
```

Kernel的默认与带参构造函数，构造Kernel实例。

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经由flatbuffers反序化为Primitive后的结果。

    - `ctx`: 算子的上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#context)。

## 析构函数

### ~Kernel

``` c++
virtual ~Kernel()
```

Kernel的析构函数。

## 公有成员函数

### Prepare

``` c++
virtual int Prepare()
```

进行算子运行前相关的准备工作，MindSpore Lite 框架运行时会对所有算子执行一遍Prepare后再执行Execute。

### Execute

``` c++
virtual int Execute()
```

运行算子。

### ReSize

``` c++
virtual int ReSize()
```

在用户调用`Model::Resize`接口时，或是模型推理中需要重新推理算子形状时，会调用到该接口。
在`ReSize`函数中，若有必要，根据输入的形状态重新推理输出形状，并分配算子运算中需要的内存。

### InferShape

``` c++
virtual int InferShape()
```

在用户调用`Model::Build`接口时，或是模型推理中需要推理算子形状时，会调用到该接口。
在自定义算子场景中，用户可以覆写该接口，实现自定义算子的形状推理逻辑。详见[自定义算子章节](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/register_kernel.html)。
在`InferShape`函数中，一般需要实现算子的形状、数据类型和数据排布的推理逻辑。

### type

``` c++
virtual schema::PrimitiveType type()
```

返回算子的类型。

### quant_type

``` c++
virtual schema::QuantType quant_type()
```

返回算子的量化类型。

### set_inputs

``` c++
virtual void set_inputs(const std::vector<mindspore::MSTensor> &in_tensors)
```

设置算子的输入列表。

- 参数

    - `in_tensors`: 算子的所有输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)列表。

### set_input

``` c++
virtual set_input(mindspore::MSTensor in_tensor, int index)
```

设置算子指定位置的输入。

- 参数

    - `in_tensor`: 算子的输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `index`: 算子输入在所有输入中的下标，从0开始计数。

### set_outputs

``` c++
virtual void set_outputs(const std::vector<mindspore::MSTensor> &out_tensors)
```

设置算子的输出列表。

- 参数

    - `out_tensor`: 算子的所有输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)列表。

### set_output

``` c++
virtual void set_output(mindspore::MSTensor out_tensor, int index)
```

设置算子指定位置的输出。

- 参数

    - `out_tensor`: 算子的输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `index`: 算子输出在所有输出中的下标，从0开始计数。

### inputs

``` c++
virtual const std::vector<mindspore::MSTensor *> &inputs()
```

返回算子的所有输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)列表。

### outputs

``` c++
virtual const std::vector<mindspore::MSTensor *> &outputs()
```

返回算子的所有输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)列表。

### name

``` c++
std::string name()
```

返回算子的名称。

### set_name

``` c++
void set_name(const std::string &name)
```

设置算子的名称。

- 参数

    - `name`: 算子名称。

### context

``` c++
const lite::Context *context() const
```

返回算子对应的[Context](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#context)。

### primitive

``` c++
const schema::Primitive *primitive() const
```

返回算子经由flatbuffers反序化为Primitive后的结果。

### GetAttr

``` c++
std::string GetAttr(const std::string &key) const
```

获取指定配置名对应的配置。

- 参数

    - `key`: 配置名。

### SetConfig

``` c++
void SetConfig(const std::map<std::string, std::map<std::string, std::string>> *config)
```

保存配置内容的常量指针到kernel里，该接口当前是由框架在加载配置文件时自动触发调用的，不建议用户使用。

- 参数

    - `config`: 配置的常量指针。

### GetConfig

``` c++
std::map<std::string, std::string> GetConfig(const std::string &section) const
```

获取指定章节名对应的配置。

- 参数

    - `section`: 配置的章节名称。

## KernelInterface

\#include <[kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/kernel_interface.h)>

算子扩展能力基类。

### ~KernelInterface

``` c++
virtual ~KernelInterface()
```

析构函数。

### KernelInterfaceCreator

``` c++
using KernelInterfaceCreator = std::function<std::shared_ptr<KernelInterface>()>
```

创建[KernelInterface](#kernelinterface)的函数原型声明。

### 公有成员函数

#### Infer

算子的InferShape能力，用于根据输入推导出输出的形状、数据类型以及format。

``` c++
virtual int Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive, const Kernel *kernel)
```

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经过flatbuffers反序化后的结果，存储算子属性。

    - `kernel`: 算子的基类结构，在Build阶段，kernel是空指针，Build完成后框架传递的kernel才有值，当kernel非空时就不建议去操作primitive了，因为有可能primtive已经无效了。

#### Infer

算子的InferShape能力，用于根据输入推导出输出的shape、数据类型以及format。

该接口已不推荐使用，建议使用带有kernel参数的Infer接口。因为如果模型通过以下Build接口执行编译，编译后框架会自动释放模型的内存，导致primitive不可用。

``` c++
Status Build(GraphCell graph, const std::shared_ptr<Context> &model_context = nullptr,
               const std::shared_ptr<TrainCfg> &train_cfg = nullptr)
```

``` c++
virtual int Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive)
```

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经过flatbuffers反序化后的结果，存储算子属性。
