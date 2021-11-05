# mindspore::kernel

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_cpp/mindspore_kernel.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 接口汇总

| 类名 | 描述 |
| --- | --- |
| [Kernel](#kernel) | 算子基类。|
| [KernelInterface](#kernelinterface) | 算子扩展能力基类。|

## Kernel

\#include <[kernel.h](https://gitee.com/mindspore/mindspore/tree/master/include/api/kernel.h)>

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

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经由flatbuffers反序化为Primitive后的结果。

    - `ctx`: 算子的上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)。

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

根据输入的形状态重新分配算子需要的内存。

### type

``` c++
virtual schema::PrimitiveType type()
```

返回算子的类型。

### set_inputs

``` c++
virtual void set_inputs(const std::vector<mindspore::MSTensor> &in_tensors)
```

设置算子的输入列表。

- 参数

    - `in_tensors`: 算子的所有输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)列表。

### set_input

``` c++
virtual set_input(mindspore::MSTensor in_tensor, int index)
```

设置算子指定位置的输入。

- 参数

    - `in_tensor`: 算子的输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `index`: 算子输入在所有输入中的下标，从0开始计数。

### set_outputs

``` c++
virtual void set_outputs(const std::vector<mindspore::MSTensor> &out_tensors)
```

设置算子的输出列表。

- 参数

    - `out_tensor`: 算子的所有输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)列表。

### set_output

``` c++
virtual void set_output(mindspore::MSTensor out_tensor, int index)
```

设置算子指定位置的输出。

- 参数

    - `out_tensor`: 算子的输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `index`: 算子输出在所有输出中的下标，从0开始计数。

### inputs

``` c++
virtual const std::vector<mindspore::MSTensor *> &inputs()
```

返回算子的所有输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)列表。

### outputs

``` c++
virtual const std::vector<mindspore::MSTensor *> &outputs()
```

返回算子的所有输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)列表。

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

返回算子对应的[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)。

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

\#include <[kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/kernel_interface.h)>

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

算子的InferShape能力，用于根据输入推导出输出的shape、数据类型以及format。

``` c++
virtual int Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs, const schema::Primitive *primitive)
```

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经过flatbuffers反序化后的结果，存储算子属性。

