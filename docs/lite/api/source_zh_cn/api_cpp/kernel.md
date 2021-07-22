# mindspore::kernel::Kernel

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_cpp/kernel.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## Kernel

\#include <[kernel.h](https://gitee.com/mindspore/mindspore/tree/master/include/api/kernel.h)>

Kernel是算子实现的基类，定义了几个必须实现的接口。

## 构造函数

### Kernel

``` c++
Kernel()

Kernel(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
         const schema::Primitive *primitive, const lite::Context *ctx)
```

Kernel的默认与带参构造函数，构造Kernel实例。

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

### inputs

``` c++
virtual const std::vector<mindspore::tensor::MSTensor *> &inputs()
```

返回输入Tensor。

### outputs

``` c++
virtual const std::vector<mindspore::tensor::MSTensor *> &outputs()
```

返回输出Tensor。

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

### context

``` c++
const lite::Context *context() const
```

返回算子对应的[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/lite.html)。

### primitive

``` c++
const schema::Primitive *primitive() const
```

返回算子反序化为Primitive后的结果。
