# mindspore::lite（registry）

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/api_cpp/source_zh_cn/registry.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## FmkType

\#include &lt;[framework.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/framework.h)&gt;

FmkType枚举类定义了转换工具支持的框架类型。

## ConverterParameters

\#include &lt;[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/model_parser_registry.h)&gt;

ConverterParameters结构体定义了模型解析时的转换参数。

```c++
struct ConverterParameters {
  FmkType fmk_;                                   // 框架类型
  schema::QuantType quant_type_;                  // 模型量化类型
  std::string model_file_;                        // 原始模型文件路径
  std::string weight_file_;                       // 原始模型权重文件路径，仅在Caffe框架下有效
  std::map<std::string, std::string> attrs_;      // 预留参数接口，暂未启用
};
```

## ModelParser

\#include &lt;[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/model_parser_registry.h)&gt;

ModelParser类定义了解析原始模型的基类。

## ModelParserCreator

\#include &lt;[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/model_parser_registry.h)&gt;

ModelParserCreator定义了创建ModelParser的函数指针类型。

```c++
typedef ModelParser *(*ModelParserCreator)()
```

- 返回值

  指向ModelParser的类指针。

## ModelParserRegistry

\#include &lt;[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/model_parser_registry.h)&gt;

ModelParserRegistry类用于注册及获取ModelParserCreator类型的函数指针。

### 构造函数和析构函数

#### ModelParserRegistry

```c++
ModelParserRegistry()
```

用默认参数构造ModelParserRegistry 对象。

#### ~ModelParserRegistry

```c++
~ModelParserRegistry()
```

ModelParserRegistry 的析构函数。

### 公有成员函数

#### GetModelParser

```c++
ModelParser *GetModelParser(const FmkType fmk)
```

获取ModelParserCreator类型的函数指针。

- 参数

    - `fmk`: 定义了ModelParserCreator类型的函数指针的框架类型[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/registry.html#fmktype)。

- 返回值

  ModelParserCreator类型的函数指针。

#### RegParser

```c++
void RegParser(const FmkType fmk, ModelParserCreator creator)
```

注册ModelParserCreator类型的函数指针。

- 参数

    - `fmk`: 定义了ModelParserCreator类型的函数指针的框架类型[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/registry.html#fmktype)。

    - `creator`: 定义了ModelParserCreator类型的函数指针。

### 静态公有成员函数

#### GetInstance

```c++
static ModelParserRegistry *GetInstance()
```

创建ModelParserRegistry单例的静态方法。

- 返回值

  指向ModelParserRegistry单例指针。

### 公有属性

#### parsers_

```c++
parsers_
```

**unordered_map&lt;FmkType, ModelParserCreator&gt;**值，存储ModelParserCreator类的函数指针。

## ModelRegistrar

\#include &lt;[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/model_parser_registry.h)&gt;

ModelRegistrar类定义了ModelParserCreator类的注册形式，用于程序运行时程序主动注册ModelParserCreator类。

### 构造函数和析构函数

#### ModelRegistrar

```c++
ModelRegistrar(const FmkType fmk, ModelParserCreator creator)
```

ModelRegistrar的构造函数。

- 参数

    - `fmk`: 定义了ModelParserCreator类型的函数指针的框架类型[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/registry.html#fmktype)。

    - `creator`: 定义了ModelParserCreator类型的函数指针。

#### ~ModelRegistrar

```c++
~ModelRegistrar()
```

ModelRegistrar的析构函数。

## REG_MODEL_PARSER

\#include &lt;[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/model_parser_registry.h)&gt;

REG_MODEL_PARSER定义了ModelParserCreator类的注册宏，便于ModelParserCreator类的注册。

```c++
#define REG_MODEL_PARSER(fmk, parserCreator)
```

- 参数

    - `fmk`: 定义了ModelParserCreator类型的函数指针的框架类型[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/registry.html#fmktype)。

    - `creator`: 定义了ModelParserCreator类型的函数指针。

> 用户自定义的ModelParser，框架类型也必须满足设定支持的类型[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/registry.html#fmktype)。

## Pass

\#include &lt;[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/pass_registry.h)&gt;

前置声明，Pass定义了图优化的基类。以供用户继承，自定义Pass。用户自定义Pass需重载以下虚函数：

```c++
virtual bool Run(const FuncGraphPtr &func_graph) = 0;
```

## PassPtr

\#include &lt;[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/pass_registry.h)&gt;

PassPtr定义了Pass的共享智能指针。

```c++
using PassPtr = std::shared_ptr<Pass>
```

## PassPosition

\#include &lt;[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/pass_registry.h)&gt;

PassPosition枚举类定义了用户自定义Pass的放置位置。

```c++
enum PassPosition {
     POSITION_BEGIN = 0,    // 用户自定义Pass置于内置融合Pass前
     POSITION_END = 1       // 用户自定义Pass置于内置融合Pass后
     };
```

> 同一个位置只能存放一个Pass类。如果同一位置，用户自定义了多个Pass类时，用户可通过组合的形式，将多个Pass类封装成一个Pass类。

## PassRegistry

\#include &lt;[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/pass_registry.h)&gt;

PassRegistry类用于注册及获取Pass类实例。

### 构造函数和析构函数

#### PassRegistry

```c++
PassRegistry()
```

用默认参数构造PassRegistry 对象。

#### ~PassRegistry

```c++
~PassRegistry()
```

PassRegistry 的析构函数。

### 公有成员函数

#### GetPasses

```c++
const std::unordered_map<int, PassPtr> &GetPasses() const
```

获取所有的Pass类实例。

- 返回值

  unordered_map<int, PassPtr>类型。key直为PassPosition类，value直为Pass类实例。

#### RegPass

```c++
void RegPass(int position, const PassPtr &pass)
```

注册Pass类实例。

- 参数

    - `int`: 定义了Pass类实例的放置位置[PassPosition](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.3/registry.html#passposition)。

    - `pass`: 定义了Pass类实例。

### 静态公有成员函数

#### GetInstance

```c++
static PassRegistry *GetInstance()
```

创建PassRegistry单例的静态方法。

- 返回值

  指向PassRegistry单例指针。

### 私有属性

#### passes_

```c++
passes_
```

**unordered_map&lt;int,PassPtr&gt;**值，存储Pass类实例。

#### mutex_

```c++
mutex_
```

**mutex**类，引入琐机制，避免竞争。

## PassRegistrar

\#include &lt;[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/pass_registry.h)&gt;

PassRegistrar类定义了Pass类的注册形式，用于程序运行时程序主动注册Pass类。

### 构造函数和析构函数

#### PassRegistrar

```c++
PassRegistrar(int pos, const PassPtr &pass)
```

ModelRegistrar的构造函数。

- 参数

    - `pos`: 定义了Pass类实例的放置位置[PassPosition](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.3/registry.html#passposition)。

    - `pass`: 定义了Pass类实例。

#### ~PassRegistrar

```c++
~PassRegistrar()
```

PassRegistrar的析构函数。

## REG_PASS

\#include &lt;[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/pass_registry.h)&gt;

REG_PASS定义了Pass类的注册宏，便于Pass类的注册。

```c++
#define REG_PASS(position, pass)
```

- 参数

    - `pos`: 定义了Pass类实例的放置位置，PassPosition枚举类[PassPosition](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.3/registry.html#passposition)。

    - `pass`: 定义了Pass类实例。

## CreateKernel

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/register_kernel.h)>

创建Kernel的函数原型

``` c++
using CreateKernel = std::function<std::shared_ptr<kernel::Kernel>(
  const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
  const schema::Primitive *primitive, const lite::Context *ctx)>;
```

- 参数

    - `inputs`: 输入tensor[tensor::MSTensor](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.3/tensor.html)。

    - `outputs`: 输出tensor[tensor::MSTensor](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.3/tensor.html)。

    - `primitive`: 算子经过flatbuffer反序化后的结果，存储算子属性。

    - `ctx`: 上下文配置[lite::Context](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.3/lite.html#Context)。

## REGISTER_KERNEL

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/register_kernel.h)>

注册算子。

``` c++
#define REGISTER_KERNEL(arch, provider, data_type, op_type, creator)
```

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/core/ir/dtype/type_id.h)中。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建算子的函数指针，具体见CreateKernel的说明。

## REGISTER_CUSTOM_KERNEL

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/register_kernel.h)>

注册Custom算子

``` c++
#define REGISTER_CUSTOM_KERNEL(arch, provider, data_type, op_type, creator)
```

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/core/ir/dtype/type_id.h)中。

    - `op_type`: 算子类型，由用户自定义，确保唯一即可。

    - `creator`: 创建算子的函数指针，具体见CreateKernel的说明。

## KernelInterface

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/kernel_interface.h)>

算子的统一接口。

### 公有成员函数

#### Infer

算子的InferShape能力，用于根据输入推导出输出的shape、数据类型以及format。

``` c++
virtual int Infer(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
                    const schema::Primitive *primitive)
```

- 参数

    - `inputs`: 输入tensor[tensor::MSTensor](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.3/tensor.html)。

    - `outputs`: 输出tensor[tensor::MSTensor](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.3/tensor.html)。

    - `primitive`: 算子经过flatbuffer反序化后的结果，存储算子属性。

## KernelInterfaceCreator

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/kernel_interface.h)>

创建KernelInterface的函数原型

``` c++
using KernelInterfaceCreator = std::function<std::shared_ptr<KernelInterface>()>;
```

## REGISTER_KERNEL_INTERFACE

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/kernel_interface.h)>

注册KernelInterface的实现。

``` c++
#define REGISTER_KERNEL_INTERFACE(provider, op_type, creator)
```

- 参数
    - `provider`: 产商，由用户自定义。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建KernelInterface的函数指针，具体见KernelInterfaceCreator的说明。

## REGISTER_CUSTOM_KERNEL_INTERFACE

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/registry/kernel_interface.h)>

注册Custom算子对应的KernelInterface实现。

``` c++
#define REGISTER_CUSTOM_KERNEL_INTERFACE(provider, op_type, creator)
```

- 参数

    - `provider`: 产商名，由用户自定义。

    - `op_type`: 算子类型，由用户自定义，确保唯一同时要与REGISTER_CUSTOM_KERNEL时注册的op_type保持一致。

    - `creator`: 创建算子的函数指针，具体见KernelInterfaceCreator的说明。
