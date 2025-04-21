# mindspore::registry

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_cpp/mindspore_registry.md)

## 接口汇总

| 类名 | 描述 |
| --- | --- |
| [NodeParserRegistry](#nodeparserregistry) | 扩展Node解析的注册类。|
| [REG_NODE_PARSER](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#reg-node-parser) | 注册扩展Node解析。|
| [ModelParserRegistry](#modelparserregistry) | 扩展Model解析的注册类。|
| [REG_MODEL_PARSER](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#reg-model-parser) | 注册扩展Model解析。|
| [PassBase](#passbase) | Pass的基类。|
| [PassPosition](#passposition) | 扩展Pass的运行位置。|
| [PassRegistry](#passregistry) | 扩展Pass注册构造类。|
| [REG_PASS](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#reg-pass) | 注册扩展Pass。|
| [REG_SCHEDULED_PASS](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#reg-scheduled-pass) | 注册扩展Pass的调度顺序。|
| [RegisterKernel](#registerkernel) | 算子注册实现类。|
| [KernelReg](#kernelreg) | 算子注册构造类。|
| [REGISTER_KERNEL](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-kernel) | 注册算子。|
| [REGISTER_CUSTOM_KERNEL](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-custom-kernel) | 注册Custom算子注册。|
| [RegisterKernelInterface](#registerkernelinterface) | 算子扩展能力注册实现类。|
| [KernelInterfaceReg](#kernelinterfacereg) | 算子扩展能力注册构造类。|
| [REGISTER_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-kernel-interface) | 注册算子扩展能力。|
| [REGISTER_CUSTOM_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-custom-kernel-interface) | 注册Custom算子扩展能力。|

## NodeParserRegistry

\#include <[node_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/node_parser_registry.h)>

NodeParserRegistry类用于注册及获取NodeParser类型的共享智能指针。

### NodeParserRegistry

```c++
NodeParserRegistry(converter::FmkType fmk_type, const std::string &node_type,
                   const converter::NodeParserPtr &node_parser);
```

构造函数。

- 参数

    - `fmk_type`: 框架类型，具体见[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#fmktype)说明。

    - `node_type`: 节点的类型。

    - `node_parser`: NodeParser类型的共享智能指针实例, 具体见[NodeParserPtr](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#nodeparserptr)说明。

### ~NodeParserRegistry

```c++
~NodeParserRegistry = default;
```

析构函数。

### 公有成员函数

#### GetNodeParser

```c++
static converter::NodeParserPtr GetNodeParser(converter::FmkType fmk_type, const std::string &node_type);
```

静态方法，获取NodeParser类型的共享智能指针实例。

- 参数

    - `fmk_type`: 框架类型，具体见[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#fmktype)说明。

    - `node_type`: 节点的类型。

## REG_NODE_PARSER

\#include <[node_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/node_parser_registry.h)>

```c++
#define REG_NODE_PARSER(fmk_type, node_type, node_parser)
```

注册NodeParser宏。

- 参数

    - `fmk_type`: 框架类型，具体见[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#fmktype)说明。

    - `node_type`: 节点的类型。

    - `node_parser`: NodeParser类型的共享智能指针实例, 具体见[NodeParserPtr](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#nodeparserptr)说明。

## ModelParserCreator

\#include <[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/model_parser_registry.h)>

```c++
typedef converter::ModelParser *(*ModelParserCreator)()
```

创建[ModelParser](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#modelparser)的函数原型声明。

## ModelParserRegistry

\#include <[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/model_parser_registry.h)>

ModelParserRegistry类用于注册及获取ModelParserCreator类型的函数指针。

### ModelParserRegistry

```c++
ModelParserRegistry(FmkType fmk, ModelParserCreator creator)
```

构造函数，构造ModelParserRegistry对象，进行Model解析注册。

- 参数

    - `fmk`: 框架类型，具体见[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#fmktype)说明。

    - `creator`: ModelParserCreator类型的函数指针, 具体见[ModelParserCreator](#modelparsercreator)说明。

### ~ModelParserRegistry

```c++
~ModelParserRegistry()
```

析构函数。

### 公有成员函数

#### GetModelParser

```c++
static ModelParser *GetModelParser(FmkType fmk)
```

获取ModelParserCreator类型的函数指针。

- 参数

    - `fmk`: 框架类型，具体见[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#fmktype)说明。

## REG_MODEL_PARSER

\#include <[model_parser_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/model_parser_registry.h)>

```c++
#define REG_MODEL_PARSER(fmk, parserCreator)
```

注册ModelParserCreator类。

- 参数

    - `fmk`: 框架类型，具体见[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#fmktype)说明。

    - `creator`: ModelParserCreator类型的函数指针, 具体见[ModelParserCreator](#modelparsercreator)说明。

> 用户自定义的ModelParser，框架类型必须满足设定支持的框架类型[FmkType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#fmktype)。

## PassBase

\#include <[pass_base.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/pass_base.h)>

PassBase定义了图优化的基类，以供用户继承并自定义图优化算法。

### PassBase

```c++
PassBase(const std::string &name = "PassBase")
```

构造函数，构造PassBase类对象。

- 参数

    - `name`: PassBase类对象的标识，需保证唯一性。

### ~PassBase

```c++
virtual ~PassBase() = default;
```

析构函数。

### 公有成员函数

#### Execute

```c++
virtual bool Execute(const api::FuncGraphPtr &func_graph) = 0;
```

对图进行操作的接口函数。

- 参数

    - `func_graph`: FuncGraph的指针类对象。

## PassBasePtr

\#include <[pass_base.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/pass_base.h)>

PassBase类的共享智能指针类型。

```c++
using PassBasePtr = std::shared_ptr<PassBase>
```

## PassPosition

\#include <[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/pass_registry.h)>

**enum**类型变量，定义扩展Pass的运行位置。

```c++
enum PassPosition {
  POSITION_BEGIN = 0,    // 扩展Pass运行于内置融合Pass前
  POSITION_END = 1       // 扩展Pass运行于内置融合Pass后
};
```

## PassRegistry

\#include <[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/pass_registry.h)>

PassRegistry类用于注册及获取Pass类实例。

### PassRegistry

```c++
PassRegistry(const std::string &pass_name, const PassBasePtr &pass)
```

构造函数，构造PassRegistry对象，进行注册Pass。

- 参数

    - `pass_name`: Pass的命名标识，保证唯一性。

    - `pass`: PassBase类实例。

```c++
PassRegistry(PassPosition position, const std::vector<std::string> &names)
```

构造函数，构造PassRegistry对象，指定扩展Pass的运行位置及其运行顺序。

- 参数

    - `position`: 扩展Pass的运行位置，具体见[PassPosition](#passposition)说明。

    - `names`: 用户指定在该运行位置处，调用Pass的命名标识，命名标识的顺序即为指定Pass的调用顺序。

### ~PassRegistry

```c++
~PassRegistry()
```

析构函数。

### 公有成员函数

#### GetOuterScheduleTask

```c++
static std::vector<std::string> GetOuterScheduleTask(PassPosition position)
```

获取指定位置处，外部设定的调度任务。

- 参数

    - `position`: 扩展Pass的运行位置，具体见[PassPosition](#passposition)说明。

#### GetPassFromStoreRoom

```c++
static PassBasePtr GetPassFromStoreRoom(const std::string &pass_name)
```

获取PassBase实例，根据指定的Pass命名标识。

- 参数

    - `pass_name`: Pass的命名标识。

## REG_PASS

\#include <[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/pass_registry.h)>

```c++
#define REG_PASS(name, pass)
```

注册Pass宏。

- 参数

    - `name`: Pass的命名标识，保证唯一性。

    - `pass`: PassBase类实例。

## REG_SCHEDULED_PASS

\#include <[pass_registry.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/pass_registry.h)>

```c++
#define REG_SCHEDULED_PASS(position, names)
```

指定扩展Pass的运行位置及其运行顺序。

- 参数

    - `position`: 扩展Pass的运行位置，具体见[PassPosition](#passposition)说明。

    - `names`: 用户指定在该运行位置处，调用Pass的命名标识，命名标识的顺序即为指定Pass的调用顺序。

> MindSpore Lite开放了部分内置Pass，请见以下说明。用户可以在`names`参数中添加内置Pass的命名标识，以在指定运行处调用内置Pass。
>
> - `ConstFoldPass`: 将输入均是常量的节点进行离线计算，导出的模型将不含该节点。特别地，针对shape算子，在[inputShape](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html#参数说明)给定的情形下，也会触发预计算。
> - `DumpGraph`: 导出当前状态下的模型。请确保当前模型为NHWC或者NCHW格式的模型，例如卷积算子等。
> - `ToNCHWFormat`: 将当前状态下的模型转换为NCHW的格式，例如，四维的图输入、卷积算子等。
> - `ToNHWCFormat`: 将当前状态下的模型转换为NHWC的格式，例如，四维的图输入、卷积算子等。
> - `DecreaseTransposeAlgo`: transpose算子的优化算法，删除冗余的transpose算子。
>
> `ToNCHWFormat`与`ToNHWCFormat`需配套使用。在开放的运行位置处，用户所得到的模型已统一为NHWC的格式，用户也需确保在当前运行位置处返回之时，模型也是NHWC的格式。
>
> 例: 指定names为{"ToNCHWFormat"， "UserPass"，"ToNHWCFormat"}。

## KernelDesc

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/register_kernel.h)>

**struct**类型结构体，定义扩展kernel的基本属性。

```c++
struct KernelDesc {
  DataType data_type;        // kernel的计算数据类型
  int type;                  // 算子的类型
  std::string arch;          // 设备标识
  std::string provider;      // 用户标识
};
```

## RegisterKernel

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/register_kernel.h)>

### CreateKernel

``` c++
using CreateKernel = std::function<std::shared_ptr<kernel::Kernel>(
  const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs, const schema::Primitive *primitive,
  const mindspore::Context *ctx)>
```

创建算子的函数原型声明。

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经由flatbuffers反序化为Primitive后的结果。

    - `ctx`: 算子的上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#context)。

### 公有成员函数

#### RegKernel

``` c++
static Status RegKernel(const std::string &arch, const std::string &provider, DataType data_type, int type, const CreateKernel creator)
```

算子注册。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 生产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，具体见[DataType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_datatype.html)。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#createkernel)的说明。

#### RegCustomKernel

``` c++
static Status RegCustomKernel(const std::string &arch, const std::string &provider, DataType data_type, const std::string &type, const CreateKernel creator)
```

Custom算子注册。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 生产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，具体见[DataType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_datatype.html)。

    - `type`: 算子类型，由用户自定义，确保唯一即可。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#createkernel)的说明。

#### GetCreator

```c++
static CreateKernel GetCreator(const schema::Primitive *primitive, KernelDesc *desc);
```

获取算子的创建函数。

- 参数

    - `primitive`: 算子经由flatbuffers反序化为Primitive后的结果。

    - `desc`: 算子的基本属性,具体见[KernelDesc](#kerneldesc)说明。

## KernelReg

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/register_kernel.h)>

### ~KernelReg

``` c++
~KernelReg() = default
```

析构函数。

### KernelReg

``` c++
KernelReg(const std::string &arch, const std::string &provider, DataType data_type, int op_type, const CreateKernel creator)
```

构造函数，构造注册算子，进行算子注册。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 生产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，具体见[DataType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_datatype.html)。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#createkernel)的说明。

``` c++
KernelReg(const std::string &arch, const std::string &provider, DataType data_type, const std::string &op_type, const CreateKernel creator)
```

构造函数，构造注册Custom算子，进行算子注册。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 生产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，具体见[DataType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_datatype.html)。

    - `op_type`: 算子类型，由用户自定义，确保唯一即可。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#createkernel)的说明。

## REGISTER_KERNEL

``` c++
#define REGISTER_KERNEL(arch, provider, data_type, op_type, creator)
```

注册算子宏。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 生产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，具体见[DataType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_datatype.html)。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#createkernel)的说明。

## REGISTER_CUSTOM_KERNEL

``` c++
#define REGISTER_CUSTOM_KERNEL(arch, provider, data_type, op_type, creator)
```

注册Custom算子。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 生产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，具体见[DataType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_datatype.html)。

    - `op_type`: 算子类型，由用户自定义，确保唯一即可。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#createkernel)的说明。

## KernelInterfaceCreator

\#include <[registry/register_kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/register_kernel_interface.h)>

定义创建算子的函数指针类型。

```c++
using KernelInterfaceCreator = std::function<std::shared_ptr<kernel::KernelInterface>()>;
```

## RegisterKernelInterface

\#include <[registry/register_kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/register_kernel_interface.h)>

算子扩展能力注册实现类。

### 公有成员函数

#### CustomReg

``` c++
static Status CustomReg(const std::string &provider, const std::string &op_type, const KernelInterfaceCreator creator)
```

Custom算子的扩展能力注册。

- 参数

    - `provider`: 生产商，由用户自定义。

    - `op_type`: 算子类型，由用户自定义。

    - `creator`: KernelInterface的创建函数，详细见[KernelInterfaceCreator](#kernelinterfacecreator)的说明。

#### Reg

``` c++
static Status Reg(const std::string &provider, int op_type, const KernelInterfaceCreator creator)
```

算子的扩展能力注册。

- 参数

    - `provider`: 生产商，由用户自定义。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: KernelInterface的创建函数，详细见[KernelInterfaceCreator](#kernelinterfacecreator)的说明。

#### GetKernelInterface

``` c++
static std::shared_ptr<kernel::KernelInterface> GetKernelInterface(const std::string &provider, const schema::Primitive *primitive, const kernel::Kernel *kernel)
```

获取注册的算子扩展能力。

- 参数

    - `provider`：生产商名，由用户自定义。

    - `primitive`：算子经过flatbuffers反序化后的结果，存储算子属性。

    - `kernel`：算子的内核，不传的话默认为空，为空时必须保证primitive非空有效。

## KernelInterfaceReg

\#include <[registry/register_kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/register_kernel_interface.h)>

算子扩展能力注册构造类。

### KernelInterfaceReg

``` c++
KernelInterfaceReg(const std::string &provider, int op_type, const KernelInterfaceCreator creator)
```

构造函数，构造注册算子的扩展能力。

- 参数

    - `provider`: 生产商，由用户自定义。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: KernelInterface的创建函数，详细见[KernelInterfaceCreator](#kernelinterfacecreator)的说明。

``` c++
KernelInterfaceReg(const std::string &provider, const std::string &op_type, const KernelInterfaceCreator creator)
```

构造函数，构造注册custom算子的扩展能力。

- 参数

    - `provider`: 生产商，由用户自定义。

    - `op_type`: 算子类型，由用户自定义。

    - `creator`: KernelInterface的创建函数，详细见[KernelInterfaceCreator](#kernelinterfacecreator)的说明。

## REGISTER_KERNEL_INTERFACE

\#include <[registry/register_kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/register_kernel_interface.h)>

注册KernelInterface的实现。

``` c++
#define REGISTER_KERNEL_INTERFACE(provider, op_type, creator)
```

- 参数

    - `provider`: 生产商，由用户自定义。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建KernelInterface的函数指针，具体见[KernelInterfaceCreator](#kernelinterfacecreator)的说明。

## REGISTER_CUSTOM_KERNEL_INTERFACE

\#include <[registry/register_kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/register_kernel_interface.h)>

注册Custom算子对应的KernelInterface实现。

``` c++
#define REGISTER_CUSTOM_KERNEL_INTERFACE(provider, op_type, creator)
```

- 参数

    - `provider`: 生产商名，由用户自定义。

    - `op_type`: 算子类型，由用户自定义，确保唯一同时要与REGISTER_CUSTOM_KERNEL时注册的op_type保持一致。

    - `creator`: 创建算子的函数指针，具体见[KernelInterfaceCreator](#kernelinterfacecreator)的说明。

