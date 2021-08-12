# mindspore::kernel

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_cpp/mindspore_kernel.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 接口汇总

| 类名 | 描述 |
| --- | --- |
| [Kernel](#Kernel) | 算子基类。|
| [RegisterKernel](#RegisterKernel) | 算子注册实现类。|
| [KernelReg](#KernelReg) | 算子注册构造类。|
| [REGISTER_KERNEL](#REGISTER_KERNEL) | 注册算子。|
| [REGISTER_CUSTOM_KERNEL](#REGISTER_CUSTOM_KERNEL) | 注册Custom算子注册。|
| [KernelInterfacel](#KernelInterface) | 算子扩展能力基类。|
| [RegisterKernelInterface](#RegisterKernelInterface) | 算子扩展能力注册实现类。|
| [KernelInterfaceReg](#KernelInterfaceReg) | 算子扩展能力注册构造类。|
| [REGISTER_KERNEL_INTERFACE](#REGISTER_KERNEL_INTERFACE) | 注册算子扩展能力。|
| [REGISTER_CUSTOM_KERNEL_INTERFACE](#REGISTER_CUSTOM_KERNEL_INTERFACE) | 注册Custom算子扩展能力。|

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

## RegisterKernel

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/register_kernel.h)>

### CreateKernel

``` c++
using CreateKernel = std::function<std::shared_ptr<kernel::Kernel>(
  const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs, const schema::Primitive *primitive,
  const mindspore::Context *ctx)>
```

创建算子的函数原型声明。

- 参数

    - `inputs`: 算子输入[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `outputs`: 算子输出[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#mstensor)。

    - `primitive`: 算子经由flatbuffers反序化为Primitive后的结果。

    - `ctx`: 算子的上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#context)。

### 公有成员函数

#### RegKernel

``` c++
static int RegKernel(const std::string &arch, const std::string &provider, TypeId data_type, int type, CreateKernel creator)
```

算子注册。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#CreateKernel)的说明。

#### RegCustomKernel

``` c++
static int RegCustomKernel(const std::string &arch, const std::string &provider, TypeId data_type, const std::string &op_type, CreateKernel creator)
```

Custom算子注册。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中。

    - `op_type`: 算子类型，由用户自定义，确保唯一即可。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#CreateKernel)的说明。

## KernelReg

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/register_kernel.h)>

### ~KernelReg

``` c++
~KernelReg() = default
```

析构函数。

### KernelReg

``` c++
KernelReg(const std::string &arch, const std::string &provider, TypeId data_type, int op_type, CreateKernel creator)
```

构造函数，构造注册算子，进行算子注册。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#CreateKernel)的说明。

``` c++
KernelReg(const std::string &arch, const std::string &provider, TypeId data_type, const std::string &op_type, CreateKernel creator)
```

构造函数，构造注册Custom算子，进行算子注册。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中。

    - `op_type`: 算子类型，由用户自定义，确保唯一即可。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#CreateKernel)的说明。

## REGISTER_KERNEL

``` c++
#define REGISTER_KERNEL(arch, provider, data_type, op_type, creator)
```

注册算子宏。

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#CreateKernel)的说明。

## REGISTER_CUSTOM_KERNEL

``` c++
#define REGISTER_CUSTOM_KERNEL(arch, provider, data_type, op_type, creator)
```

注册Custom算子

- 参数

    - `arch`: 算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。

    - `provider`: 产商名，由用户自定义。

    - `data_type`: 算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中。

    - `op_type`: 算子类型，由用户自定义，确保唯一即可。

    - `creator`: 创建算子的函数指针，具体见[CreateKernel](#CreateKernel)的说明。

## KernelInterface

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

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

创建[KernelInterface](#KernelInterface)的函数原型声明。

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

## RegisterKernelInterface

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

算子扩展能力注册实现类。

### 公有成员函数

#### CustomReg

``` c++
static int CustomReg(const std::string &provider, const std::string &op_type, KernelInterfaceCreator creator)
```

Custom算子的扩展能力注册。

- 参数

    - `provider`: 产商，由用户自定义。

    - `op_type`: 算子类型，由用户自定义。

    - `creator`: KernelInterface的创建函数，详细见[KernelInterfaceCreator](#KernelInterfaceCreator)的说明。

#### Reg

``` c++
static int Reg(const std::string &provider, int op_type, KernelInterfaceCreator creator)
```

算子的扩展能力注册。

- 参数

    - `provider`: 产商，由用户自定义。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: KernelInterface的创建函数，详细见[KernelInterfaceCreator](#KernelInterfaceCreator)的说明。

#### GetKernelInterface

``` c++
static std::shared_ptr<kernel::KernelInterface> GetKernelInterface(const std::string &provider, const schema::Primitive *primitive)
```

获取注册的算子扩展能力。

- 参数

    - `provider`：产商名，由用户自定义。

    - `primitive`：算子经过flatbuffers反序化后的结果，存储算子属性。

## KernelInterfaceReg

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

算子扩展能力注册构造类。

### KernelInterfaceReg

``` c++
KernelInterfaceReg(const std::string &provider, int op_type, KernelInterfaceCreator creator)
```

构造函数，构造注册算子的扩展能力

- 参数

    - `provider`: 产商，由用户自定义。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: KernelInterface的创建函数，详细见[KernelInterfaceCreator](#KernelInterfaceCreator)的说明。

``` c++
KernelInterfaceReg(const std::string &provider, const std::string &op_type, KernelInterfaceCreator creator)
```

构造函数，构造注册custom算子的扩展能力

- 参数

    - `provider`: 产商，由用户自定义。

    - `op_type`: 算子类型，由用户自定义。

    - `creator`: KernelInterface的创建函数，详细见[KernelInterfaceCreator](#KernelInterfaceCreator)的说明。

## REGISTER_KERNEL_INTERFACE

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

注册KernelInterface的实现。

``` c++
#define REGISTER_KERNEL_INTERFACE(provider, op_type, creator)
```

- 参数

    - `provider`: 产商，由用户自定义。

    - `op_type`: 算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。

    - `creator`: 创建KernelInterface的函数指针，具体见[KernelInterfaceCreator](#KernelInterfaceCreator)的说明。

## REGISTER_CUSTOM_KERNEL_INTERFACE

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

注册Custom算子对应的KernelInterface实现。

``` c++
#define REGISTER_CUSTOM_KERNEL_INTERFACE(provider, op_type, creator)
```

- 参数

    - `provider`: 产商名，由用户自定义。

    - `op_type`: 算子类型，由用户自定义，确保唯一同时要与REGISTER_CUSTOM_KERNEL时注册的op_type保持一致。

    - `creator`: 创建算子的函数指针，具体见[KernelInterfaceCreator](#KernelInterfaceCreator)的说明。
