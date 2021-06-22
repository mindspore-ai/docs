# Registry

## CreateKernel

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/register_kernel.h)>

创建Kernel的函数原型

``` c++
using CreateKernel = std::function<std::shared_ptr<kernel::Kernel>(
  const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
  const schema::Primitive *primitive, const lite::Context *ctx)>;
```

* [tensor::MSTensor](https://mindspore.cn/doc/api_cpp/zh-CN/master/tensor.html)。
* schema::Primitive：算子经过flatbuffer反序化后的结果。
* [lite::Context](https://mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#Context)。

## REGISTER_KERNEL

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/register_kernel.h)>

注册算子。

``` c++
#define REGISTER_KERNEL(arch, provider, data_type, op_type, creator)
```

* arch：算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。
* provider：产商名，由用户自定义。
* data_type：算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中。
* op_type：算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。
* creator：创建算子的函数指针，具体见CreateKernel的说明。

## REGISTER_CUSTOM_KERNEL

\#include <[registry/register_kernel.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/register_kernel.h)>

注册Custom算子

``` c++
#define REGISTER_CUSTOM_KERNEL(arch, provider, data_type, op_type, creator)
```

* arch：算子运行的平台，由用户自定义，如果算子是运行在CPU平台，或者算子运行完后的output tensor里的内存是在CPU平台上的，则此处也写CPU，MindSpore Lite内部会切成一个子图，在异构并行场景下有助于性能提升。
* provider：产商名，由用户自定义。
* data_type：算子支持的数据类型，定义在[type_id.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/core/ir/dtype/type_id.h)中。
* op_type：算子类型，由用户自定义，确保唯一即可。
* creator：创建算子的函数指针，具体见CreateKernel的说明。

## KernelInterface

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

算子的统一接口，未来算子能力没有在[Kernel](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/kernel.h)中有承载的话都会在这里进行添加。

### 公有成员函数

#### Infer

算子的InferShape能力，用于根据输入推导出输出的shape、数据类型以及format。

``` c++
virtual int Infer(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
                    const schema::Primitive *primitive)
```

* inputs：输入。
* outputs：输出。
* primitive：算子经过flatbuffer反序化后的结果。

## KernelInterfaceCreator

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

创建KernelInterface的函数原型

``` c++
using KernelInterfaceCreator = std::function<std::shared_ptr<KernelInterface>()>;
```

## REGISTER_KERNEL_INTERFACE

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

注册KernelInterface的实现。

``` c++
#define REGISTER_KERNEL_INTERFACE(provider, op_type, creator)
```

* provider：产商，由用户自定义。
* op_type：算子类型，定义在[ops.fbs](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/schema/ops.fbs)中，编绎时会生成到ops_generated.h，该文件可以在发布件中获取。
* creator：创建KernelInterface的函数指针，具体见KernelInterfaceCreator的说明。

## REGISTER_CUSTOM_KERNEL_INTERFACE

\#include <[registry/kernel_interface.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/kernel_interface.h)>

注册Custom算子对应的KernelInterface实现。

``` c++
#define REGISTER_CUSTOM_KERNEL_INTERFACE(provider, op_type, creator)
```

* provider：产商名，由用户自定义。
* op_type：算子类型，由用户自定义，确保唯一同时要与REGISTER_CUSTOM_KERNEL时注册的op_type保持一致。
* creator：创建算子的函数指针，具体见KernelInterfaceCreator的说明。
