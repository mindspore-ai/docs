# 使用Delegate支持第三方AI框架接入

`Linux` `第三方接入` `自定义算子` `高级`

<!-- TOC -->

- [使用Delegate支持第三方AI框架接入](#使用Delegate支持第三方AI框架接入)
    - [概述](#概述)
    - [Delegate使用](#Delegate使用)
        - [新增自定义Delegate类](#新增自定义Delegate类)
        - [实现初始化接口](#实现初始化接口)
        - [实现构图接口](#实现构图接口)
            - [DelegateModel定义](#DelegateModel定义)
        - [实现子图Kernel](#实现子图Kernel)
    - [Lite框架调度](#Lite框架调度)
    - [NPUDelegate示例](#NPUDelegate示例)
        - [新增NPUDelegate类](#新增NPUDelegate类)
        - [实现Init接口](#实现Init接口)
        - [实现Build接口](#实现Build接口)
        - [实现构图代码](#实现构图代码)
        - [实现NPUGraph](#实现NPUGraph)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/docs/source_zh_cn/use/delegate.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

MindSpore Lite的Delegate接口用于支持第三方AI框架（例如：NPU、TensorRT）能快速接入Lite的推理流程。第三方框架可以是用户自己实现，也可以是业内其他开源的框架，一般都具备在线构图的能力，即可以将多个算子构建成一张子图发放给设备执行。如果用户想通过MindSpore Lite框架调度到其他框架的推理流程，可参考本文。

## Delegate使用

使用Delegate接入第三方AI框架执行推理主要包含以下步骤：

1. 新增自定义Delegate类：继承Delegate类实现自定义的Delegate。
2. 实现初始化接口：Init接口主要完后与Delegate相关的资源申请、config设置等功能，会在[CreateSession](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/src/lite_session.cc#L655)流程中被调用。
3. 实现构图接口：Build接口要实现算子支持判断、子图构建、在线构图等功能，会在[LiteSession::CompileGraph](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/src/scheduler.cc#L126)流程中被调用。
4. 实现子图Kernel：继承Kernel类实现Delegate子图Kernel。

### 新增自定义Delegate类

自定义Delegate要继承自[Delegate](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/include/delegate.h#L71)类。可以在构造函数中完成对第三方框架调度硬件设备有关config的必要初始化，如NPU指定频率、CPU制定线程数等。

```cpp
class XXXDelegate : public Delegate {
 public:
  XXXDelegate() = default;

  virtual ~XXXDelegate() = default;

  virtual int Init() = 0;

  virtual int Build(DelegateModel *model) = 0;
};
}
```

### 实现初始化接口

Init会在CreateSession中初始化Context时被调用。Init的返回码分为三类：RET_OK(正常推理)、RET_ERROR(错误，流程终止)、RET_NOT_SUPPORT(不支持，退回到Lite内置推理流程)

```cpp
int XXXDelegate::Init() {
  // 运行时判断推理设备与用户框架是否匹配，初始化delegate相关的资源等
}
```

### 实现构图接口

#### DelegateModel定义

`Build(DelegateModel *model)`接口的入参是[DelegateModel](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/include/delegate.h#L29)的实例。DelegateModel的定义如下：

```cpp
using KernelIter = std::vector<kernel::Kernel *>::iterator;
class DelegateModel {
 public:
  DelegateModel(std::vector<kernel::Kernel *> *kernels,
                const std::map<kernel::Kernel *, const schema::Primitive *> primitives)
      : kernels_(kernels), primitives_(primitives) {}

  ~DelegateModel() = default;

  // 获取算子对应的属性Primitive
  const schema::Primitive *GetPrimitive(kernel::Kernel *kernel) const;

  // 返回算子列表vector的起始迭代器
  KernelIter BeginKernelIterator();

  // 返回算子列表vector的末尾迭代器
  KernelIter EndKernelIterator();

  // 用delegate子图kernel去替换delegate所支持的一段连续的算子，即替换从[from, end)的算子，返回值为替换后子图kernel的下一个元素的迭代器，用于继续遍历
  KernelIter Replace(KernelIter from, KernelIter end, kernel::Kernel *graph_kernel);

 protected:
  std::vector<kernel::Kernel *> *kernels_;
  const std::map<kernel::Kernel *, const schema::Primitive *> primitives_;
};
```

> `std::vector<kernel::Kernel *> *kernels_`是已经完成内置算子注册、经过拓扑排序的算子列表。
>
> `const std::map<kernel::Kernel *, const schema::Primitive *> primitives_`保存了每个算子对应的属性值`schema::Primitive`，用于delegate解析每个算子的原始属性信息。

构图接口Build会在LiteSession::CompileGraph时，完成内部算子选择之后被调用，此时，内部算子存放在DelegateModel的Kernel列表中。Build需要实现以下功能：

1. 解析每个算子的属性值，判断Delegate框架是否支持。
2. 对连续可支持的一段算子列表，构建一张子图，用子图去替换这段算子。

```cpp
int XXXDelegate::Build(DelegateModel *model) {
  KernelIter from = model->BeginKernelIterator();  // 记录Delegate子图支持算子列表的起始位置
  KernelIter end = model->BeginKernelIterator();  // 记录Delegate子图支持算子列表的终止位置
  for (KernelIter iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    kernel::Kernel *kernel = *iter;
    if (IsSupport(kernel, model->GetPrimitive(kernel))) {  // 根据Kernel和对应的属性判断Delegate框架是否支持
      end = iter;
    } else {  // 当前算子不支持，子图被截断
      if (from != end) {
        auto xxx_graph_kernel = CreateXXXGraph(from, end, model);  // 创建一个Delegate子图Kernel
        iter = model->Replace(from, end + 1, xxx_graph_kernel);  // 调用Replace用Delegate子图Kernel去替换支持的Kernel列表
      }
      from = iter + 1;
      end = iter + 1;
    }
  }
  return RET_OK;
}
```

### 实现子图Kernel

上述`CreateXXXGraph`接口要返回一张Delegate的子图，示例代码如下所示:

```cpp
kernel::Kernel *XXXDelegate::CreateXXXGraph(KernelIter from, KernelIter end, DelegateModel *model) {
  auto in_tensors = GraphInTensors(...);  // 找到子图的input tensors
  auto out_tensors = GraphOutTensors(...);  // 找到子图的output tensors
  auto graph_kernel = new (std::nothrow) XXXGraph(in_tensors, out_tensors);
  if (graph_kernel == nullptr) {
    MS_LOG(ERROR) << "New XXX Graph failed.";
    return nullptr;
  }
  // 在线构图、初始化、模型加载等操作
  return graph_kernel;
}
```

Delegate子图`XXXGraph`的定义要继承自[Kernel](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/kernel.h#L27)，如下代码所示。对这张子图，要注意：

1. 要根据原始的Kernel列表找到正确的in_tensors和out_tensors：以便Execute时，能找到正确的输入tensor和输入数据，并将output写回到正确的地址中。
2. 重写对应的Prepare()、Resize()、Execute()接口。子图Kernel会返回给Lite框架，因此重写的接口会被Lite框架调用。

```cpp
class XXXGraph : public kernel::Kernel {
 public:
  XXXGraph(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr) {}

  ~XXXGraph() override;

  int Prepare() override {
    // LiteSession::CompileGraph中被调用，实现与输入数据无关的操作，如常量权重tensor的重排等
  }

  int Execute() override {
    // 从in_tensors中获取输入数据；执行推理流程；计算结束后把结果写回out_tensors
  }

  int ReSize() override {
    // 支持动态shape
  }
};
```

## Lite框架调度

Lite框架要调度用户自定义的Delegate，在创建[Context](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#context)时，用户应该设置相应的自定义Delegate指针，见以下示例代码。再通过CreateSession(Context)传递给Lite框架。LiteSession从Context中获取到自定义Delegate的指针，就可以调用它的Init、Build接口。如果Context中的delegate为空指针，推理流程会调用到Lite框架内置的推理。

```cpp
auto context = std::make_shared<Context>();
if (context == nullptr) {
  MS_LOG(ERROR) << "New context failed";
  return RET_ERROR;
}
auto delegate = std::make_shared<XXXDelegate>();;
if (delegate == nullptr) {
  MS_LOG(ERROR) << "New XXX delegate failed";
  return RET_ERROR;
}
context->delegate = delegate;
auto session = session::LiteSession::CreateSession(context.get());
if (session_ == nullptr) {
  MS_LOG(ERROR) << "CreateSession failed while running";
  return RET_ERROR;
}
```

## NPUDelegate示例

目前，MindSpore Lite对于NPU后端的集成采用了[NPUDelegate](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/src/delegate/npu/npu_delegate.h#L34)接口。本教程对NPUDelegate做简单说明，使用户能快速了解Delegate相关API的使用。

```cpp
class NPUDelegate : public Delegate {
 public:
  explicit NPUDelegate(lite::NpuDeviceInfo device_info) : Delegate() { frequency_ = device_info.frequency_; }

  ~NPUDelegate() override;

  int Init() override;

  int Build(DelegateModel *model) override;

 protected:
  // 解析Kernel和其属性值；NPU如果支持，返回一个NPUOp指针，NPUOp中保存了和其他算子的连接关系以及该算子的属性，如果不支持，返回一个空指针
  NPUOp *GetOP(kernel::Kernel *kernel, const schema::Primitive *primitive);

  // 用一段连续的NPUOp构建一张NPU子图
  kernel::Kernel *CreateNPUGraph(const std::vector<NPUOp *> &ops, DelegateModel *model, KernelIter from,
                                 KernelIter end);

  NPUManager *npu_manager_ = nullptr;
  NPUPassManager *pass_manager_ = nullptr;
  std::map<schema::PrimitiveType, NPUGetOp> op_func_lists_;
  int frequency_ = 0;  // NPU频率设置
};
```

### 实现Init接口

[Init](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/src/delegate/npu/npu_delegate.cc#L71)接口实现和NPU有关的资源申请。

```cpp
int NPUDelegate::Init() {
  npu_manager_ = new (std::nothrow) NPUManager();  // NPU管理器，管理NPU模型buffer、Client等
  if (npu_manager_ == nullptr) {
    MS_LOG(ERROR) << "New npu manager failed.";
    return RET_ERROR;
  }
  if (!npu_manager_->IsSupportNPU()) {  // 判断当前执行设备的系统版本是否支持NPU
    MS_LOG(DEBUG) << "Checking npu is unsupported.";
    return RET_NOT_SUPPORT;
  }
  pass_manager_ = new (std::nothrow) NPUPassManager();  // 由于NPU和Lite默认format不一致，需要在子图之间做数据重排，NPUPassManager用于实现数据重排
  if (pass_manager_ == nullptr) {
    MS_LOG(ERROR) << "New npu pass manager failed.";
    return RET_ERROR;
  }

  op_func_lists_.clear();
  // ...  op_func_lists_，初始化算子对接函数，op_func_lists_的key值为算子类型，value值为对应的GetNPUOp函数，在这个函数里，要完成判断NPU是否支持该算子、属性获取等功能。
  return RET_OK;
}
```

### 实现Build接口

Build接口解析DelegateModel实例，主要实现算子支持判断、子图构建、在线构图等功能。
下面[示例代码](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/src/delegate/npu/npu_delegate.cc#L157)是NPUDelegate Build接口的实现。

```cpp
int NPUDelegate::Build(DelegateModel *model) {
  KernelIter from, end;  // 记录NPU子图支持算子的起始和终止位置
  std::vector<NPUOp *> npu_ops;  // 保存用于构建一张NPU子图的所有的NPUOp
  int graph_index = 0;
  for (KernelIter iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    kernel::Kernel *kernel = *iter;
    auto npu_op = GetOP(kernel, model->GetPrimitive(kernel));  // 根据Kernel和对应的Primitive获取一个NPUOp，每个NPUOp包含输入输出MSTensor、算子属性等信息
    if (npu_op != nullptr) {  // NPU支持当前kernel
      if (npu_ops.size() == 0) {
        from = iter;
      }
      npu_ops.push_back(npu_op);
      end = iter;
    } else {  // NPU不支持当前kernel
      if (npu_ops.size() > 0) {
        auto npu_graph_kernel = CreateNPUGraph(npu_ops);  // 创建一个NPU子图Kernel
        if (npu_graph_kernel == nullptr) {
          MS_LOG(ERROR) << "Create NPU Graph failed.";
          return RET_ERROR;
        }
        npu_graph_kernel->set_name("NpuGraph" + std::to_string(graph_index++));
        iter = model->Replace(from, end + 1, npu_graph_kernel);  // 用NPU子图Kernel去替换支持的Kernel列表
        npu_ops.clear();
      }
    }
  }
  auto ret = npu_manager_->LoadOMModel();  // NPU在线构图，加载OMModel
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NPU client load model failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
```

### 实现构图代码

以下[示例代码](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/src/delegate/npu/npu_delegate.cc#L273)是NPUDelegate的CreateNPUGraph接口，用于生成一张NPU子图。

```cpp
kernel::Kernel *NPUDelegate::CreateNPUGraph(const std::vector<NPUOp *> &ops) {
  auto in_tensors = GraphInTensors(ops);
  auto out_tensors = GraphOutTensors(ops);
  auto graph_kernel = new (std::nothrow) NPUGraph(ops, npu_manager_, in_tensors, out_tensors);
  if (graph_kernel == nullptr) {
    MS_LOG(DEBUG) << "New NPU Graph failed.";
    return nullptr;
  }
  ret = graph_kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(DEBUG) << "NPU Graph Init failed.";
    return nullptr;
  }
  return graph_kernel;
}
```

### 实现NPUGraph

[NPUGraph](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/src/delegate/npu/npu_graph.h#L30)继承自[Kernel](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/kernel.h#L27)，需要重写Prepare、Execute、ReSize接口。

```cpp
class NPUGraph : public kernel::Kernel {
 public:
  NPUGraph(std::vector<NPUOp *> npu_ops, NPUManager *npu_manager, const std::vector<tensor::MSTensor *> &inputs,
           const std::vector<tensor::MSTensor *> &outputs)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr), npu_ops_(std::move(npu_ops)), npu_manager_(npu_manager) {}

  ~NPUGraph() override;

  int Prepare() override;

  int Execute() override;

  int ReSize() override {  // NPU不支持动态shape
    MS_LOG(ERROR) << "NPU does not support the resize function temporarily.";
    return lite::RET_ERROR;
  }

 protected:
  std::vector<NPUOp *> npu_ops_{};
  NPUManager *npu_manager_ = nullptr;  
  NPUExecutor *executor_ = nullptr;  // NPU推理执行器
};
```

[NPUGraph::Prepare](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/src/delegate/npu/npu_graph.cc#L193)接口主要实现:

```cpp
int NPUGraph::Prepare() {
  // 找到NPU定义的hiai::AiTensor和MSTensor的映射关系等
}
```

[NPUGraph::Execute](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/src/delegate/npu/npu_graph.cc#L209)接口主要实现:

```cpp
int NPUGraph::Execute() {
  // 1. input从MSTensor向hiai::AiTensor拷贝数据
  // 2. 执行推理
  executor_->Execute();
  // 3. output从hiai::AiTensor向MSTensor拷贝数据
  // 4...
}
```

> [NPU](https://mindspore.cn/tutorial/lite/zh-CN/master/use/npu_info.html)是MindSpore Lite开发人员对接的第三方AI框架，使用方法和用户自定义的Delegate略有不同，既可以设置Context的Delegate，也可以设置[DeviceContext](https://www.mindspore.cn/doc/api_cpp/zh-CN/master/lite.html#devicecontext)的device_type为DT_NPU。
