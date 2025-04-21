# 使用Delegate支持第三方AI框架接入（端上）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/advanced/third_party/delegate.md)

## 概述

MindSpore Lite的Delegate接口用于支持第三方AI框架（例如：NPU、TensorRT）能快速接入Lite的推理流程。第三方框架可以是用户自己实现，也可以是业内其他开源的框架，一般都具备在线构图的能力，即可以将多个算子构建成一张子图发放给设备执行。如果用户想通过MindSpore Lite框架调度到其他框架的推理流程，可参考本文。

## Delegate使用

使用Delegate接入第三方AI框架执行推理主要包含以下步骤：

1. 新增自定义Delegate类：继承[Delegate](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#delegate)类实现自定义的Delegate。
2. 实现初始化接口：[Init](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#init)接口实现判断运行设备是否支持Delegate框架，初始化Delegate资源等功能。
3. 实现构图接口：[Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build)接口要实现算子支持判断、子图构建、在线构图功能。
4. 实现子图Kernel：继承[Kernel](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#kernel)实现Delegate的子图Kernel。

### 新增自定义Delegate类

自定义Delegate要继承自[Delegate](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#delegate)类。可以在构造函数中完成对第三方框架调度硬件设备有关config的初始化，如NPU指定频率、CPU指定线程数等。

```cpp
class XXXDelegate : public Delegate {
 public:
  XXXDelegate() = default;

  ~XXXDelegate() = default;

  Status Init() = 0;

  Status Build(DelegateModel *model) = 0;
}
```

### 实现初始化接口

[Init](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#init)接口会在[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#model)的[Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build)流程中被调用。具体的调用位置在MindSpore Lite内部代码[LiteSession::Init](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/lite_session.cc#L696)函数中。

```cpp
Status XXXDelegate::Init() {
  // 1. Check whether the inference device matches the delegate framework.
  // 2. Initialize delegate related resources.
}
```

### 实现构图接口

构图接口[Build(DelegateModel *model)](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build)接口的入参是[DelegateModel](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#delegatemodel)的实例。

> `DelegateModel`中，[std::vector<kernel::Kernel *> *kernels_](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#kernel)是已经完成MindSpore Lite内置算子注册、经过拓扑排序的算子列表。
>
> [const std::map<kernel::Kernel *, const schema::Primitive *> primitives_](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#primitives-)保存了每个算子对应的属性值`schema::Primitive`，用于解析每个算子的原始属性信息。

Build会在[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#model)的[Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build)接口被调用。具体的位置在MindSpore Lite内部代码[Schedule::Schedule](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/scheduler.cc#L132)函数中，此时已完成内置算子选择，算子存放在DelegateModel的[Kernel列表](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#kernel)中。Build需要实现以下功能：

1. 遍历Kernel列表，调用[GetPrimitive](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#getprimitive)获取每个算子对应的属性值，解析该算子的属性值，判断Delegate框架是否支持。
2. 对连续可支持的一段算子列表，构建一张Delegate子图，调用[Replace](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#replace)用子图Kernel去替换这段连续的算子。

```cpp
Status XXXDelegate::Build(DelegateModel *model) {
  KernelIter from = model->BeginKernelIterator();                   // Record the start operator position supported by the Delegate
  KernelIter end = model->BeginKernelIterator();                    // Record the end operator position supported by the Delegate
  for (KernelIter iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    kernel::Kernel *kernel = *iter;
    if (IsSupport(kernel, model->GetPrimitive(kernel))) {           // Check whether the Delegate framework supports the kernel according to the primitive
      end = iter;
    } else {                                                        // The current kernel is not supported, and the sub-graph is truncated
      if (from != end) {
        auto xxx_graph_kernel = CreateXXXGraph(from, end, model);   // Create a Delegate sub-graph Kernel
        iter = model->Replace(from, end + 1, xxx_graph_kernel);     // Replace the supported kernels list with a Delegate sub-graph Kernel
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
  auto in_tensors = GraphInTensors(...);    // Find the input tensors of the Delegate sub-graph
  auto out_tensors = GraphOutTensors(...);  // Find the output tensors of the Delegate sub-graph
  auto graph_kernel = new (std::nothrow) XXXGraph(in_tensors, out_tensors);
  if (graph_kernel == nullptr) {
    MS_LOG(ERROR) << "New XXX Graph failed.";
    return nullptr;
  }
  // Build graph online, load model, etc.
  return graph_kernel;
}
```

Delegate子图`XXXGraph`的定义要继承自[Kernel](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#kernel)，如下代码所示。对这张子图，要注意：

1. 要根据原始的Kernel列表找到正确的in_tensors和out_tensors，以便Execute时，能找到正确的输入tensor和输入数据，并将输出数据写回到正确的地址中。
2. 重写对应的Prepare、Resize、Execute接口。其中，[Prepare](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#prepare)会在Model的[Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build)阶段调用。[Execute](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#execute)会在Model的[Predict](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#predict)阶段被调用。[ReSize](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#resize)会在Model的[Resize](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#resize)阶段被调用。

```cpp
class XXXGraph : public kernel::Kernel {
 public:
  XXXGraph(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr) {}

  ~XXXGraph() override;

  int Prepare() override {
    // Generally, the model will be built only once, so Prepare is also called once.
    // Do something without input data, such as pack the constant weight tensor, etc.
  }

  int Execute() override {
    // Obtain input data from in_tensors.
    // Execute the inference process.
    // Write the result back to out_tensors.
  }

  int ReSize() override {
    // Support dynamic shape, and input shape will changed.
  }
};
```

## Lite框架调度

Lite框架要调度用户自定义的Delegate，在创建[Context](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#context)时，需要通过[SetDelegate](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#setdelegate)设置自定义Delegate指针，见以下示例代码。再通过[Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build)传递给Lite框架。如果Context中的Delegate为空指针，推理流程会调用到Lite框架内置的推理。

```cpp
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
  MS_LOG(ERROR) << "New context failed";
  return RET_ERROR;
}
auto delegate = std::make_shared<XXXDelegate>();
if (delegate == nullptr) {
  MS_LOG(ERROR) << "New XXX delegate failed";
  return RET_ERROR;
}
context->SetDelegate(delegate);

auto model = new (std::nothrow) mindspore::Model();
if (model == nullptr) {
  std::cerr << "New Model failed." << std::endl;
}
// Assuming that we have read a ms file and stored in the address pointed by model_buf
auto build_ret = model->Build(model_buf, size, mindspore::kMindIR, context);
delete[](model_buf);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model failed." << std::endl;
}
```

## NPUDelegate示例

目前，MindSpore Lite对于NPU后端的集成采用了[NPUDelegate](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_delegate.h#L29)接口。本教程对NPUDelegate做简单说明，使用户能快速了解Delegate相关API的使用。

### 新增NPUDelegate类

```cpp
class NPUDelegate : public Delegate {
 public:
  explicit NPUDelegate(lite::NpuDeviceInfo device_info) : Delegate() { frequency_ = device_info.frequency_; }

  ~NPUDelegate() override;

  Status Init() override;

  Status Build(DelegateModel *model) override;

 protected:
  // Analyze a kernel and its attribute.
  // If NPU supports it, return an NPUOp, which has the information of connection relationship with other kernels and the attributes.
  // If not support, return null pointer.
  NPUOp *GetOP(kernel::Kernel *kernel, const schema::Primitive *primitive);

  // Construct a NPU sub-graph with a continuous NPUOps
  kernel::Kernel *CreateNPUGraph(const std::vector<NPUOp *> &ops, DelegateModel *model, KernelIter from,
                                 KernelIter end);

  NPUManager *npu_manager_ = nullptr;
  NPUPassManager *pass_manager_ = nullptr;
  std::map<schema::PrimitiveType, NPUGetOp> op_func_lists_;
  int frequency_ = 0;  // NPU frequency
};
```

### 实现Init接口

[Init](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_delegate.cc#L75)接口实现和NPU有关的资源申请。

```cpp
Status NPUDelegate::Init() {
  npu_manager_ = new (std::nothrow) NPUManager();       // NPU manager of model buffer and client.
  if (npu_manager_ == nullptr) {
    MS_LOG(ERROR) << "New npu manager failed.";
    return RET_ERROR;
  }
  if (!npu_manager_->IsSupportNPU()) {                  // Check whether the current device supports NPU.
    MS_LOG(DEBUG) << "Checking npu is unsupported.";
    return RET_NOT_SUPPORT;
  }
  pass_manager_ = new (std::nothrow) NPUPassManager();  // The default format of MindSpore Lite is NHWC, and the default format of NPU is NCHW. The NPUPassManager is used to pack data between the sub-graphs.
  if (pass_manager_ == nullptr) {
    MS_LOG(ERROR) << "New npu pass manager failed.";
    return RET_ERROR;
  }

  // Initialize op_func lists. Get the correspondence between kernel type and GetOP function.
  op_func_lists_.clear();
  return RET_OK;
}
```

### 实现Build接口

Build接口解析DelegateModel实例，主要实现算子支持判断、子图构建、在线构图等功能。下面[示例代码](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_delegate.cc#L163)是NPUDelegate Build接口的实现。

```cpp
Status NPUDelegate::Build(DelegateModel *model) {
  KernelIter from, end;                     // Record the start and end positions of kernel supported by the NPU sub-graph.
  std::vector<NPUOp *> npu_ops;             // Save all NPUOp used to construct an NPU sub-graph.
  int graph_index = 0;
  for (KernelIter iter = model->BeginKernelIterator(); iter != model->EndKernelIterator(); iter++) {
    kernel::Kernel *kernel = *iter;
    auto npu_op = GetOP(kernel, model->GetPrimitive(kernel));  // Obtain an NPUOp according to the kernel and the primitive. Each NPUOp contains information such as input tensors, output tensors and operator attribute.
    if (npu_op != nullptr) {                // NPU supports the current kernel.
      if (npu_ops.size() == 0) {
        from = iter;
      }
      npu_ops.push_back(npu_op);
      end = iter;
    } else {                                 // NPU does not support the current kernel.
      if (npu_ops.size() > 0) {
        auto npu_graph_kernel = CreateNPUGraph(npu_ops);  // Create a NPU sub-graph kernel.
        if (npu_graph_kernel == nullptr) {
          MS_LOG(ERROR) << "Create NPU Graph failed.";
          return RET_ERROR;
        }
        npu_graph_kernel->set_name("NpuGraph" + std::to_string(graph_index++));
        iter = model->Replace(from, end + 1, npu_graph_kernel);  // Replace the supported kernel list with a NPU sub-graph kernel.
        npu_ops.clear();
      }
    }
  }
  auto ret = npu_manager_->LoadOMModel();    // Build model online. Load NPU model.
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "NPU client load model failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
```

### 实现构图代码

以下[示例代码](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_delegate.cc#L273)是NPUDelegate的CreateNPUGraph接口，用于生成一张NPU子图。

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

[NPUGraph](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_graph.h#L29)继承自[Kernel](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#kernel)，需要重写Prepare、Execute、ReSize接口。

```cpp
class NPUGraph : public kernel::Kernel {
 public:
  NPUGraph(std::vector<NPUOp *> npu_ops, NPUManager *npu_manager, const std::vector<tensor::MSTensor *> &inputs,
           const std::vector<tensor::MSTensor *> &outputs)
      : kernel::Kernel(inputs, outputs, nullptr, nullptr), npu_ops_(std::move(npu_ops)), npu_manager_(npu_manager) {}

  ~NPUGraph() override;

  int Prepare() override;

  int Execute() override;

  int ReSize() override {               // NPU does not support dynamic shapes.
    MS_LOG(ERROR) << "NPU does not support the resize function temporarily.";
    return lite::RET_ERROR;
  }

 protected:
  std::vector<NPUOp *> npu_ops_{};
  NPUManager *npu_manager_ = nullptr;  
  NPUExecutor *executor_ = nullptr;     // NPU inference executor.
};
```

[NPUGraph::Prepare](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_graph.cc#L306)接口主要实现:

```cpp
int NPUGraph::Prepare() {
  // Find the mapping relationship between hiai::AiTensor defined by NPU and MSTensor defined by MindSpore Lite
}
```

[NPUGraph::Execute](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_graph.cc#L322)接口主要实现:

```cpp
int NPUGraph::Execute() {
  // 1. Processing input: copy input data from MSTensor to hiai::AiTensor
  // 2. Perform inference
  executor_->Execute();
  // 3. Processing output: copy output data from hiai::AiTensor to MSTensor
}
```

> [NPU](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/advanced/third_party/npu_info.html)是MindSpore Lite开发人员对接的第三方AI框架，使用方法和用户自定义的Delegate略有不同，既可以通过[SetDelegate](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#setdelegate)设置[Context](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#context)，也可以设置Context的[MutableDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mutabledeviceinfo)，增加NPU设备的描述[KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#kirinnpudeviceinfo)。
