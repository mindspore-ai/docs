# Using Delegate to Support Third-party AI Framework (Device)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/advanced/third_party/delegate.md)

## Overview

Delegate of MindSpore Lite is used to support third-party AI frameworks (such as NPU, TensorRT) to quickly access to the inference process in MindSpore Lite. Third-party frameworks can be implemented by users themselves, or other open source frameworks. Generally, the framework has the ability to build model online, that is, multiple operators can be built into a sub-graph and distributed to the device for inference. If the user wants to schedule other inference frameworks through MindSpore Lite, please refer to this article.

## Usage of Delegate

Using Delegate to support a third-party AI framework mainly includes the following steps:

1. Add a custom delegate class: Inherit the [Delegate](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Delegate.html) class to implement XXXDelegate.
2. Implementing the Init Function: The [Init](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Delegate.html) function needs to check whether the device supports the delegate framework and to apply for resources related to delegate.
3. Implementing the Build Function: The [Build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Delegate.html) function will implement the kernel support judgment, the sub-graph construction, and the online graph building.
4. Implementing the sub-graph Kernel: Inherit the [Kernel](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_kernel_Kernel.html#class-kernel) to implement delegate sub-graph Kernel.

### Adding a Custom Delegate Class

XXXDelegate should inherit from [Delegate](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Delegate.html). In the constructor of XXXDelegate, configure settings for third-party AI framework to build and execute the model, such as NPU frequency, CPU thread number, etc.

```cpp
class XXXDelegate : public Delegate {
 public:
  XXXDelegate() = default;

  ~XXXDelegate() = default;

  Status Init() = 0;

  Status Build(DelegateModel *model) = 0;
}
```

### Implementing the Init

[Init](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Delegate.html) will be called during the [Build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) process of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html#class-model). The specific location is in the [LiteSession::Init](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/lite_session.cc#L696) function of MindSpore Lite internal process.

```cpp
Status XXXDelegate::Init() {
  // 1. Check whether the inference device matches the delegate framework.
  // 2. Initialize delegate related resources.
}
```

### Implementing the Build

The input parameter of the [Build(DelegateModel *model)](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Delegate.html) interface is [DelegateModel](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_DelegateModel.html#template-class-delegatemodel).

> [std::vector<kernel::Kernel *> *kernels_](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_kernel_Kernel.html): A list of kernels that have been selected by MindSpore Lite and topologically sorted.
>
> [const std::map<kernel::Kernel *, const schema::Primitive *> primitives_](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_DelegateModel.html): A map of kernel and its attribute `schema::Primitive`, which is used to analyze the original attribute information.

[Build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Delegate.html) will be called during the [Build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) process of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html#class-model). The specific location is in the [Schedule::Schedule](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/scheduler.cc#L132) function of MindSpore Lite internal process. At this time, the inner kernels have been selected by MindSpore Lite. The following steps should be implemented in Build function:

1. Traverse the kernel list, use [GetPrimitive](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_DelegateModel.html) to get the attribute of kernel. Analyze the attribute to judge whether the delegate framework supports it.
2. For a continuous supported kernel list, construct a delegate sub-graph kernel and [Replace](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_DelegateModel.html) the continuous supported kernels with it.

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

### Implementing the Sub-graph Kernel

The `CreateXXXGraph` interface above will return a sub-graph kernel of XXXDelegate. The code is as follows:

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

The delegate sub-graph kernel `XXXGraph` should inherit from [Kernel](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_kernel_Kernel.html#class-kernel). The realization of `XXXGraph` should focus on:

1. Find the correct in_tensors and out_tensors for `XXXGraph` according to the original kernels list.
2. Rewrite the Prepare, Resize, and Execute interfaces. [Prepare](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#prepare) will be called in [Build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) of [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html#class-model). [Execute](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#execute) will be called in [Predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) of Model. [ReSize](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#resize) will be called in [Resize](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) of Model.

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

## Calling Delegate by Lite Framework

MindSpore Lite schedules user-defined delegate by [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Context.html#class-context). Use [SetDelegate](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#setdelegate) to set a custom delegate for Context.  Delegate will be passed by [Build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) to MindSpore Lite. If the Delegate in the Context is a null pointer, the process will call the inner inference of MindSpore Lite.

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

## Example of NPUDelegate

Currently, MindSpore Lite uses the [NPUDelegate](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_delegate.h#L29) for the NPU backend. This tutorial gives a brief description of NPUDelegate, so that users can quickly understand the usage of Delegate APIs.

### Adding the NPUDelegate Class

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

### Implementing the Init of NPUDelegate

[Init](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_delegate.cc#L75) function is used to apply resource for NPU and determine whether the hardware supports NPU.

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

### Implementing the Build of NPUDelegate

The [Build](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_delegate.cc#L163) interface parses the DelegateModel and mainly implements the kernel support judgment, the sub-graph construction, and the online graph building.

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

### Creating NPUGraph

The following [Sample Code](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_delegate.cc#L273) is the CreateNPUGraph interface of NPUDelegate, used to generate an NPU sub-graph kernel.

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

### Adding the NPUGraph Class

[NPUGraph](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_graph.h#L29) inherits from [Kernel](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_kernel_Kernel.html#class-kernel). And we need to rewrite the Prepare, Execute, and ReSize interfaces.

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

[NPUGraph::Prepare](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_graph.cc#L306) mainly implements:

```cpp
int NPUGraph::Prepare() {
  // Find the mapping relationship between hiai::AiTensor defined by NPU and MSTensor defined by MindSpore Lite
}
```

[NPUGraph::Execute](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/src/litert/delegate/npu/npu_graph.cc#L322) mainly implements:

```cpp
int NPUGraph::Execute() {
  // 1. Processing input: copy input data from MSTensor to hiai::AiTensor
  // 2. Perform inference
  executor_->Execute();
  // 3. Processing output: copy output data from hiai::AiTensor to MSTensor
}
```

> [NPU](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/npu_info.html) is a third-party AI framework that added by MindSpore Lite internal developers. The usage of NPU is slightly different. You can set the [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Context.html#class-context) through [SetDelegate](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#setdelegate), or you can add the description of the NPU device [KirinNPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_KirinNPUDeviceInfo.html#class-kirinnpudeviceinfo) to [MutableDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Context.html) of the Context.
