# Use Runtime for Model Inference 

<!-- TOC -->

- [Use Runtime for Model Inference](#use-runtime-for-model-inference)
  - [Overview](#overview)
  - [Reading Models](#reading-models)
  - [Session Creation](#session-creation)
    - [Creating Contexts](#creating-contexts)
    - [Creating Sessions](#creating-sessions)
    - [Example](#example)
  - [Graph Compilation](#graph-compilation)
    - [Variable Dimension](#variable-dimension)
    - [Example](#example-1)
    - [Compiling Graphs](#compiling-graphs)
    - [Example](#example-2)
  - [Data Input](#data-input)
    - [Obtaining Input Tensors](#obtaining-input-tensors)
    - [Copying Data](#copying-data)
    - [Example](#example-3)
  - [Graph Execution](#graph-execution)
    - [Executing Sessions](#executing-sessions)
    - [Core Binding](#core-binding)
    - [Callback Running](#callback-running)
    - [Example](#example-4)
  - [Obtaining Outputs](#obtaining-outputs)
    - [Obtaining Output Tensors](#obtaining-output-tensors)
    - [Example](#example-5)
  - [Obtaining Version String](#obtaining-version-string)
    - [Example](#example-6)

<!-- /TOC -->


<a href="https://gitee.com/mindspore/docs/blob/master/lite/tutorials/source_en/use/runtime.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

After model conversion using MindSpore Lite, the model inference process needs to be completed in Runtime.

The procedure for using Runtime is shown in the following figure:

![img](../images/side_infer_process.png)

Its components and their functions are described as follows:
- `Model`: model used by MindSpore Lite, which instantiates the list of operator prototypes through image composition or direct network loading.
- `Lite Session`: provides the graph compilation function and calls the graph executor for inference.
- `Scheduler`: operator heterogeneous scheduler. It can select a proper kernel for each operator based on the heterogeneous scheduling policy, construct a kernel list, and split a graph into subgraphs.
- `Executor`: graph executor, which executes the kernel list to dynamically allocate and release tensors.
- `Operator`: operator prototype, including operator attributes and methods for inferring the shape, data type, and format.
- `Kernel`: operator, which provides specific operator implementation and the operator forwarding function.
- `Tensor`: tensor used by MindSpore Lite, which provides functions and APIs for tensor memory operations.
  
## Reading Models

In MindSpore Lite, a model file is an `.ms` file converted using the model conversion tool. During model inference, the model needs to be loaded from the file system and parsed. Related operations are mainly implemented in the Model component. The Model component holds model data such as weight data and operator attributes.

A model is created based on memory data using the static `Import` method of the Model class. The `Model` instance returned by the function is a pointer, which is created by using `new`. If the pointer is not required, you need to release it by using `delete`.

```cpp
/// \brief   Static method to create a Model pointer.
///
/// \param[in] model_buf  Define the buffer read from a model file.
/// \param[in] size  Define bytes number of model buffer.
///
/// \return  Pointer of MindSpore Lite Model.
static Model *Import(const char *model_buf, size_t size);
```

## Session Creation

When MindSpore Lite is used for inference, sessions are the main entrance of inference. You can compile and execute graphs through sessions.

### Creating Contexts

Contexts save some basic configuration parameters required by sessions to guide graph compilation and execution. The definition of context is as follows:

MindSpore Lite supports heterogeneous inference. The preferred backend for inference is specified by `device_ctx_` in `Context` and is CPU by default. During graph compilation, operator selection and scheduling are performed based on the preferred backend.

```cpp
/// \brief   DeviceType defined for holding user's preferred backend.
typedef enum {
  DT_CPU, /**< CPU device type */
  DT_GPU, /**< GPU device type */
  DT_NPU  /**< NPU device type, not supported yet */
} DeviceType;

/// \brief   DeviceContext defined for holding DeviceType.
typedef struct {
  DeviceType type; /**< device type */
} DeviceContext;

DeviceContext device_ctx_{DT_CPU};
```

MindSpore Lite has a built-in thread pool shared by processes. During inference, `thread_num_` is used to specify the maximum number of threads in the thread pool. The default maximum number is 2. It is recommended that the maximum number be no more than 4. Otherwise, the performance may be affected.

```c++
int thread_num_ = 2; /**< thread number config for thread pool */
```

MindSpore Lite supports dynamic memory allocation and release. If `allocator` is not specified, a default `allocator` is generated during inference. You can also use the `Context` method to allow multiple `Context` to share the memory allocator.

If users create the `Context` by using `new`,  it should be released by using `delete` once it's not required. Usually the `Context` is released after finishing the session creation.

```cpp
/// \brief  Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;

/// \brief  Context defined for holding environment variables during runtime.
class MS_API Context {
 public:
  /// \brief  Constructor of MindSpore Lite Context using input value for parameters.
  ///
  /// \param[in] thread_num  Define the work thread number during the runtime.
  /// \param[in] allocator  Define the allocator for malloc.
  /// \param[in] device_ctx  Define device information during the runtime.
  Context(int thread_num, std::shared_ptr<Allocator> allocator, DeviceContext device_ctx);
    
 public:
	std::shared_ptr<Allocator> allocator = nullptr;
}
```

### Creating Sessions

Use the `Context` created in the previous step to call the static `CreateSession` method of LiteSession to create `LiteSession`. The `LiteSession` instance returned by the function is a pointer, which is created by using `new`. If the pointer is not required, you need to release it by using `delete`.

```cpp
/// \brief  Static method to create a LiteSession pointer.
///
/// \param[in] context  Define the context of session to be created.
///
/// \return  Pointer of MindSpore Lite LiteSession.
static LiteSession *CreateSession(lite::Context *context);
```

### Example

The following sample code demonstrates how to create a `Context` and how to allow two `LiteSession` to share a memory pool.

```cpp
auto context = new (std::nothrow) lite::Context;
if (context == nullptr) {
    MS_LOG(ERROR) << "New context failed while running %s", modelName.c_str();
    return RET_ERROR;
}
// The preferred backend is GPU, which means, if there is a GPU operator, it will run on the GPU first, otherwise it will run on the CPU.
context->device_ctx_.type = lite::DT_GPU;
// The medium core takes priority in thread and core binding methods. This parameter will work in the BindThread interface. For specific binding effect, see the "Run Graph" section.
context->cpu_bind_mode_ = MID_CPU;
// Configure the number of worker threads in the thread pool to 2, including the main thread. 
context->thread_num_ = 2;
// Allocators can be shared across multiple Contexts.
auto *context2 = new Context(context->thread_num_, context->allocator, context->device_ctx_);
context2->cpu_bind_mode_ = context->cpu_bind_mode_;
// Use Context to create Session.
auto session1 = session::LiteSession::CreateSession(context);
// After the LiteSession is created, the Context can be released.
delete (context);
if (session1 == nullptr) {
    MS_LOG(ERROR) << "CreateSession failed while running %s", modelName.c_str();
    return RET_ERROR;
}
// session1 and session2 can share one memory pool.
auto session2 = session::LiteSession::CreateSession(context2);
delete (context2);
if (session == nullptr) {
    MS_LOG(ERROR) << "CreateSession failed while running %s", modelName.c_str();
    return RET_ERROR;
}
```

## Graph Compilation

### Variable Dimension

When using MindSpore Lite for inference, after the session creation and graph compilation have been completed, if you need to resize the input shape, you can reset the shape of the input tensor, and then call the session's Resize() interface.

```cpp
/// \brief  Get input MindSpore Lite MSTensors of model.
///
/// \return  The vector of MindSpore Lite MSTensor.
virtual std::vector<tensor::MSTensor *> GetInputs() const = 0;

/// \brief  Resize inputs shape.
///
/// \param[in] inputs  Define Model inputs.
/// \param[in] dims    Define All inputs new shape.
///
/// \return  STATUS as an error code of resize inputs, STATUS is defined in errorcode.h.
virtual int Resize(const std::vector<tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims) = 0;
```

### Example

The following code demonstrates how to resize the input of MindSpore Lite:

```cpp
// Assume we have created a LiteSession instance named session.
auto inputs = session->GetInputs();
std::vector<int> resize_shape = {1, 128, 128, 3};
// Assume the model has only one input,resize input shape to [1, 128, 128, 3]
std::vector<std::vector<int>> new_shapes;
new_shapes.push_back(resize_shape);
session->Resize(inputs, new_shapes);
```

### Compiling Graphs

Before graph execution, call the `CompileGraph` API of the `LiteSession` to compile graphs and further parse the Model instance loaded from the file, mainly for subgraph split and operator selection and scheduling. This process takes a long time. Therefore, it is recommended that `LiteSession` achieve multiple executions with one creation and one compilation.

```cpp
/// \brief  Compile MindSpore Lite model.
///
/// \note  CompileGraph should be called before RunGraph.
///
/// \param[in] model  Define the model to be compiled.
///
/// \return  STATUS as an error code of compiling graph, STATUS is defined in errorcode.h.
virtual int CompileGraph(lite::Model *model) = 0;
```

### Example

The following code demonstrates how to compile graph of MindSpore Lite:

```cpp
// Assume we have created a LiteSession instance named session and a Model instance named model before.
// The methods of creating model and session can refer to "Import Model" and "Create Session" two sections.
auto ret = session->CompileGraph(model);
if (ret != RET_OK) {
    std::cerr << "CompileGraph failed" << std::endl;
    // session and model need to be released by users manually.
    delete (session);
    delete (model);
    return ret;
}
```

## Data Input

### Obtaining Input Tensors

Before graph execution, you need to copy the input data to model input tensors.

MindSpore Lite provides the following methods to obtain model input tensors.

1. Use the `GetInputsByName` method to obtain vectors of the model input tensors that are connected to the model input node based on the node name.

   ```cpp
   /// \brief  Get input MindSpore Lite MSTensors of model by node name.
   ///
   /// \param[in] node_name  Define node name.
   ///
   /// \return  The vector of MindSpore Lite MSTensor.
   virtual std::vector<tensor::MSTensor *> GetInputsByName(const std::string &node_name) const = 0;
   ```

2. Use the `GetInputs` method to directly obtain the vectors of all model input tensors.

   ```cpp
   /// \brief  Get input MindSpore Lite MSTensors of model.
   ///
   /// \return  The vector of MindSpore Lite MSTensor.
   virtual std::vector<tensor::MSTensor *> GetInputs() const = 0;
   ```

### Copying Data

After model input tensors are obtained, you need to enter data into the tensors. Use the `Size` method of `MSTensor` to obtain the size of the data to be entered into tensors, use the `data_type` method to obtain the data type of tensors, and use the `MutableData` method of `MSTensor` to obtain the writable pointer.

```cpp
/// \brief  Get byte size of data in MSTensor.
///
/// \return  Byte size of data in MSTensor.
virtual size_t Size() const = 0;

/// \brief  Get the pointer of data in MSTensor.
///
/// \note  The data pointer can be used to both write and read data in MSTensor.
///
/// \return  The pointer points to data in MSTensor.
virtual void *MutableData() const = 0;
```

### Example

The following sample code shows how to obtain the entire graph input `MSTensor` from `LiteSession` and enter the model input data to `MSTensor`.

```cpp
// Assume we have created a LiteSession instance named session.
auto inputs = session->GetInputs();
// Assume that the model has only one input tensor.
auto in_tensor = inputs.front();
if (in_tensor == nullptr) {
    std::cerr << "Input tensor is nullptr" << std::endl;
    return -1;
}
// It is omitted that users have read the model input file and generated a section of memory buffer: input_buf, as well as the byte size of input_buf: data_size.
if (in_tensor->Size() != data_size) {
    std::cerr << "Input data size is not suit for model input" << std::endl;
    return -1;
}
auto *in_data = in_tensor->MutableData();
if (in_data == nullptr) {
    std::cerr << "Data of in_tensor is nullptr" << std::endl;
    return -1;
}
memcpy(in_data, input_buf, data_size);
// Users need to free input_buf.
// The elements in the inputs are managed by MindSpore Lite so that users do not need to free inputs.
```

Note:  
- The data layout in the model input tensors of MindSpore Lite must be NHWC.
- The model input `input_buf` is read from disks. After it is copied to model input tensors, you need to release `input_buf`.
- Vectors returned by using the `GetInputs` and `GetInputsByName` methods do not need to be released by users.

## Graph Execution

### Executing Sessions

After a MindSpore Lite session performs graph compilation, you can use `RunGraph` of `LiteSession` for model inference.

```cpp
/// \brief  Run session with callback.
///
/// \param[in] before  Define a call_back_function to be called before running each node.
/// \param[in] after  Define a call_back_function to be called after running each node.
///
/// \note RunGraph should be called after CompileGraph.
///
/// \return  STATUS as an error code of running graph, STATUS is defined in errorcode.h.
virtual int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr) = 0;
```

### Core Binding

The built-in thread pool of MindSpore Lite supports core binding and unbinding. By calling the `BindThread` API, you can bind working threads in the thread pool to specified CPU cores for performance analysis. The core binding operation is related to the context specified when `LiteSession` is created. The core binding operation sets the affinity between a thread and CPU based on the core binding policy in the context.

```cpp
/// \brief  Attempt to bind or unbind threads in the thread pool to or from the specified cpu core.
///
/// \param[in] if_bind  Define whether to bind or unbind threads.
virtual void BindThread(bool if_bind) = 0;
```

Note that core binding is an affinity operation, which is affected by system scheduling. Therefore, successful binding to the specified CPU core cannot be ensured. After executing the code of core binding, you need to perform the unbinding operation. The following is an example:

```cpp
// Assume we have created a LiteSession instance named session.
session->BindThread(true);
auto ret = session->RunGraph();
if (ret != mindspore::lite::RET_OK) {
    std::cerr << "RunGraph failed" << std::endl;
    delete session;
    return -1;
}
session->BindThread(false);
```

> Core binding parameters can be used to bind big cores first or middle cores first.  
> The rule for determining big core or middle core is based on the CPU core frequency instead of CPU architecture. For the CPU architecture where big, middle, and little cores are not distinguished, this rule can be used.  
> Big core first indicates that threads in the thread pool are bound to cores according to core frequency. The first thread is bound to the core with the highest frequency, and the second thread is bound to the core with the second highest frequency. This rule also applies to other threads.  
> Middle cores are defined based on experience. By default, middle cores are cores with the third and fourth highest frequency. Middle core first indicates that threads are bound to middle cores preferentially. When there are no available middle cores, threads are bound to little cores.

### Callback Running

MindSpore Lite can transfer two `KernelCallBack` function pointers to call back the inference model when calling `RunGraph`. Compared with common graph execution, callback running can obtain extra information during the running process to help developers analyze performance and fix bugs. The extra information includes:
- Name of the running node
- Input and output tensors before inference of the current node
- Input and output tensors after inference of the current node

```cpp
/// \brief  CallBackParam defines input arguments for callback function.
struct CallBackParam {
std::string name_callback_param; /**< node name argument */
std::string type_callback_param; /**< node type argument */
};

/// \brief  KernelCallBack defines the function pointer for callback.
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>;
```

### Example

The following sample code demonstrates how to use `LiteSession` to compile a graph, defines two callback functions as the before-callback pointer and after-callback pointer, transfers them to the `RunGraph` API for callback inference, and demonstrates the scenario of multiple graph executions with one graph compilation.

```cpp
// Assume we have created a LiteSession instance named session and a Model instance named model before.
// The methods of creating model and session can refer to "Import Model" and "Create Session" two sections.
auto ret = session->CompileGraph(model);
if (ret != RET_OK) {
    std::cerr << "CompileGraph failed" << std::endl;
    // session and model need to be released by users manually.
    delete (session);
    delete (model);
    return ret;
}
// Copy input data into the input tensor. Users can refer to the "Input Data" section. We uses random data here.
auto inputs = session->GetInputs();
for (auto in_tensor : inputs) {
    in_tensor = inputs.front();
    if (in_tensor == nullptr) {
        std::cerr << "Input tensor is nullptr" << std::endl;
        return -1;
    }
    // When calling the MutableData method, if the data in MSTensor is not allocated, it will be malloced. After allocation, the data in MSTensor can be considered as random data.
    (void) in_tensor->MutableData();
}
// Definition of callback function before forwarding operator.
auto before_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &before_inputs,
                             const std::vector<mindspore::tensor::MSTensor *> &before_outputs,
                             const session::CallBackParam &call_param) {
    std::cout << "Before forwarding " << call_param.name_callback_param << std::endl;
    return true;
};
// Definition of callback function after forwarding operator.
auto after_call_back_ = [&](const std::vector<mindspore::tensor::MSTensor *> &after_inputs,
                            const std::vector<mindspore::tensor::MSTensor *> &after_outputs,
                            const session::CallBackParam &call_param) {
    std::cout << "After forwarding " << call_param.name_callback_param << std::endl;
    return true;
};
// Call the callback function when performing the model inference process.
ret = session_->RunGraph(before_call_back_, after_call_back_);
if (ret != RET_OK) {
  MS_LOG(ERROR) << "Run graph failed.";
  return RET_ERROR;
}
// CompileGraph would cost much time, a better solution is calling CompileGraph only once and RunGraph much more times.
for (size_t i = 0; i < 10; i++) {
    auto ret = session_->RunGraph();
    if (ret != RET_OK) {
        MS_LOG(ERROR) << "Run graph failed.";
        return RET_ERROR;
    }
}
// session and model needs to be released by users manually.
delete (session);
delete (model);
```

## Obtaining Outputs

### Obtaining Output Tensors

After performing inference, MindSpore Lite can obtain the model inference result.

MindSpore Lite provides the following methods to obtain the model output `MSTensor`.
1. Use the `GetOutputsByNodeName` method to obtain vectors of the model output `MSTensor` that is connected to the model output node based on the node name.

   ```cpp
   /// \brief  Get output MindSpore Lite MSTensors of model by node name.
   ///
   /// \param[in] node_name Define node name.
   ///
   /// \return  The vector of MindSpore Lite MSTensor.
   virtual std::vector<tensor::MSTensor *> GetOutputsByNodeName(const std::string &node_name) const = 0;
   ```

2. Use the `GetOutputByTensorName` method to obtain the model output `MSTensor` based on the tensor name.

   ```cpp
   /// \brief  Get output MindSpore Lite MSTensors of model by tensor name.
   ///
   /// \param[in] tensor_name  Define tensor name.
   ///
   /// \return  Pointer of MindSpore Lite MSTensor.
   virtual mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const = 0;
   ```

3. Use the `GetOutputs` method to directly obtain the mapping between the names of all model output tensors and the model output `MSTensor`.

   ```cpp
   /// \brief  Get output MindSpore Lite MSTensors of model mapped by tensor name.
   ///
   /// \return  The map of output tensor name and MindSpore Lite MSTensor.
   virtual std::unordered_map<std::string, mindspore::tensor::MSTensor *> GetOutputs() const = 0;
   ```

After model output tensors are obtained, you need to enter data into the tensors. Use the `Size` method of `MSTensor` to obtain the size of the data to be entered into tensors, use the `data_type` method to obtain the data type of `MSTensor`, and use the `MutableData` method of `MSTensor` to obtain the writable pointer.

```cpp
/// \brief  Get byte size of data in MSTensor.
///
/// \return  Byte size of data in MSTensor.
virtual size_t Size() const = 0;

/// \brief  Get data type of the MindSpore Lite MSTensor.
///
/// \note  TypeId is defined in mindspore/mindspore/core/ir/dtype/type_id.h. Only number types in TypeId enum are
/// suitable for MSTensor.
///
/// \return  MindSpore Lite TypeId of the MindSpore Lite MSTensor.
virtual TypeId data_type() const = 0;

/// \brief  Get the pointer of data in MSTensor.
///
/// \note The data pointer can be used to both write and read data in MSTensor.
///
/// \return  The pointer points to data in MSTensor.
virtual void *MutableData() const = 0;
```

### Example

The following sample code shows how to obtain the output `MSTensor` from `LiteSession` using the `GetOutputs` method and print the first ten data or all data records of each output `MSTensor`.

```cpp
// Assume we have created a LiteSession instance named session before.
auto output_map = session->GetOutputs();
// Assume that the model has only one output node.
auto out_node_iter = output_map.begin();
std::string name = out_node_iter->first;
// Assume that the unique output node has only one output tensor.
auto out_tensor = out_node_iter->second;
if (out_tensor == nullptr) {
    std::cerr << "Output tensor is nullptr" << std::endl;
    return -1;
}
// Assume that the data format of output data is float 32.
if (out_tensor->data_type() != mindspore::TypeId::kNumberTypeFloat32) {
    std::cerr << "Output of lenet should in float32" << std::endl;
    return -1;
}
auto *out_data = reinterpret_cast<float *>(out_tensor->MutableData());
if (out_data == nullptr) {
    std::cerr << "Data of out_tensor is nullptr" << std::endl;
    return -1;
}
// Print the first 10 float data or all output data of the output tensor. 
std::cout << "Output data: ";
for (size_t i = 0; i < 10 && i < out_tensor->ElementsNum(); i++) {
    std::cout << " " << out_data[i];
}
std::cout << std::endl;
// The elements in outputs do not need to be free by users, because outputs are managed by the MindSpore Lite.
```

Note that the vectors or map returned by the `GetOutputsByNodeName`, `GetOutputByTensorName` and `GetOutputs` methods do not need to be released by users.

The following sample code shows how to obtain the output `MSTensor` from `LiteSession` using the `GetOutputsByNodeName` method.

```cpp
// Assume we have created a LiteSession instance named session before.
// Assume that model has a output node named output_node_name_0.
auto output_vec = session->GetOutputsByNodeName("output_node_name_0");
// Assume that output node named output_node_name_0 has only one output tensor.
auto out_tensor = output_vec.front();
if (out_tensor == nullptr) {
    std::cerr << "Output tensor is nullptr" << std::endl;
    return -1;
}
```

The following sample code shows how to obtain the output `MSTensor` from `LiteSession` using the `GetOutputByTensorName` method.

```cpp
// We can use GetOutputTensorNames method to get all name of output tensor of model which is in order.
auto tensor_names = this->GetOutputTensorNames();
// Assume we have created a LiteSession instance named session before.
// Use output tensor name returned by GetOutputTensorNames as key
for (auto tensor_name : tensor_names) {
    auto out_tensor = this->GetOutputByTensorName(tensor_name);
    if (out_tensor == nullptr) {
        std::cerr << "Output tensor is nullptr" << std::endl;
        return -1;
    }
}
```

## Obtaining Version String

### Example

The following sample code shows how to obtain version string using `Version` method.

```cpp
#include "include/version.h"
std::string version = mindspore::lite::Version(); 
```
