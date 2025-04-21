# Building Custom Operators Online

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/advanced/third_party/register_kernel.md)

## Implementing Custom Operators

MindSpore Lite provides a southbound operator registration mechanism. This document describes how to schedule your own operators through the MindSpore Lite framework.

To implement custom operators, perform the following steps:

1. Determine operator types: Classify operators into common and custom operators.
2. Implement operators: Inherit the Kernel class to implement custom operators and register them in MindSpore Lite.
3. Implement the InferShape capability: Inherit mindspore::kernel::KernelInteface to implement the InferShape capability of custom operators and register them in MindSpore Lite.

### Determining Operator Types

View the operator prototype definition in mindspore/lite/schema/ops.fbs. Check whether the operator prototype to be registered is defined in PrimitiveType. If yes, the operator is a common operator, and you can implement and register the operator based on the existing IR. Otherwise, the operator is a custom operator.

### Common Operators

For details about code related to implementation, registration, and InferShape of an operator, see [the code repository](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/src/registry/registry_test.cc).

#### Implementing Common Operators

Inherit [mindspore::kernel::Kernel](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_cpp/mindspore_kernel.html) and overload necessary APIs. The following describes how to customize an Add operator:

1. An operator inherits a kernel.
2. PreProcess() pre-allocates memory.
3. Execute() adds inputs.

```cpp
using mindspore::kernel::Kernel;

class TestCustomAdd : public Kernel {
 public:
  TestCustomAdd(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
                const schema::Primitive *primitive, const lite::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}
  int Prepare() override { return kSuccess; }

  int Execute() override;

  int ReSize() { return kSuccess; }

 private:
  int PreProcess() {
    for (auto *output : outputs_) {
      // malloc data for output tensor
      auto data = output->MutableData();
      if (data == nullptr) {
        MS_LOG(ERROR) << "Get data failed";
        return kLiteError;
      }
    }
    return kSuccess;
  }
};

int TestCustomAdd::Execute() {
  if (inputs_.size() != 2) {
    return kLiteParamInvalid;
  }
  PreProcess();
  auto *in0 = static_cast<const float *>(inputs_[0].Data().get());
  auto *in1 = static_cast<const float *>(inputs_[1].Data().get());
  float *out = static_cast<float *>(outputs_[0].MutableData());
  auto num = outputs_[0].ElementNum();
  for (int i = 0; i < num; ++i) {
    out[i] = in0[i] + in1[i];
  }
  return kSuccess;
}
```

#### Registering Common Operators

Currently, the generated macro [REGISTER_KERNEL](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_registry_RegisterKernel.html) is provided for operator registration. The implementation procedure is as follows:

1. The TestCustomAddCreator function is used to create a kernel.
2. Use the macro REGISTER_KERNEL to register the kernel. Assume that the vendor is BuiltInTest.

```cpp
using mindspore::schema::PrimitiveType_AddFusion;

std::shared_ptr<Kernel> TestCustomAddCreator(const std::vector<tensor::MSTensor *> &inputs,
                                             const std::vector<tensor::MSTensor *> &outputs,
                                             const schema::Primitive *primitive, const lite::Context *ctx) {
  return std::make_shared<TestCustomAdd>(inputs, outputs, primitive, ctx);
}
const auto kFloat32 = DataType::kNumberTypeFloat32;

REGISTER_KERNEL(CPU, BuiltInTest, kFloat32, PrimitiveType_AddFusion, TestCustomAddCreator)
```

#### Common Operator InferShape

Reload the Infer function after inheriting KernelInterface to implement the InferShape capability. The implementation procedure is as follows:

1. Inherit [KernelInterface](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_kernel_KernelInterface.html).
2. Overload the Infer function to derive the shape, format, and data_type of the output tensor.

The following uses the custom Add operator as an example:

```cpp
using mindspore::kernel::KernelInterface;

class TestCustomAddInfer : public KernelInterface {
 public:
  TestCustomAddInfer() = default;
  ~TestCustomAddInfer() = default;
  Status Infer(std::vector<mindspore::MSTensor *> *inputs, std::vector<mindspore::MSTensor *> *outputs,
               const schema::Primitive *primitive) override {
    (*outputs)[0].SetFormat((*inputs)[0].format());
    (*outputs)[0].SetDataType((*inputs)[0].DataType());
    (*outputs)[0].SetShape((*inputs)[0].Shape());
    return kSuccess;
  }
};
```

#### Registering the Common Operator InferShape

Currently, the generated macro [REGISTER_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_registry_RegisterKernelInterface.html) is provided for registering the operator InferShape. The procedure is as follows:

1. Use the CustomAddInferCreator function to create a KernelInterface instance.
2. Call the REGISTER_KERNEL_INTERFACE macro to register the common operator InferShape. Assume that the vendor is BuiltInTest.

```cpp
std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomAddInfer>(); }

REGISTER_KERNEL_INTERFACE(BuiltInTest, PrimitiveType_AddFusion, CustomAddInferCreator)
```

### Custom Operators

For details about code related to parsing, creating, and operating custom operators, see [the code repository](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/tools/converter/registry/pass_registry_test.cc).

#### Defining Custom Operators

```css
table Attribute {
    name: string;
    data: [ubyte];
}

table Custom {
    type: string;
    attr: [Attribute];
}
```

Attributes are stored in a dictionary. `name` indicates the attribute name. `data` indicates the byte stream of the attribute content.
`type` indicates the custom operator type.

#### Creating Custom Operators

You can register your own Pass using the Pass registration API of the `Converter` to export the required operator structure. The following describes how to convert the AddN operator into a custom operator:

1. Assume that the custom operator has the input_num and op_kind attributes.
2. Customize Pass subclasses to convert and create custom operators.
3. Register the custom Pass class.

```cpp
namespace mindspore::opt {
class Test2Fusion : public Pass {
 public:
  AnfNodePtr CreateCustomOp(const FuncGraphPtr func_graph, const CNodePtr cnode) {
    if (func_graph == nullptr || cnode == nullptr) {
      return nullptr;
    }
    auto primc = std::make_shared<ops::Custom>();      // Create a primitive to store operator attributes.
    if (primc == nullptr) {
      return nullptr;
    }
    primc->set_type("Custom_AddN");        // Set the custom operator type.
    std::map<std::string, std::vector<uint8_t>> custom_attrs;
    std::string input_num = std::to_string(cnode->size() - 1);
    std::vector<uint8_t> input_num_attr(input_num.begin(), input_num.end());
    custom_attrs["input_num"] = input_num_attr;
    std::string op_kind = "custom op";
    std::vector<uint8_t> op_kind_attr(op_kind.begin(), op_kind.end());
    custom_attrs["op_kind"] = op_kind_attr;
    primc->set_attr(custom_attrs);         // Set the custom operator attributes.
    auto inputs = cnode->inputs();
    inputs.erase(inputs.begin());
    auto custom_cnode = func_graph->NewCNode(primc, inputs);         // Create a CNode.
    custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());     // Set the node name.
    custom_cnode->set_abstract(cnode->abstract()->Clone());          // Set basic attributes of the operator output and store them in abstract.
    return custom_cnode;
  }

  bool Run(const FuncGraphPtr &func_graph) override {
    auto manager = Manage(func_graph, true);       // Create a FuncGrap manager.
    if (manager == nullptr) {
      return false;
    }
    auto node_list = TopoSort(func_graph->get_return());      // Obtain all nodes.
    for (auto &node : node_list) {
      if (!utils::isa<CNode>(node)) {
        continue;
      }
      if (!opt::CheckPrimitiveType(node, prim::kPrimAddN)) {     // Check whether the current node is an AddN operator.
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      auto custom_cnode = CreateCustomOp(func_graph, cnode);    // Create a custom operator.
      if (custom_cnode == nullptr) {
        return false;
      }
      manager->Replace(node, custom_cnode)        // Replace old nodes with new nodes through the manager.
    }
    return true;
  }
};

REG_PASS(Test1Fusion, Test1Fusion)    // Register Test1Fusion.
REG_PASS(Test2Fusion, Test2Fusion)    // Register Test2Fusion.
std::vector<std::string> schedule = {"Test1Fusion", "Test2Fusion"};
REG_SCHEDULED_PASS(POSITION_BEGIN, schedule)       // Set the external Pass scheduling logic and run the external Pass before the built-in fusion.
}  // namespace mindspore::opt
```

For details about code related to implementation, registration, and InferShape of a custom operator, see [the code repository](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/src/registry/registry_custom_op_test.cc).

#### Implementing Custom Operators

The implementation procedure of a custom operator is the same as that of a common operator, because they are specific subclasses of [Kernel](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_cpp/mindspore_kernel.html).
If the custom operator does not run on the CPU platform, the result needs to be copied back to the output tensor after the running is complete. The following describes how to create a custom operator with the Add capability:

1. An operator inherits a kernel.
2. PreProcess() pre-allocates memory.
3. Execute() adds inputs.

```cpp
using mindspore::kernel::Kernel;

class TestCustomOp : public Kernel {
 public:
  TestCustomOp(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
               const schema::Primitive *primitive, const lite::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}
  int Prepare() override { return kSuccess; }

  int Execute() override;

  int ReSize() override { return kSuccess; }

 private:
  int PreProcess() {
    for (auto *output : outputs_) {
      // malloc data for output tensor
      auto data = output->MutableData();
      if (data == nullptr) {
        MS_LOG(ERROR) << "Get data failed";
        return kLiteError;
      }
    }
    return kSuccess;
  }

int TestCustomOp::Execute() {
  if (inputs_.size() != 2) {
    return kLiteParamInvalid;
  }
  PreProcess();
  GetAttrData();
  const float *in0 = static_cast<const float *>(inputs_[0].Data().get());
  const float *in1 = static_cast<const float *>(inputs_[1].Data().get());
  float *out = static_cast<float *>(outputs_[0].MutableData());
  auto num = outputs_[0].ElementNum();
  for (int i = 0; i < num; ++i) {
    out[i] = in0[i] + in1[i];
  }
  return kSuccess;
}
```

#### Custom Operator Attribute Decoding Example

In the example, the byte stream in the attribute is copied to the buffer.

```cpp
    auto prim = primitive_->value_as_Custom();
    if (prim->attr()->size() < 1) {
      return;
    }
    auto data_bytes = prim->attr()->Get(0)->data();
    auto data_size = data_bytes->size();
    char buf[100];
    for (size_t i = 0; i < data_size; ++i) {
      buf[i] = static_cast<char>(data_bytes->Get(i));
    }
    buf[data_size] = 0;
```

#### Registering Custom Operators

Currently, the generated macro [REGISTER_CUSTOM_KERNEL](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_register_kernel.h_REGISTER_CUSTOM_KERNEL-1.html) is provided for operator registration. The procedure is as follows:

1. The TestCustomAddCreator function is used to create a kernel.
2. Use the macro REGISTER_CUSTOM_KERNEL to register an operator. Assume that the vendor is BuiltInTest and the operator type is Add.

```cpp
using mindspore::schema::PrimitiveType_AddFusion;

std::shared_ptr<Kernel> TestCustomAddCreator(const std::vector<tensor::MSTensor *> &inputs,
                                             const std::vector<tensor::MSTensor *> &outputs,
                                             const schema::Primitive *primitive, const lite::Context *ctx) {
  return std::make_shared<TestCustomOp>(inputs, outputs, primitive, ctx);
}
constexpr auto kFloat32 = DataType::kNumberTypeFloat32;
REGISTER_CUSTOM_KERNEL(CPU, BuiltInTest, kFloat32, Add, TestCustomAddCreator)
```

#### Custom Operator InferShape

The overall implementation is the same as that of the common operator InferShape. The procedure is as follows:

1. Inherit [KernelInterface](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_kernel_KernelInterface.html).
2. Overload the Infer function to derive the shape, format, and data_type of the output tensor.

```cpp
class TestCustomOpInfer : public KernelInterface {
 public:
  TestCustomOpInfer() = default;
  ~TestCustomOpInfer() = default;
  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
             const schema::Primitive *primitive) override {
    (*outputs)[0].SetFormat((*inputs)[0].format());
    (*outputs)[0].SetDataType((*inputs)[0].DataType());
    (*outputs)[0].SetShape((*inputs)[0].Shape());
    return kSuccess;
  }
};
```

#### Registering the Custom Operator InferShape

Currently, the generated macro [REGISTER_CUSTOM_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_register_kernel_interface.h_REGISTER_CUSTOM_KERNEL_INTERFACE-1.html) is provided for registering the custom operator InferShape. The procedure is as follows:

1. Use the CustomAddInferCreator function to create a custom KernelInterface.
2. The macro [REGISTER_CUSTOM_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/define_register_kernel_interface.h_REGISTER_CUSTOM_KERNEL_INTERFACE-1.html) is provided for registering the InferShape capability. The operator type Add must be the same as that in REGISTER_CUSTOM_KERNEL_INTERFACE.

```cpp
std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomOpInfer>(); }

REGISTER_CUSTOM_KERNEL_INTERFACE(BuiltInTest, Add, CustomAddInferCreator)
```

## Custom GPU Operators

A set of GPU-related functional APIs are provided to facilitate the development of the GPU-based custom operator and enable the GPU-based custom operator to share the same resources with the internal GPU-based operators to improve the scheduling efficiency. For details about the APIs, see [mindspore::registry::opencl](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_cpp/mindspore_registry_opencl.html).
This document describes how to develop a custom GPU operator by parsing sample code. Before reading this document, you need to understand [Implement Custom Operators](#implementing-custom-operators).
The [code repository](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/src/registry/registry_gpu_custom_op_test.cc) contains implementation and registration of custom GPU operators.

### Registering Operators

In this example, the custom operator `Custom_Add` is registered. For details about how to create and implement this operator, see [Defining Custom Operators](#defining-custom-operators) and [Implementing Custom Operators](#implementing-custom-operators).

#### Implementing a Function for Creating an Operator Instance

Implement a function for creating an operator instance to implement the first step of custom operator registration. The function type is declared in `include/registry/register_kernel.h`, as shown in the following:

```cpp
/// \brief CreateKernel Defined a functor to create a kernel.
///
/// \param[in] inputs Define input tensors of kernel.
/// \param[in] outputs Define output tensors of kernel.
/// \param[in] primitive Define attributes of op.
/// \param[in] ctx Define for holding environment variables during runtime.
///
/// \return Smart Pointer of kernel.
using CreateKernel = std::function<std::shared_ptr<kernel::Kernel>(
  const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs, const schema::Primitive *primitive,
  const mindspore::Context *ctx)>;
```

In this example, the operator instance creation function is implemented as follows. The function returns a `CustomAddKernel` class instance. This class is the user-defined operator class that inherits the `kernel::Kernel` class. For details about the implementation of this class, see [Implementing Operators](#implementing-operators).
In the function, in addition to transferring the function parameters to the constructor function of the `CustomAddKernel` class, a Boolean variable is also transferred. The variable is used to control whether the data type processed by the created `CustomAddKernel` instance is FLOAT32 or FLOAT16.

```cpp
namespace custom_gpu_demo {
std::shared_ptr<kernel::Kernel> CustomAddCreator(const std::vector<MSTensor> &inputs,
                                                 const std::vector<MSTensor> &outputs,
                                                 const schema::Primitive *primitive, const mindspore::Context *ctx) {
  bool fp16_enable = false;

  std::cout << "using fp32 add.\n" << std::endl;
  return std::make_shared<CustomAddKernel>(inputs, outputs, primitive, ctx, fp16_enable);
}
}
```

#### Registering Operators

When registering GPU operators, you must declare the device type as GPU and transfer the operator instance creation function `CustomAddCreator` implemented in the previous step.
In this example, the Float32 implementation of the Custom_Add operator is registered. The registration code is as follows. For details about other parameters in the registration macro, see the [API](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_cpp/mindspore_registry.html).

```cpp
const auto kFloat32 = DataType::kNumberTypeFloat32;
// Register custom "Custom_Add" operator
REGISTER_CUSTOM_KERNEL(GPU, BuiltInTest, kFloat32, Custom_Add, CustomAddCreator)
```

### Implementing Operators

In this example, the operator is implemented as the `CustomAddKernel` class. This class inherits [mindspore::kernel::Kernel](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_cpp/mindspore_kernel.html) and reloads necessary APIs to implement the custom operator computation.

#### Constructor and Destructor Functions

In the constructor function of the `CustomAddKernel` class, the Boolean variable `fp16_enable` is saved, and other parameters are transferred to the constructor function of the base class.
In the destructor function of the `CustomAddKernel` class, `FreeWeight()` is called to release the memory that is temporarily allocated for computation.

```cpp
class CustomAddKernel : public kernel::Kernel {
 public:
  CustomAddKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                  const schema::Primitive *primitive, const mindspore::Context *ctx,
                  bool fp16_enable)
      : Kernel(inputs, outputs, primitive, ctx), fp16_enable_(fp16_enable) {}
  ~CustomAddKernel() override { FreeWeight(); }

  ...
}
```

#### Class Member Variable Description

- opencl_runtime_

  An instance of the OpenCLRuntimeWrapper class. In an operator, this object can be used to call the OpenCL-related API [mindspore::registry::opencl](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_cpp/mindspore_registry_opencl.html) provided by MindSpore Lite.

- fp16_enable_

  Determines whether the operator uses FP16 for computation. To use FP16 for computation, you need to register the operator as an FP16 operator. In this example, the FP32 operator is registered.

- weight_ptrs_

  Pointer to the temporary memory required for operator computation.

- Other variables

  Other variables are required for OpenCL operations. For details, see [mindspore::registry::opencl](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_cpp/mindspore_registry_opencl.html).

```c++
class CustomAddKernel : public kernel::Kernel {
  ...
 private:
  const bool fp16_enable_;
  cl::Kernel kernel_;
  cl::Event event_;
  cl::NDRange global_range_{cl::NullRange};
  cl::NDRange local_range_{cl::NullRange};
  std::vector<void *> weight_ptrs_;
  registry::opencl::OpenCLRuntimeWrapper opencl_runtime_;
}
```

#### Code and Description of the Prepare Implementation

In the graph build phase `mindspore::Model::Build`, the Prepare function of the operator is called. You can perform some time-consuming and one-off operations to save the operator computation time of `mindspore::Model::Predict`.
In this example, the Prepare API is overloaded to load and build the custom OpenCL code.

1. Check the environment.

    In the example, `CheckSpecs` is called to check the running environment of the operator.
    In `CheckSpecs`, the input and output data types and the number of input and output tensors are checked.
    The `MSTensor::IsConst()` API can be used to determine whether the data of a tensor is a constant. The data type of the non-constant input is also compared with the data type declared during operator registration. For details about how to process constant data, see the subsequent tutorials.

    ```cpp
    int Prepare() override {
        auto ret = CheckSpecs();
        if (ret != kSuccess) {
        std::cerr << "Prepare failed for check kernel specs!";
        return ret;
        }
        ...
    }

    int CheckSpecs() {
        for (auto &tensor : inputs_) {
        if (tensor.DataType() != DataType::kNumberTypeFloat32 && tensor.DataType() != DataType::kNumberTypeFloat16) {
            std::cerr << "ArithmeticOpenCLKernel only support fp32/fp16 input";
            return kLiteError;
        }
        }
        for (auto &tensor : outputs_) {
        if (tensor.DataType() != DataType::kNumberTypeFloat32 && tensor.DataType() != DataType::kNumberTypeFloat16) {
            std::cerr << "ArithmeticOpenCLKernel only support fp32/fp16 output";
            return kLiteError;
        }
        }

        if (inputs_.size() != 2 || outputs_.size() != 1) {
        std::cerr << "in size: " << inputs_.size() << ", out size: " << outputs_.size();
        return kLiteError;
        }

        for (int i = 0; i < inputs_.size(); ++i) {
        auto &in_tensor = inputs_.at(i);
        if (!in_tensor.IsConst()) {
            if (fp16_enable_ && in_tensor.DataType() == mindspore::DataType::kNumberTypeFloat32) {
            std::cerr << "Inputs data type error, expectation kNumberTypeFloat16 but kNumberTypeFloat32.";
            return kLiteError;
            } else if (!fp16_enable_ && in_tensor.DataType() == mindspore::DataType::kNumberTypeFloat16) {
            std::cerr << "Inputs data type error, expectation kNumberTypeFloat32 but kNumberTypeFloat16.";
            return kLiteError;
            }
        }
        }

        return kSuccess;
    }
    ```

2. Load the custom OpenCL code.

    Use `opencl_runtime_` to call the `OpenCLRuntimeWrapper::LoadSource` API to load the custom OpenCL code.

    ```cpp
    int Prepare() override {
        ...
        const std::string kernel_name_ = "ElementAdd";
        const std::string program_name = "Arithmetic";
        std::string source = arithmetic_source;
        if (opencl_runtime_.LoadSource(program_name, source) != kSuccess) {
        std::cerr << "Load source failed.";
        return kLiteError;
        }
        ...
    }
    ```

    `arithmetic_source` is the user-defined OpenCL code, as shown in the following:

    ```cpp
    static const char *arithmetic_source =
        "\n"
        "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
        "__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
        "\n"
        "__kernel void ElementAdd(__read_only image2d_t input_a, __read_only image2d_t input_b, __write_only image2d_t "
        "output,\n"
        "                         const int2 output_shape) {\n"
        "  int X = get_global_id(0);\n"
        "  int Y = get_global_id(1);\n"
        "  if (X >= output_shape.x || Y >= output_shape.y) {\n"
        "    return;\n"
        "  }\n"
        "\n"
        "  FLT4 a = READ_IMAGE(input_a, smp_none, (int2)(X, Y));\n"
        "  FLT4 b = READ_IMAGE(input_b, smp_none, (int2)(X, Y));\n"
        "  FLT4 result = a + b;\n"
        "\n"
        "  WRITE_IMAGE(output, (int2)(X, Y), result);\n"
        "}\n";
    ```

3. Build the OpenCL code.

    Use `fp16_enable_` to specify different build options to generate the code for processing FLOAT16 or FLOAT32 data.
    Use `opencl_runtime_` to call the `OpenCLRuntimeWrapper::BuildKernel` API, obtain the built `cl::Kernel` variable, and save it in `kernel_`.

    ```cpp
    int Prepare() override {
        ...
        std::vector<std::string> build_options_ext = {"-cl-mad-enable -cl-fast-relaxed-math -Werror"};
        if (fp16_enable_) {
        build_options_ext.push_back(" -DFLT4=half4 -DWRITE_IMAGE=write_imageh -DREAD_IMAGE=read_imageh");
        } else {
        build_options_ext.push_back(" -DFLT4=float4 -DWRITE_IMAGE=write_imagef -DREAD_IMAGE=read_imagef");
        }

        if (opencl_runtime_.BuildKernel(&kernel_, program_name, kernel_name_, build_options_ext) != kSuccess) {
        std::cerr << "Build kernel failed.";
        return kLiteError;
        }
        ...
    }
    ```

4. Set the OpenCL working group and work items.

    For an operator registered as a GPU, the input data received is in image format except that the input is a constant. The format is NHWC4 (C-axis 4-byte aligned NHWC format data).
    In this example, all data is converted to this format for computation and output.
    In the routine, a simple addition custom operator is implemented. Therefore, the `GpuTensorInfo` function is used to compute the width and height of the `Image` memory used by the output data to set the work items.

    ```cpp
    int Prepare() override {
        ...
        auto out_shape = GpuTensorInfo(&outputs_[0], &opencl_runtime_);
        local_range_ = cl::NullRange;
        global_range_ = cl::NDRange(out_shape.width, out_shape.height);
        ...
    }
    ```

    The implementation of `GpuTensorInfo` is as follows: Use the `Broadcast2GpuShape` function to convert the shape of a tensor to four dimensions, and then compute the shape value when the format is NHWC4.
    Then, obtain the maximum width and height supported by the image memory by calling `OpenCLRuntimeWrapper::GetMaxImage2DWidth` and `OpenCLRuntimeWrapper::GetMaxImage2DHeight`, and determine the image memory width and height actually used by the operator.

    ```cpp
    struct GpuTensorInfo {
        GpuTensorInfo() = default;
        explicit GpuTensorInfo(const MSTensor *tensor, registry::opencl::OpenCLRuntimeWrapper *opencl_run) {
        if (tensor == nullptr) {
            return;
        }
        auto shape_ori = tensor->Shape();
        int64_t shape[4];
        Broadcast2GpuShape(shape, shape_ori.data(), shape_ori.size(), 1l);
        N = shape[0];
        H = shape[1];
        W = shape[2];
        C = shape[3];
        Slice = UP_DIV(C, C4NUM);
        if (tensor->DataType() == mindspore::DataType::kNumberTypeFloat16) {
            FLT_size = sizeof(cl_half);
        } else {
            FLT_size = sizeof(cl_float);
        }
        FLT4_size = FLT_size * 4;
        if (W * Slice <= opencl_run->GetMaxImage2DWidth()) {
            height = N * H;
            width = W * Slice;
        } else {
            height = N * H * W;
            width = Slice;
            if (height > opencl_run->GetMaxImage2DHeight()) {
            height = -1;
            width = -1;
            }
        }

        ElementsNum = N * H * W * C;
        Image2DSize = height * width * FLT4_size;
        }
        size_t N{1};
        size_t H{1};
        size_t W{1};
        size_t C{1};
        size_t Slice{};
        size_t width{};
        size_t height{};
        size_t FLT_size{4};
        size_t FLT4_size{16};
        size_t ElementsNum{};
        size_t Image2DSize{};
    };
    }  // namespace
    ```

    The implementation of `Broadcast2GpuShape` is as follows:

    ```cpp
    template <typename SrcT, typename DstT>
    void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num) {
        if (src == nullptr || src_num <= 0) {
        return;
        }
        auto *N = dst;
        auto *H = dst + 1;
        auto *W = dst + 2;
        auto *C = dst + 3;
        if (src_num == 1) {  // 1 1 1 C
        *C = src[0];
        } else if (src_num == 2) {  // N 1 1 C
        *N = src[0];
        *C = src[1];
        } else if (src_num == 3) {  // N 1 W C
        *N = src[0];
        *W = src[1];
        *C = src[2];
        } else if (src_num == 4) {  // N H W C
        *N = src[0];
        *H = src[1];
        *W = src[2];
        *C = src[3];
        } else if (src_num > 4) {
        std::cerr << "GPU doesn't support ndim>=" << src_num;
        }
    }

    template <typename SrcT, typename DstT>
    void Broadcast2GpuShape(DstT *dst, const SrcT *src, int src_num, DstT default_value) {
        for (int i = 0; i < 4; ++i) {
        dst[i] = default_value;
        }
        if (src == nullptr || src_num <= 0) {
        return;
        }
        Broadcast2GpuShape(dst, src, src_num);
    }
    ```

5. Convert the constant input into data in a proper format and allocate the GPU memory.

    For an operator registered as a GPU, the input data is GPU memory data in image format except when the input is a constant.
    To meet the operator computation requirements, you need to set a proper format for the constant input and allocate GPU memory if necessary. In this example, the operations on a constant tensor are as follows:

    Use the `MSTensor::IsConst()` API to check whether the input is a constant, and use `GpuTensorInfo` to compute the memory size required for converting the image format.
    Allocate the local memory `weight` of this size, and use the `PackNHWCToNHWC4` function to transfer the tensor memory to `weight` for storage.

    ```cpp
    for (int i = 0; i < inputs_.size(); ++i) {
        auto &in_tensor = inputs_.at(i);
        if (in_tensor.IsConst()) {
        GpuTensorInfo in_shape = GpuTensorInfo(&in_tensor, &opencl_runtime_);
        std::vector<char> weight(in_shape.Image2DSize, 0);
        bool src_is_fp16 = in_tensor.DataType() == mindspore::DataType::kNumberTypeFloat16;
        PackNHWCToNHWC4(in_tensor.MutableData(), weight.data(), src_is_fp16, fp16_enable_, in_shape,
                        in_tensor.DataType());
        ...
    ```

    The `PackNHWCToNHWC4` function is implemented as follows, including the conversion between the FLOAT16 and FLOAT32 types.

    ```cpp
    void PackNHWCToNHWC4(void *src, void *dst, bool src_is_fp16, bool dst_is_fp16, const GpuTensorInfo &tensor,
                        mindspore::DataType data_type) {
        auto src_fp16 = reinterpret_cast<float16_t *>(src);
        auto src_fp32 = reinterpret_cast<float32_t *>(src);
        auto src_int32 = reinterpret_cast<int32_t *>(src);
        auto dst_fp16 = reinterpret_cast<float16_t *>(dst);
        auto dst_fp32 = reinterpret_cast<float32_t *>(dst);
        auto dst_int32 = reinterpret_cast<int32_t *>(dst);
        for (int n = 0, src_idx = 0; n < tensor.N; n++) {
        for (int h = 0; h < tensor.H; ++h) {
            for (int w = 0; w < tensor.W; ++w) {
            for (int c = 0; c < tensor.C; ++c, ++src_idx) {
                int dst_idx = ((n * tensor.H + h) * tensor.W + w) * tensor.Slice * C4NUM + c;
                if (data_type == mindspore::DataType::kNumberTypeInt32) {
                dst_int32[dst_idx] = src_int32[src_idx];
                } else if (dst_is_fp16) {
                dst_fp16[dst_idx] = src_is_fp16 ? src_fp16[src_idx] : static_cast<float16_t>(src_fp32[src_idx]);
                } else {
                dst_fp32[dst_idx] = src_is_fp16 ? static_cast<float32_t>(src_fp16[src_idx]) : src_fp32[src_idx];
                }
            }
            }
        }
        }
        if (tensor.ElementsNum == 1) {
        if (dst_is_fp16) {
            dst_fp16[3] = dst_fp16[2] = dst_fp16[1] = dst_fp16[0];
        } else {
            dst_fp32[3] = dst_fp32[2] = dst_fp32[1] = dst_fp32[0];
        }
        }
    }
    ```

    `OpenCLRuntimeWrapper::GetAllocator` is used to obtain the memory allocator that allocates the memory.
    Then, the `mindspore::Allocator::Malloc` API of the allocator can be used to apply for the GPU memory in image format.
    Write the `weight` data in the NHWC4 format to the GPU memory through the `OpenCLRuntimeWrapper::WriteImage(void *buffer, void *src_data)` API.
    The pointer to the requested GPU memory is stored in weight_ptrs_ so that it can be released during destruction.

    ```cpp
    DataType dtype =
        fp16_enable_ ? mindspore::DataType::kNumberTypeFloat16 : mindspore::DataType::kNumberTypeFloat32;
    auto allocator = opencl_runtime_.GetAllocator();
    if (allocator == nullptr) {
        std::cerr << "GetAllocator fail.";
        FreeWeight();
        return kLiteError;
    }
    auto weight_ptr = allocator->Malloc(in_shape.width, in_shape.height, dtype);
    if (weight_ptr == nullptr) {
        std::cerr << "Malloc fail.";
        FreeWeight();
        return kLiteError;
    }
    weight_ptrs_.push_back(weight_ptr);
    if (opencl_runtime_.WriteImage(weight_ptr, weight.data()) != kSuccess) {
        std::cerr << "WriteImage fail.";
        FreeWeight();
        return kLiteError;
    }
    ```

    During destruction, the function called for releasing the GPU memory is as follows. The memory allocator that allocates the GPU memory is obtained by using `OpenCLRuntimeWrapper::GetAllocator`.
    Then, the `mindspore::Allocator::Free` API of the allocator can be used to release the applied GPU memory.

    ```cpp
    void FreeWeight() {
        auto allocator = opencl_runtime_.GetAllocator();
        if (allocator == nullptr) {
            std::cerr << "GetAllocator fail.";
            return;
        }
        for (auto &weight_ptr : weight_ptrs_) {
            if (weight_ptr != nullptr) {
            allocator->Free(weight_ptr);
            weight_ptr = nullptr;
            }
        }
        }
    ```

6. Set values of OpenCL kernel runtime parameters.

    Some unchanged parameters during the OpenCL kernel running can be set in the `Prepare` phase.
    In this example, `OpenCLRuntimeWrapper::SetKernelArg` is used to set the third parameter (computation range) of the `ElementAdd` runtime.

    ```cpp
    int arg_idx = 3;
    cl_int2 output_shape{static_cast<int>(global_range_[0]), static_cast<int>(global_range_[1])};
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx, output_shape) != kSuccess) {
        std::cerr << "Set kernel arg" << arg_idx << "failed.";
        FreeWeight();
        return kLiteError;
    }
    ```

#### Code and Description of the ReSize and Execute Implementations

By overloading `Execute`, you can customize the computation operations of the operator during inference.

1. Call the `ReSize` function to support shape changes during running.

    In this example, `PreProcess` is called to prepare for the computation.
    In `PreProcess()`, call the `ReSize` function first. This function is the runtime shape change adaptation API that needs to be overloaded.
    In the `ReSize` function, call `CheckOutputs` to check whether the shape of the output tensor of the operator contains invalid values to determine whether shape inference needs to be performed again. If no, the function returns directly.
    When shape inference is required, call `registry::RegisterKernelInterface::GetKernelInterface` to obtain the shape inference function registered by the operator. The obtained function is the InferShape function implemented and registered by the user in this routine.
    After re-inference, call the previously implemented `Prepare` API to re-apply for and allocate the memory and related variables required for operator computation.

    ```cpp
    int ReSize() override {
        if (CheckOutputs(outputs_) == kSuccess) {
        return kSuccess;
        }
        auto status =
        registry::RegisterKernelInterface::GetKernelInterface("", primitive_)->Infer(&inputs_, &outputs_, primitive_);
        if (status != kSuccess) {
        std::cerr << "infer failed." << std::endl;
        return kLiteError;
        }
        ret = Prepare();
        if (ret != kSuccess) {
        std::cerr << "ReSize failed for kernel prepare!";
        return ret;
        }
        return kSuccess;
    }

    int PreProcess() {
        int ret;
        ret = ReSize();
        if (ret != kSuccess) {
        return ret;
        }
        ...
    }

    int Execute() override {
        if (inputs_.size() != 2) {
        return kLiteParamInvalid;
        }
        PreProcess();
        ...
    }
    ```

2. Allocate memory for the output tensor.

    Before running the operator, you need to apply for GPU memory for the output tensor. Due to the limitation of the framework, the GPU memory needs to be hosted by the framework for management and cannot be manually released. The process is as follows:

    1. Call the `allocator()` API of the output tensor to obtain the memory allocator that manages the tensor in the framework. In the GPU registration operator, the memory allocator is responsible for allocating the GPU memory.
    2. Compute the size of the memory to be allocated. In this example, the `GpuTensorInfo` function is used to compute the size of the memory to be allocated.
    3. Apply for memory by using the `Malloc` API of the memory allocator. You can obtain the memory in image or buffer format by using the `void *Malloc(size_t weight, size_t height, DataType type)` and `void *Malloc(size_t size)` APIs.
    4. Use the `SetData` API to assign the requested memory to the tensor. After that, the memory is managed by the framework in a unified manner and cannot be manually released.

    ```cpp
    int PreProcess() {
        ...
        for (auto i = 0; i < outputs_.size(); ++i) {
        auto *output = &outputs_.at(i);
        auto img_info = GpuTensorInfo(output, &opencl_runtime_);
        auto allocator = output->allocator();
        if (allocator == nullptr) {
            std::cerr << "The output tensor of OpenCL kernel must have an allocator.";
            return kLiteError;
        }
        auto data_ptr = allocator->Malloc(img_info.width, img_info.height, output->DataType());
        if (data_ptr == nullptr) {
            std::cerr << "Malloc data failed";
            return kLiteError;
        }
        output->SetData(data_ptr);
        }
        return kSuccess;
    }
    ```

3. Run the OpenCL kernel.

    The `SetKernelArg` API is used to set parameters for running the OpenCL kernel, and the `RunKernel` API is used to run the OpenCL kernel.

    ```cpp
    int Execute() override {
        ...
        std::cout << this->name() << " Running!" << std::endl;
        auto input_0_ptr = weight_ptrs_[0] == nullptr ? inputs_[0].MutableData() : weight_ptrs_[0];
        auto input_1_ptr = weight_ptrs_[1] == nullptr ? inputs_[1].MutableData() : weight_ptrs_[1];
        int arg_idx = 0;
        if (opencl_runtime_->SetKernelArg(kernel_, arg_idx++, input_0_ptr) != kSuccess) {
        std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
        return kLiteError;
        }
        if (opencl_runtime_->SetKernelArg(kernel_, arg_idx++, input_1_ptr) != kSuccess) {
        std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
        return kLiteError;
        }
        if (opencl_runtime_->SetKernelArg(kernel_, arg_idx++, outputs_[0].MutableData()) != kSuccess) {
        std::cerr << "Set kernel arg" << arg_idx - 1 << "failed.";
        return kLiteError;
        }
        if (opencl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != kSuccess) {
        std::cerr << "Run kernel failed.";
        return kLiteError;
        }

        return kSuccess;
    }
    ```
