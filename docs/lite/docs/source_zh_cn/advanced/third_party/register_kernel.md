# 在线构建自定义算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/advanced/third_party/register_kernel.md)

## 如何实现自定义算子

MindSpore Lite当前提供了一套南向的算子注册机制，如果用户想通过MindSpore Lite框架调度到自己的算子实现上，可参考本文。

实现自定义算子大概有以下几个步骤：

1. 确定算子类型：分为通用算子与Custom算子。
2. 算子实现：继承Kernel类实现自定义算子，并注册进MindSpore Lite。
3. 算子InferShape：继承mindspore::kernel::KernelInteface实现自定义算子的InferShape能力，并注册进MindSpore Lite。

### 确定算子类型

查看mindspore/lite/schema/ops.fbs中的算子原型定义，确认要注册实现的算子原型是否在PrimitiveType中有定义，有定义的话则要注册的算子为通用算子，可以按照已有的IR直接实现算子与注册，否则即为Custom算子。

### 通用算子

整个算子的实现、注册、infershape等相关的代码可以参考代码仓里的[样例](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/src/registry/registry_test.cc)。

#### 通用算子实现

继承[mindspore::kernel::Kernel](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html)，重载实现必要的接口。以自定义一个Add算子为例：

1. 算子继承Kernel。
2. PreProcess()对内存进行了预分配。
3. Execute()对input进行了相加。

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

#### 通用算子注册

当前提供的现成的宏[REGISTER_KERNEL](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-kernel)可以进行算子注册，实现步骤如下：

1. 函数TestCustomAddCreator用来创建Kernel。
2. 通过宏REGISTER_KERNEL进行Kernel注册，这里生产商假定为BuiltInTest。

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

#### 通用算子InferShape

继承KernelInterface后重载Infer函数，实现InferShape能力。实现步骤如下：

1. 继承[KernelInterface](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#kernelinterface)。
2. 重载实现Infer函数，推导出output tensor的shape、format、data_type。

这里以自定义Add算子为例：

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

#### 通用算子InferShape注册

当前提供现成的宏[REGISTER_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-kernel-interface)可以进行算子InferShape注册，步骤如下：

1. 函数CustomAddInferCreator用来创建KernelInterface实例。
2. 调用REGISTER_KERNEL_INTERFACE宏对通用算子InferShape进行注册，这里生产商假定为BuiltInTest。

```cpp
std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomAddInfer>(); }

REGISTER_KERNEL_INTERFACE(BuiltInTest, PrimitiveType_AddFusion, CustomAddInferCreator)
```

### Custom算子

Custom算子的解析、创建、操作等相关的代码可以参考代码仓里的[样例](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/tools/converter/registry/pass_registry_test.cc)。

#### Custom算子定义

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

属性是以字典的形式进行存储：name解释了属性名，data里存储了属性内容的字节流。
type：Custom算子的类型。

#### Custom算子创建

通过转换工具`Converter`的Pass注册接口，可以注册用户自己的Pass，用以导出想要的算子结构。这里以AddN算子转为一个Custom算子为例：

1. 假设Custom算子存在"input_num"、"op_kind"属性。
2. 通过自定义Pass子类，实现Custom算子的转换与创建。
3. 注册自定义Pass类。

```cpp
namespace mindspore::opt {
class Test2Fusion : public Pass {
 public:
  AnfNodePtr CreateCustomOp(const FuncGraphPtr func_graph, const CNodePtr cnode) {
    if (func_graph == nullptr || cnode == nullptr) {
      return nullptr;
    }
    auto primc = std::make_shared<ops::Custom>();      // 创建Primitive，存储算子属性
    if (primc == nullptr) {
      return nullptr;
    }
    primc->set_type("Custom_AddN");        // 设置Custom算子类型
    std::map<std::string, std::vector<uint8_t>> custom_attrs;
    std::string input_num = std::to_string(cnode->size() - 1);
    std::vector<uint8_t> input_num_attr(input_num.begin(), input_num.end());
    custom_attrs["input_num"] = input_num_attr;
    std::string op_kind = "custom op";
    std::vector<uint8_t> op_kind_attr(op_kind.begin(), op_kind.end());
    custom_attrs["op_kind"] = op_kind_attr;
    primc->set_attr(custom_attrs);         // 设置Custom算子属性
    auto inputs = cnode->inputs();
    inputs.erase(inputs.begin());
    auto custom_cnode = func_graph->NewCNode(primc, inputs);         // 创建CNode节点
    custom_cnode->set_fullname_with_scope(cnode->fullname_with_scope());     // 设置节点名
    custom_cnode->set_abstract(cnode->abstract()->Clone());          // 设置算子输出的基本属性，存储于abstract中
    return custom_cnode;
  }

  bool Run(const FuncGraphPtr &func_graph) override {
    auto manager = Manage(func_graph, true);       // 创建FuncGrap管理器
    if (manager == nullptr) {
      return false;
    }
    auto node_list = TopoSort(func_graph->get_return());      // 获取所有节点
    for (auto &node : node_list) {
      if (!utils::isa<CNode>(node)) {
        continue;
      }
      if (!opt::CheckPrimitiveType(node, prim::kPrimAddN)) {     // 判断当前节点是否为AddN算子
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      auto custom_cnode = CreateCustomOp(func_graph, cnode);    // 创建Custom算子
      if (custom_cnode == nullptr) {
        return false;
      }
      manager->Replace(node, custom_cnode)        // 通过管理器用新节点替换旧节点
    }
    return true;
  }
};

REG_PASS(Test1Fusion, Test1Fusion)    // 注册Test1Fusion
REG_PASS(Test2Fusion, Test2Fusion)    // 注册Test2Fusion
std::vector<std::string> schedule = {"Test1Fusion", "Test2Fusion"};
REG_SCHEDULED_PASS(POSITION_BEGIN, schedule)       // 设置外部Pass调度逻辑，在内置融合前运行外部Pass
}  // namespace mindspore::opt
```

整个Custom算子的实现、注册、infershape等相关的代码可以参考代码仓里的[样例](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/src/registry/registry_custom_op_test.cc)。

#### Custom算子实现

Custom算子的实现整体流程与通用算子的实现是一致的，因为都是[Kernel](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html)的具体子类。
如果自定义算子不是运行在CPU平台上，需要在运行结束时把结果重新拷回output tensor。这里以创建一个Add能力的Custom算子为例：

1. 算子继承Kernel。
2. PreProcess()对内存进行了预分配。
3. Execute()对input进行了相加。

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

#### Custom算子属性解码样例

样例中是把属性里的字节流复制到了buf内。

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

#### Custom算子注册

当前提供的现成的宏[REGISTER_CUSTOM_KERNEL](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-custom-kernel)可以进行算子注册，步骤如下：

1. TestCustomAddCreator函数用来创建Kernel。
2. 通过宏REGISTER_CUSTOM_KERNEL进行算子注册，这里假定生产商为BuiltInTest，算子类型为Add。

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

#### Custom算子InferShape

整体实现与通用算子InferShape是一样的。步骤如下：

1. 继承[KernelInterface](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html#kernelinterface)。
2. 重载实现Infer函数，推导出output tensor的shape、format、data_type。

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

#### Custom算子InferShape注册

当前提供的现成的宏[REGISTER_CUSTOM_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-custom-kernel-interface)可以进行Custom算子InferShape的注册，步骤如下：

1. CustomAddInferCreator函数用于创建自定义的KernelInterface。
2. 通过宏[REGISTER_CUSTOM_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#register-custom-kernel-interface)注册InferShape能力，这里的算子类型Add必须与REGISTER_CUSTOM_KERNEL_INTERFACE时的算子类型一致。

```cpp
std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomOpInfer>(); }

REGISTER_CUSTOM_KERNEL_INTERFACE(BuiltInTest, Add, CustomAddInferCreator)
```

## 自定义GPU算子

为支持GPU自定义算子的便捷开发，并使GPU自定义算子与内部的GPU算子共享一套资源，以加快调度效率，我们还提供了一套GPU相关的功能接口，相关API说明请参考[mindspore::registry::opencl](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry_opencl.html)。
本文以样例代码解析的方式，向用户阐明自定义GPU算子开发的相关实现。用户需对[如何实现自定义算子](#如何实现自定义算子)有所了解的情况下，再来阅读此文。
在代码仓[样例代码](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/ut/src/registry/registry_gpu_custom_op_test.cc)中包含了对自定义GPU算子的实现、注册。

### 算子注册

本样例中注册的是`Custom_Add`自定义算子。关于该算子的创建与实现，请参考[Custom算子定义](#custom算子定义)和[Custom算子实现](#custom算子实现)。

#### 实现创建算子实例的函数

实现自定义算子注册的第一步，需实现一个创建算子实例的函数。函数类型声明在`include/registry/register_kernel.h`，如下所示：

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

本例中实现的创建算子实例函数如下，函数返回一个`CustomAddKernel`类实例，该类为继承`kernel::Kernel`类的用户自定义算子类，关于该类的实现参考[算子实现](#算子实现)。
在函数内，除了将函数参数传递给`CustomAddKernel`类的构造函数外，还传递了一个布尔型的变量，该变量用于控制创建的`CustomAddKernel`实例处理的数据类型是float32还是float16。

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

#### 注册算子

在注册GPU算子时，必须将设备类型声明为GPU，并将上一步实现的创建算子实例函数`CustomAddCreator`传入。
本样例注册了`Custom_Add`算子GPU内的float32实现，注册代码如下所示，注册宏中的其他参数参考[API说明](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html)。

```cpp
const auto kFloat32 = DataType::kNumberTypeFloat32;
// Register custom "Custom_Add" operator
REGISTER_CUSTOM_KERNEL(GPU, BuiltInTest, kFloat32, Custom_Add, CustomAddCreator)
```

### 算子实现

在本样例中算子实现为`CustomAddKernel`类，该类继承[mindspore::kernel::Kernel](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_kernel.html)，重载实现必要的接口，从而实现自定义算子的运算。

#### 构造及析构函数说明

在`CustomAddKernel`类构造函数中，保存了传递进来的布尔变量`fp16_enable`，并将其他参数传递给基类的构造函数。
在`CustomAddKernel`类析构函数中，调用`FreeWeight()`对因运算需要而临时申请的内存进行释放。

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

#### 类成员变量说明

- opencl_runtime_

  为OpenCLRuntimeWrapper类的实例，在算子内部可通过该对象调取MindSpore Lite提供的OpenCL操作相关接口[mindspore::registry::opencl](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry_opencl.html)。

- fp16_enable_

  为算子是否使用FP16进行运算的标志。若要使用FP16进行运算，需将算子注册为FP16算子。本例中注册的是FP32算子。

- weight_ptrs_

  保存算子运算所需的临时内存的指针。

- 其他变量

  其他变量为进行OpenCL操作时所需的变量，详细意义可查看OpenCL操作时对应的接口说明[mindspore::registry::opencl](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry_opencl.html)。

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

#### Prepare实现代码与说明

在图编译阶段`mindspore::Model::Build`，将调用算子的Prepare实现。用户可以在这里进行一些较为耗时的一次性操作，以节约`mindspore::Model::Predict`时算子计算的时间。
在该样例中，通过重载Prepare接口，实现对自定义的OpenCL代码进行加载并编译。

1. 检验环境

    样例中，首先通过调用`CheckSpecs`，对算子的运行环境进行检查。
    此处，在`CheckSpecs`中，检查了输入和输出的数据类型，及输入和输出的tensor数量。
    通过`MSTensor::IsConst()`接口可以判断一个tensor的数据是否为常量，此处对非常量输入的数据类型，和算子注册时所声明处理的数据类型也进行了对比校验。对于常量数据的处理，参考本章后续的教程。

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

2. 加载自定义的OpenCL代码

    通过`opencl_runtime_`调用`OpenCLRuntimeWrapper::LoadSource`接口加载自定义的OpenCL代码。

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

    `arithmetic_source`中为用户自定义的OpenCL代码，如下所示：

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

3. 编译OpenCL代码

    通过`fp16_enable_`指定不同的编译选项，以生成处理float16或float32数据的代码。
    通过`opencl_runtime_`调用`OpenCLRuntimeWrapper::BuildKernel`接口，得到编译后的`cl::Kernel`变量，保存在`kernel_`。

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

4. 设置OpenCL工作组和工作项

    对注册为GPU的算子来说，除输入为常量的情况，所接收到的是Image格式的输入数据，Format为NHWC4（指C轴4字节对齐的NHWC格式数据）。
    本例中也将所有数据转为这种格式进行计算和输出。
    例程中实现的是一个简单的加法自定义算子，所以这里直接通过`GpuTensorInfo`函数计算输出数据`Image`内存所用宽度和高度来设置工作项。

    ```cpp
    int Prepare() override {
        ...
        auto out_shape = GpuTensorInfo(&outputs_[0], &opencl_runtime_);
        local_range_ = cl::NullRange;
        global_range_ = cl::NDRange(out_shape.width, out_shape.height);
        ...
    }
    ```

    `GpuTensorInfo`的实现如下，首先通过`Broadcast2GpuShape`函数将tensor的shape转为四维，然后计算Format为NHWC4时的shape值。
    再接着通过`OpenCLRuntimeWrapper::GetMaxImage2DWidth`及`OpenCLRuntimeWrapper::GetMaxImage2DHeight`接口得到Image内存所支持的最大宽度和高度，以此确定算子实际使用的Image内存宽度和高度。

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

    `Broadcast2GpuShape`的实现如下所示：

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

5. 将常量输入转为合适格式的数据，并分配GPU内存

    对注册为GPU的算子来说，除输入为常量的情况，其他情况下，输入数据已经为Image格式的GPU内存数据。
    为满足算子运算所需，用户需为常量输入设置合适的格式，必要时为其分配GPU内存。在此例，针对常量tensor的操作如下所示。

    首先通过`MSTensor::IsConst()`接口判断输入是否为常量，并通过`GpuTensorInfo`计算转为Image格式时所需的内存大小。
    然后分配该大小的局部内存`weight`，并通过`PackNHWCToNHWC4`函数将tensor内存转到`weight`中存储。

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

    `PackNHWCToNHWC4`函数实现如下，其中包含了对float16和float32类型的转换。

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

    通过`OpenCLRuntimeWrapper::GetAllocator`得到分配GPU内存的内存分配器。
    然后通过分配器的`mindspore::Allocator::Malloc`接口，可以申请到Image格式的GPU内存。
    接着通过`OpenCLRuntimeWrapper::WriteImage(void *buffer, void *src_data)`接口，将已经转为NHWC4格式的`weight`数据写入到GPU内存中。
    申请的GPU内存指针保存到weight_ptrs_中，以便在析构时释放。

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

    析构时调用的释放GPU内存函数如下，通过`OpenCLRuntimeWrapper::GetAllocator`得到分配GPU内存的内存分配器。
    然后通过分配器的`mindspore::Allocator::Free`接口，可以释放申请到的GPU内存。

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

6. 设置OpenCL内核运行时参数的值

    某些OpenCL内核运行时不会改变的参数，可以在`Prepare`阶段进行设置。
    在此例中，通过`OpenCLRuntimeWrapper::SetKernelArg`，设置`ElementAdd`运行时的第三个参数（计算的范围）。

    ```cpp
    int arg_idx = 3;
    cl_int2 output_shape{static_cast<int>(global_range_[0]), static_cast<int>(global_range_[1])};
    if (opencl_runtime_.SetKernelArg(kernel_, arg_idx, output_shape) != kSuccess) {
        std::cerr << "Set kernel arg" << arg_idx << "failed.";
        FreeWeight();
        return kLiteError;
    }
    ```

#### ReSize及Execute实现代码与说明

通过重载实现`Execute`可以实现推理时算子的自定义运算操作。

1. 调用`ReSize`函数，以支持运行时shape变更

    在本例中，首先调用`PreProcess`来处理运算前的一些准备工作。
    在`PreProcess()`中，首先调用`ReSize`函数，该函数为需要用户重载实现的运行时shape变更适配接口。
    在`ReSize`函数中，通过调用`CheckOutputs`判断算子的输出tensor的shape是否存在非法值，以判断是否需要重新进行shape推理。若不需要，直接返回。
    在需要进行shape推理时，通过`registry::RegisterKernelInterface::GetKernelInterface`获得该算子所注册的shape推理函数，此处得到的就是本例程中用户实现并注册的`InferShape`函数。
    在重新推理之后，通过调用之前实现的`Prepare`接口，重新申请和分配算子运算时需要的内存及相关变量。

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

2. 为输出tensor申请内存分配

    在算子运行前，需要为输出tensor申请分配GPU内存，由于框架的限制，该GPU内存需要托管给框架管理，不可人为释放。具体操作流程如下：
    1. 通过调用输出tensor的`allocator()`接口，可以得到框架中管理这个tensor的内存分配器，在GPU注册算子中，则为负责分配GPU内存的内存分配器。
    2. 计算需要分配的内存大小，此例中通过`GpuTensorInfo`函数来计算。
    3. 通过内存分配器的`Malloc`接口申请内存，用户可分别通过`void *Malloc(size_t weight, size_t height, DataType type)`和`void *Malloc(size_t size)`接口得到Image或Buffer格式的内存。
    4. 通过`SetData`接口，将申请的内存赋值给tensor，此后，此内存将由框架统一管理，用户不可手动释放。

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

3. 运行OpenCL内核

    通过`SetKernelArg`接口设置OpenCL的Kernel运行时的参数，通过`RunKernel`运行OpenCL的Kernel。

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
