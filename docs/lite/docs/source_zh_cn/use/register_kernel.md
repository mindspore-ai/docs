# 自定义南向算子

`Windows` `Linux` `Android` `C++` `推理应用` `高级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.5/docs/lite/docs/source_zh_cn/use/register_kernel.md)

## 概述

MindSpore Lite当前提供了一套南向算子的注册机制，南向算子可以理解为用户自己的算子实现，如果用户想通过MindSpore Lite框架调度到自己的算子实现上，可参考本文。

实现南向算子大概有以下几个步骤：

1. 确定算子类型 ：分为通用算子与Custom算子。
2. 算子实现：继承Kernel类实现自有算子。
3. 算子注册：把自有算子注册进MindSpore Lite。
4. 算子InferShape：继承mindspore::kernel::KernelInteface实现自有算子的InferShape能力。
5. 算子InferShape注册：把自有算子的InferShape功能注册进MindSpore Lite。

## 确定算子类型

查看mindspore/lite/schema/ops.fbs中的算子原型定义，确认要注册实现的算子原型是否在PrimitiveType中有定义，有定义的话则要注册的算子为通用算子，可以按照已有的IR直接实现算子与注册，否则即为Custom算子。

## 通用算子

整个算子的实现、注册、infershape等相关的代码可以参看代码仓里的[样例](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/test/ut/src/registry/registry_test.cc)。

### 通用算子实现

继承[mindspore::kernel::Kernel](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_kernel.html)，重载实现必要的接口。

#### 样例代码与说明

以自定义一个Add算子为例：

1. 算子继承Kernel。
2. PreProcess()对内存进行了预分配。
3. Execute()对input进行了相加。

``` cpp
using mindspore::kernel::Kernel;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;

class TestCustomAdd : public Kernel {
 public:
  TestCustomAdd(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
                const schema::Primitive *primitive, const lite::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}
  int Prepare() override { return 0; }

  int Execute() override;

  int ReSize() { return 0; }

 private:
  int PreProcess() {
    for (auto *output : outputs_) {
      // malloc data for output tensor
      auto data = output->MutableData();
      if (data == nullptr) {
        MS_LOG(ERROR) << "Get data failed";
        return RET_ERROR;
      }
    }
    return RET_OK;
  }
};

int TestCustomAdd::Execute() {
  if (inputs_.size() != 2) {
    return RET_PARAM_INVALID;
  }
  PreProcess();
  auto *in0 = static_cast<const float *>(inputs_[0].Data().get());
  auto *in1 = static_cast<const float *>(inputs_[1].Data().get());
  float *out = static_cast<float *>(outputs_[0].MutableData());
  auto num = outputs_[0].ElementNum();
  for (int i = 0; i < num; ++i) {
    out[i] = in0[i] + in1[i];
  }
  return RET_OK;
}
```

### 通用算子注册

当前有提供现成的宏[REGISTER_KERNEL](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_registry.html#register-kernel)可以进行算子注册，用户也可以仿照宏内对应的代码去调用具体的接口。

#### 样例代码与说明

1. 函数TestCustomAddCreator用来创建Kernel。
2. 通过宏REGISTER_KERNEL进行Kernel注册，这里产商假定为BuiltInTest。

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

### 通用算子InferShape

1. 继承[KernelInterface](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_kernel.html#kernelinterface)。
2. 重载实现Infer函数，推导出output tensor的shape，format，data_type。

#### 样例代码与说明

这里以自定义Add算子为例：

继承KernelInterface后重载Infer函数，实现InferShape能力。

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

### 通用算子InferShape注册

当前有提供现成的宏[REGISTER_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_registry.html#register-kernel-interface)可以进行算子InferShape注册，用户也可以仿照宏内对应的代码去调用具体的接口。

#### 样例代码与说明

1. 函数CustomAddInferCreator用来创建KernelInterface实例。
2. 调用REGISTER_KERNEL_INTERFACE宏对通用算子InferShape进行注册，这里产商假定为BuiltInTest。

```cpp
std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomAddInfer>(); }

REGISTER_KERNEL_INTERFACE(BuiltInTest, PrimitiveType_AddFusion, CustomAddInferCreator)
```

## Custom算子

Custom算子的解析、创建、操作等相关的代码可以参看代码仓里的[样例](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/test/ut/tools/converter/registry/pass_registry_test.cc)。

### Custom算子定义

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

这里以AddN算子转为一个Custom算子为例：

1. 设Custom算子存在“input_num”、“op_kind”属性。
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

整个Custom算子的实现、注册、infershape等相关的代码可以参看代码仓里的[样例](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/test/ut/src/registry/registry_custom_op_test.cc)。

### Custom算子实现

Custom算子的实现整体流程与通用算子的实现是一致的，因为都是[Kernel](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_kernel.html)的具体子类。
如果自定义算子不是运行在CPU平台上，那样需要在运行结束时把结果重新拷回output tensor。

#### 样例代码与说明

这里以创建一个Add能力的Custom算子为例：

1. 算子继承Kernel。
2. PreProcess()对内存进行了预分配。
3. Execute()对input进行了相加。

``` cpp
using mindspore::kernel::Kernel;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

class TestCustomOp : public Kernel {
 public:
  TestCustomOp(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
               const schema::Primitive *primitive, const lite::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}
  int Prepare() override { return 0; }

  int Execute() override;

  int ReSize() override { return 0; }

 private:
  int PreProcess() {
    for (auto *output : outputs_) {
      // malloc data for output tensor
      auto data = output->MutableData();
      if (data == nullptr) {
        MS_LOG(ERROR) << "Get data failed";
        return RET_ERROR;
      }
    }
    return RET_OK;
  }

int TestCustomOp::Execute() {
  if (inputs_.size() != 2) {
    return RET_PARAM_INVALID;
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
  return RET_OK;
}
```

#### Custom算子属性解码样例

样例中是把属性里的字节流复制到了buf内。

``` cpp
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

### Custom算子注册

当前有提供的现成的宏[REGISTER_CUSTOM_KERNEL](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_registry.html#register-custom-kernel)可以进行算子注册。

#### 样例代码与说明

1. TestCustomAddCreator函数用来创建Kernel。
2. 通过宏REGISTER_CUSTOM_KERNEL进行算子注册，这里假定产商为BuiltInTest，算子类型为Add。

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

### Custom算子InferShape

整体实现与通用算子InferShape是一样的。

#### 样例代码与说明

1. 继承[KernelInterface](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_kernel.html#kernelinterface)。
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

### Custom算子InferShape注册

当前有提供的现成的宏[REGISTER_CUSTOM_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_registry.html#register-custom-kernel-interface)可以进行Custom算子InferShape的注册。

#### 样例代码与说明

1. CustomAddInferCreator函数用于创建自定义的KernelInterface。
2. 通过宏[REGISTER_CUSTOM_KERNEL_INTERFACE](https://www.mindspore.cn/lite/api/zh-CN/r1.5/api_cpp/mindspore_registry.html#register-custom-kernel-interface)注册InferShape能力，这里的算子类型Add必须与REGISTER_CUSTOM_KERNEL时的算子类型一致。

```cpp
std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomOpInfer>(); }

REGISTER_CUSTOM_KERNEL_INTERFACE(BuiltInTest, Add, CustomAddInferCreator)
```
