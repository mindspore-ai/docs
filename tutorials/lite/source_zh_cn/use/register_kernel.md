# 自定义南向算子

`Windows` `Linux` `Android` `C++` `推理应用` `高级`

## 概述

MindSpore Lite当前提供了一套南向算子的注册机制，南向算子可以理解为用户自己的算子实现，如果用户想通过MindSpore Lite框架调度到自己的算子实现上，可参考本文。

实现南向算子大概有以下几个步骤：

1. 确定算子类型 ：分为通用算子与Custom算子。
2. 算子实现：继承Kernel类实现自有算子。
3. 算子注册：把自有算子注册进MindSpore Lite。
4. 算子InferShape：继承mindspore::kernel::KernelInteface实现自有算子的InferShape能力。

## 确定算子类型

查看mindspore/lite/schema/ops.fbs中的算子原型定义，确认要注册实现的算子原型是否在PrimitiveType中有定义，有定义的话则要注册的算子为通用算子，可以按照已有的IR直接实现算子与注册，否则即为Custom算子。

## 通用算子

整个算子的实现、注册、infershape等相关的代码可以参看代码仓里的[样例](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/test/ut/src/registry/registry_test.cc)。

### 通用算子实现

继承[mindspore::kernel::Kernel](https://mindspore.cn/doc/api_cpp/zh-CN/master/kernel.html)，重载实现必要的接口。

#### 样例代码与说明

以自定义一个Add算子为例：

1. 算子继承Kernel。
2. PreProcess()对内存进行了预分配。
3. Execute()对input进行了相加。

``` c++
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
  float *in0 = static_cast<float *>(inputs_[0]->data());
  float *in1 = static_cast<float *>(inputs_[1]->data());
  float *out = static_cast<float *>(outputs_[0]->data());
  auto num = outputs_[0]->ElementsNum();
  for (int i = 0; i < num; ++i) {
    out[i] = in0[i] + in1[i];
  }
  return RET_OK;
}
```

### 通用算子注册

当前有提供现成的宏[REGISTER_KERNEL](https://mindspore.cn/doc/api_cpp/zh-CN/master/registry.html#REGISTER_KERNEL)可以进行算子注册，用户也可以仿照宏内对应的代码去调用具体的接口。

#### 样例代码与说明

1. 函数TestCustomAddCreator用来创建Kernel。
2. 通过宏REGISTER_KERNEL进行Kernel注册，这里产商假定为BuiltInTest。

```c++
using mindspore::schema::PrimitiveType_AddFusion;

std::shared_ptr<Kernel> TestCustomAddCreator(const std::vector<tensor::MSTensor *> &inputs,
                                             const std::vector<tensor::MSTensor *> &outputs,
                                             const schema::Primitive *primitive, const lite::Context *ctx) {
  return std::make_shared<TestCustomAdd>(inputs, outputs, primitive, ctx);
}

REGISTER_KERNEL(CPU, BuiltInTest, kNumberTypeFloat32, PrimitiveType_AddFusion, TestCustomAddCreator)
```

### 通用算子InferShape

1. 继承[KernelInterface](https://mindspore.cn/doc/api_cpp/zh-CN/master/registry.html#KernelInterface)。
2. 重载实现Infer函数，推导出output tensor的shape，format，data_type。
3. 注册自定义的KernelInterface，可以使用注册宏[REGISTER_KERNEL_INTERFACE](https://mindspore.cn/doc/api_cpp/zh-CN/master/registry.html#REGISTER_KERNEL_INTERFACE)。

#### 样例代码与说明

这里以自定义Add算子为例：

1. 继承KernelInterface后重载Infer函数，实现InferShape能力。
2. 通过宏REGISTER_KERNEL_INTERFACE实现能力注册。

```c++
using mindspore::kernel::KernelInterface;

class TestCustomAddInfer : public KernelInterface {
 public:
  TestCustomAddInfer() = default;
  ~TestCustomAddInfer() = default;
  int Infer(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
            const schema::Primitive *primitive) override {
    outputs[0]->set_format(inputs[0]->format());
    outputs[0]->set_data_type(inputs[0]->data_type());
    outputs[0]->set_shape(inputs[0]->shape());
    return RET_OK;
  }
};

std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomAddInfer>(); }

REGISTER_KERNEL_INTERFACE(BuiltInTest, PrimitiveType_AddFusion, CustomAddInferCreator)
```

## Custom算子

整个Custom算子的实现、注册、infershape等相关的代码可以参看代码仓里的[样例](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/test/ut/src/registry/registry_custom_op_test.cc)。

### Custom 算子定义

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

### Custom算子实现

Custom算子的实现整体流程与通用算子的实现是一致的，因为都是[Kernel](https://mindspore.cn/doc/api_cpp/zh-CN/master/kernel.html#Kernel)的具体子类。
如果自定义算子不是运行在CPU平台上，那样需要在运行结束时把结果重新拷回output tensor。

#### 样例代码与说明

这里以创建一个Add能力的Custom算子为例：

1. 算子继承Kernel。
2. PreProcess()对内存进行了预分配。
3. Execute()对input进行了相加。

``` c++
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
  float *in0 = static_cast<float *>(inputs_[0]->data());
  float *in1 = static_cast<float *>(inputs_[1]->data());
  float *out = static_cast<float *>(outputs_[0]->data());
  auto num = outputs_[0]->ElementsNum();
  for (int i = 0; i < num; ++i) {
    out[i] = in0[i] + in1[i];
  }
  return RET_OK;
}
```

#### Custom算子属性解码样例

样例中是把属性里的字节流复制到了buf内。

``` c++
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

当前有提供的现成的宏[REGISTER_CUSTOM_KERNEL](https://mindspore.cn/doc/api_cpp/zh-CN/master/registry.html#REGISTER_CUSTOM_KERNEL)可以进行算子注册。

#### 样例代码与说明

1. TestCustomAddCreator函数用来创建Kernel。
2. 通过宏REGISTER_CUSTOM_KERNEL进行算子注册，这里假定产商为BuiltInTest，算子类型为Add。

```c++
using mindspore::schema::PrimitiveType_AddFusion;

std::shared_ptr<Kernel> TestCustomAddCreator(const std::vector<tensor::MSTensor *> &inputs,
                                             const std::vector<tensor::MSTensor *> &outputs,
                                             const schema::Primitive *primitive, const lite::Context *ctx) {
  return std::make_shared<TestCustomOp>(inputs, outputs, primitive, ctx);
}

REGISTER_CUSTOM_KERNEL(CPU, BuiltInTest, kNumberTypeFloat32, Add, TestCustomAddCreator)
```

### Custom算子InferShape

整体实现与通用算子的InferShape差不多，主要的差异在注册上。
Custom算子的InferShape采用宏[REGISTER_CUSTOM_KERNEL_INTERFACE](https://mindspore.cn/doc/api_cpp/zh-CN/master/registry.html#REGISTER_CUSTOM_KERNEL_INTERFACE)进行注册。

#### 样例代码与说明

1. CustomAddInferCreator函数用于创建自定义的KernelInterface。
2. 通过宏 REGISTER_CUSTOM_KERNEL_INTERFACE注册InferShape能力，这里的算子类型Add必须与REGISTER_CUSTOM_KERNEL时的算子类型一致。

```c++
class TestCustomOpInfer : public KernelInterface {
 public:
  TestCustomOpInfer() = default;
  ~TestCustomOpInfer() = default;
  int Infer(const std::vector<tensor::MSTensor *> &inputs, const std::vector<tensor::MSTensor *> &outputs,
            const schema::Primitive *primitive) override {
    outputs[0]->set_format(inputs[0]->format());
    outputs[0]->set_data_type(inputs[0]->data_type());
    outputs[0]->set_shape(inputs[0]->shape());
    return RET_OK;
  }
};

std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomOpInfer>(); }

REGISTER_CUSTOM_KERNEL_INTERFACE(BuiltInTest, Add, CustomAddInferCreator)
```
