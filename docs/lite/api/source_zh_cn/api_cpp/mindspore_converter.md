# mindspore::converter

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_cpp/mindspore_converter.md)

以下描述了MindSpore Lite转换支持的模型类型及用户扩展所需的必要信息。

## 接口汇总

| 类名 | 描述 |
| --- | --- |
| [FmkType](#fmktype) | MindSpore Lite支持的框架类型。|
| [ConverterParameters](#converterparameters) | 模型解析时的只读参数。|
| [ConverterContext](#convertercontext) | 模型转换时的基本信息设置与获取。|
| [NodeParser](#nodeparser) | op节点的解析基类。|
| [ModelParser](#modelparser) | 模型解析的基类。|

## FmkType

\#include <[converter_context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/converter_context.h)>

 **enum**类型变量，定义MindSpore Lite转换支持的框架类型。

| 类型定义 | 值 | 描述 |
| --- | -- | --- |
|kFmkTypeTf| 0 | 表示tensorflow框架。 |
|kFmkTypeCaffe| 1 | 表示caffe框架。 |
|kFmkTypeOnnx| 2 | 表示onnx框架。 |
|kFmkTypeMs| 3 | 表示mindspore框架。 |
|kFmkTypeTflite| 4 | 表示tflite框架。 |
|kFmkTypePytorch| 5 | 表示pytorch框架。 |

## ConverterParameters

\#include <[converter_context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/converter_context.h)>

**struct**类型结构体，定义模型解析时的转换参数，用于模型解析时的只读参数。

```c++
struct ConverterParameters {
  FmkType fmk;                                   // 框架类型
  std::string model_file;                        // 原始模型文件路径
  std::string weight_file;                       // 原始模型权重文件路径，仅在Caffe框架下有效
  std::map<std::string, std::string> attrs;      // 预留参数接口，暂未启用
};
```

## ConverterContext

\#include <[converter_context.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/converter_context.h)>

模型转换过程中，基本信息的设置与获取。

### ConverterContext

```c++
ConverterContext() = default;
```

构造函数。

### ~ConverterContext

```c++
~ConverterContext() = default;
```

析构函数。

### 公有成员函数

#### SetGraphOutputTensorNames

```c++
static void SetGraphOutputTensorNames(const std::vector<std::string> &output_names);
```

静态方法，设置导出模型的输出名称。

- 参数

    - `output_names`: 模型的输出名称。

#### GetGraphOutputTensorNames

```c++
static std::vector<std::string> GetGraphOutputTensorNames();
```

静态方法，获取模型的输出名称。

- 返回值

    模型的输出名称，默认与原始模型的输出名称一致。

## NodeParser

\#include <[node_parser.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/node_parser.h)>

op节点的解析基类。

### NodeParser

```c++
NodeParser() = default;
```

构造函数。

### ~NodeParser

```c++
virtual ~NodeParser() = default;
```

析构函数。

### 公有成员函数

#### Parse

```c++
ops::PrimitiveC *Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node);
```

onnx节点解析接口函数。

- 参数

    - `onnx_graph`: 模型结构，包含模型的所有信息。

    - `onnx_node`: 待解析节点。

- 返回值

    PrimitiveC类指针对象，存储节点属性。

#### Parse

```c++
ops::PrimitiveC *Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight);
```

caffe节点解析接口函数。

- 参数

    - `proto`: 待解析节点，包含节点的属性信息。

    - `weight`: 待解析节点的权重信息。

- 返回值

    PrimitiveC类指针对象，存储节点属性。

#### Parse

```c++
ops::PrimitiveC *Parse(const tensorflow::NodeDef &tf_op,
                       const std::map<std::string, const tensorflow::NodeDef *> &tf_node_map,
                       std::vector<std::string> *inputs, int *output_size);
```

tf节点解析接口函数。

- 参数

    - `tf_op`: 待解析节点。

    - `tf_node_map`: 模型的所有节点信息。

    - `inputs`: 用户指定当前节点需要哪些原始输入，及其解析后的输入顺序。

    - `output_size`: 用户指定当前节点的输出个数。

- 返回值

    PrimitiveC类指针对象，存储节点属性。

#### Parse

```c++
ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                       const std::unique_ptr<tflite::ModelT> &tflite_model);
```

tflite节点解析接口函数。

- 参数

    - `tflite_op`: 待解析节点，包含节点的属性信息。

    - `tflite_model`: 模型结构，包含模型的所有信息。

- 返回值

    PrimitiveC类指针对象，存储节点属性。

## NodeParserPtr

\#include <[node_parser.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/node_parser.h)>

NodeParser类的共享智能指针类型。

```c++
using NodeParserPtr = std::shared_ptr<NodeParser>;
```

## ModelParser

\#include <[model_parser.h](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/include/registry/model_parser.h)>

解析原始模型的基类。

### ModelParser

```c++
ModelParser() = default;
```

构造函数。

### ~ModelParser

```c++
virtual ~ModelParser() = default;
```

析构函数。

### 公有成员函数

#### Parse

```c++
api::FuncGraphPtr Parse(const converter::ConverterParameters &flags);
```

模型解析接口。

- 参数

    - `flags`: 解析模型时基本信息，具体见[ConverterParameters](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#converterparameters)。

- 返回值

    FuncGraph的共享智能指针。

### 保护数据成员

#### res_graph_

```c++
api::FuncGraphPtr res_graph_ = nullptr;
```

FuncGraph的共享智能指针。
