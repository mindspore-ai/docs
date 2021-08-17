# mindspore::converter

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_cpp/mindspore_converter.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

以下描述了Mindspore Lite转换支持的模型类型及用户扩展所需的必要信息。

## FmkType

\#include <[parser_context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/parser_context.h)>

 **enum**类型变量，定义MindSpore Lite转换支持的框架类型。

| 类型定义 | 值 | 描述 |
| --- | --- | --- |
|kFmkTypeTf| 0 | 表示tensorflow框架。 |
|kFmkTypeCaffe| 1 | 表示caffe框架。 |
|kFmkTypeOnnx| 2 | 表示onnx框架。 |
|kFmkTypeMs| 3 | 表示mindspore框架。 |
|kFmkTypeTflite| 4 | 表示tflite框架。 |

## ConverterParameters

\#include <[parser_context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/parser_context.h)>

**struct**类型结构体，定义模型解析时的转换参数，用于模型解析时的只读参数。

```c++
struct ConverterParameters {
  FmkType fmk;                                   // 框架类型
  schema::QuantType quant_type;                  // 模型量化类型
  std::string model_file;                        // 原始模型文件路径
  std::string weight_file;                       // 原始模型权重文件路径，仅在Caffe框架下有效
  std::map<std::string, std::string> attrs;      // 预留参数接口，暂未启用
};
```

## ModelParser

\#include <[parser_context.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/registry/parser_context.h)>

ModelParser类的前置声明，定义了解析原始模型的基类。

```c++
class ModelParser;
```
