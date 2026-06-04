# Class Serialization

\#include &lt;[serialization.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/serialization.h)&gt;

Serialization类汇总了模型文件读写的方法。

## 静态公有成员函数

| 函数                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------------------------|--------|--------|
| [static Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph)](#load)     |    ✕    |    √    |
| [static inline Status Load(const std::string &file, ModelType model_type, Graph *graph)](#load-1)     |    ✕    |    √    |
| [static inline Status Load(const std::vector\<std::string\> &files, ModelType model_type, std::vector\<Graph\> *graphs)](#load-2)     |    ✕    |    ✕    |
| [static inline Status SetParameters(const std::map\<std::string, Buffer\> &parameters, Model *model)](#setparameters)     |    ✕    |    ✕    |
| [static inline Status ExportModel(const Model &model, ModelType model_type, Buffer *model_data)](#exportmodel)     |    ✕    |    √    |
| [static inline Status ExportModel(const Model &model, ModelType model_type, const std::string &model_file, QuantizationType quantization_type = kNoQuant, bool export_inference_only = true, std::vector\<std::string\> output_tensor_name = {})](#exportmodel-1)     |    ✕    |    √    |
| [static inline Status ExportWeightsCollaborateWithMicro(const Model &model, ModelType model_type, const std::string &weight_file, bool is_inference = true, bool enable_fp16 = false, const std::vector\<std::string\> &changeable_weights_name = {})](#exportweightscollaboratewithmicro)     |    ✕    |    √    |

### Load

从内存缓冲区加载模型。

```cpp
static inline Status Load(const void *model_data, size_t data_size, ModelType model_type, Graph *graph)
```

- 参数

    - `model_data`：模型数据指针。
    - `data_size`：模型数据字节数。
    - `model_type`：模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kMindIR_Lite`、`ModelType::kOM`。
    - `graph`：输出参数，保存图数据的对象。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Load

从文件加载模型。

```cpp
static inline Status Load(const std::string &file, ModelType model_type, Graph *graph)
```

- 参数

    - `file`：模型文件路径。
    - `model_type`：模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kMindIR_Lite`、`ModelType::kOM`。
    - `graph`：输出参数，保存图数据的对象。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Load

从多个文件加载多个模型。

```cpp
static inline Status Load(const std::vector<std::string> &files, ModelType model_type, std::vector<Graph> *graphs)
```

- 参数

    - `files`：多个模型文件路径，用vector存储。
    - `model_type`：模型文件类型，可选有`ModelType::kMindIR`、`ModelType::kMindIR_Lite`、`ModelType::kOM`。
    - `graphs`：输出参数，依次保存图数据的对象。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### SetParameters

配置模型参数。

```cpp
static inline Status SetParameters(const std::map<std::string, Buffer> &parameters, Model *model)
```

- 参数

    - `parameters`：参数。
    - `model`：模型。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### ExportModel

导出训练模型，MindSpore Lite训练使用。

```cpp
static inline Status ExportModel(const Model &model, ModelType model_type, Buffer *model_data)
```

- 参数

    - `model`：模型数据。
    - `model_type`：模型文件类型。
    - `model_data`：模型参数数据。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### ExportModel

导出训练模型，MindSpore Lite训练使用。

```cpp
static inline Status ExportModel(const Model &model, ModelType model_type, const std::string &model_file,
                        QuantizationType quantization_type = kNoQuant, bool export_inference_only = true,
                        std::vector<std::string> output_tensor_name = {})
```

- 参数

    - `model`：模型数据。
    - `model_type`：模型文件类型。
    - `model_file`：保存的模型文件。
    - `quantization_type`: 量化类型。
    - `export_inference_only`: 是否导出只做推理的模型。
    - `output_tensor_name`: 设置导出的推理模型的输出张量的名称，默认为空，导出完整的推理模型。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### ExportWeightsCollaborateWithMicro

试验接口，导出供micro推理使用的模型权重，MindSpore Lite训练使用。

```cpp
static inline Status ExportWeightsCollaborateWithMicro(const Model &model, ModelType model_type,
                                                const std::string &weight_file, bool is_inference = true,
                                                bool enable_fp16 = false,
                                                const std::vector<std::string> &changeable_weights_name = {})
```

- 参数

    - `model`：模型数据。
    - `model_type`：模型文件类型。
    - `weight_file`：保存的权重文件。
    - `is_inference`: 是否是对推理模型的导出，当前仅支持推理模型。
    - `enable_fp16`: 权重保存类型。
    - `changeable_weights_name`: 设置shape会变化的权重名称。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。
