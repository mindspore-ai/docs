# Class Model

\#include &lt;[model.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/model.h)&gt;

Model定义了MindSpore中的模型，便于计算图管理。

## 构造函数

```cpp
Model()
```

## 析构函数

```cpp
~Model()
```

## 公有成员函数

| 函数                                                                                                                                                                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------|
| [Status Build(const void *model_data, size_t data_size, ModelType model_type, const std::shared_ptr\<Context\> &model_context = nullptr)](#build)     |    √    |    √    |
| [Status Build(const void *model_data, size_t data_size, const void *weight_data, size_t weight_size, ModelType model_type, const std::shared_ptr\<Context\> &model_context = nullptr)](#build-1)     |    √    |    ✕    |
| [inline Status Build(const std::string &model_path, ModelType model_type, const std::shared_ptr\<Context\> &model_context = nullptr)](#build-2)     |    √    |    √    |
| [inline Status Build(const void *model_data, size_t data_size, ModelType model_type, const std::shared_ptr\<Context\> &model_context, const Key &dec_key, const std::string &dec_mode, const std::string &cropto_lib_path)](#build-3)     |    √    |    √    |
| [inline Status Build(const std::string &model_path, ModelType model_type, const std::shared_ptr\<Context\> &model_context, const Key &dec_key, const std::string &dec_mode, const std::string &cropto_lib_path)](#build-4)     |    √    |    √    |
| [Status Build(GraphCell graph, const std::shared_ptr\<Context\> &model_context = nullptr, const std::shared_ptr\<TrainCfg\> &train_cfg = nullptr)](#build-5)     |    ✕    |    √    |
| [Status BuildTransferLearning(GraphCell backbone, GraphCell head, const std::shared_ptr\<Context\> &context, const std::shared_ptr\<TrainCfg\> &train_cfg = nullptr)](#buildtransferlearning)     |    ✕    |    √    |
| [Status Resize(const std::vector\<MSTensor\> &inputs, const std::vector\<std::vector\<int64_t\>\> &dims)](#resize)     |    √    |    √    |
| [Status UpdateWeights(const std::vector\<MSTensor\> &new_weights)](#updateweights)     |    ✕    |    √    |
| [Status UpdateWeights(const std::vector\<std::vector\<MSTensor\>\> &new_weights)](#updateweights-1) | √    |  √     |
| [Status Predict(const std::vector\<MSTensor\> &inputs, std::vector\<MSTensor\> *outputs, const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#predict)     |    √    |    √    |
| [Status Predict(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#predict-1)     |    ✕    |    √    |
| [Status RunStep(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#runstep)     |    ✕    |    √    |
| [Status PredictWithPreprocess(const std::vector\<std::vector\<MSTensor\>\> &inputs, std::vector\<MSTensor\> *outputs, const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#predictwithpreprocess)     |    ✕    |    ✕    |
| [Status Preprocess(const std::vector\<std::vector\<MSTensor\>\> &inputs, std::vector\<MSTensor\> *outputs)](#preprocess)     |    ✕    |    ✕    |
| [bool HasPreprocess()](#haspreprocess)     |    ✕    |    ✕    |
| [inline Status LoadConfig(const std::string &config_path)](#loadconfig)     |    √    |    √    |
| [inline Status UpdateConfig(const std::string &section, const std::pair\<std::string, std::string\> &config)](#updateconfig)     |    √    |    √    |
| [std::vector\<MSTensor\> GetInputs()](#getinputs)     |    √    |    √    |
| [inline MSTensor GetInputByTensorName(const std::string &tensor_name)](#getinputbytensorname)     |    √    |    √    |
| [std::vector\<MSTensor\> GetGradients() const](#getgradients)     |    ✕    |    √    |
| [Status ApplyGradients(const std::vector\<MSTensor\> &gradients)](#applygradients)     |    ✕    |    √    |
| [std::vector\<MSTensor\> GetFeatureMaps() const](#getfeaturemaps)     |    ✕    |    √    |
| [std::vector\<MSTensor\> GetTrainableParams() const](#gettrainableparams)     |    ✕    |    √    |
| [Status UpdateFeatureMaps(const std::vector\<MSTensor\> &new_weights)](#updatefeaturemaps)     |    ✕    |    √    |
| [std::vector\<MSTensor\> GetOptimizerParams() const](#getoptimizerparams)     |    ✕    |    √    |
| [Status SetOptimizerParams(const std::vector\<MSTensor\> &params)](#setoptimizerparams)     |    ✕    |    √    |
| [Status SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f)](#setupvirtualbatch)     |    ✕    |    √    |
| [Status SetLearningRate(float learning_rate)](#setlearningrate)     |    ✕    |    √    |
| [float GetLearningRate()](#getlearningrate)     |    ✕    |    √    |
| [Status InitMetrics(std::vector\<Metrics *\> metrics)](#initmetrics)     |    ✕    |    √    |
| [std::vector\<Metrics *\> GetMetrics()](#getmetrics)     |    ✕    |    √    |
| [std::vector\<MSTensor\> GetOutputs()](#getoutputs)     |    √    |    √    |
| [inline std::vector\<std::string\> GetOutputTensorNames()](#getoutputtensornames)     |    √    |    √    |
| [inline MSTensor GetOutputByTensorName(const std::string &tensor_name)](#getoutputbytensorname)     |    √    |    √    |
| [inline std::vector\<MSTensor\> GetOutputsByNodeName(const std::string &node_name)](#getoutputsbynodename)     |    ✕    |    √    |
| [Status BindGLTexture2DMemory(const std::map\<std::string, unsigned int\> &inputGLTexture, std::map\<std::string, unsigned int\> *outputGLTexture)](#bindgltexture2dmemory)     |    ✕    |    √    |
| [static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type)](#checkmodelsupport)     |    √    |    √    |
| [Status SetTrainMode(bool train)](#settrainmode)     |    ✕    |    √    |
| [bool GetTrainMode() const](#gettrainmode)     |    ✕    |    √    |
| [const std::shared_ptr\<ModelImpl\> impl()](#impl) | √    |  √     |
| [inline std::string GetModelInfo(const std::string &key)](#getmodelinfo) | √    |  √     |
| [Status Finalize()](#finalize)     |  √ |  √ |

### Build

```cpp
Status Build(const void *model_data, size_t data_size, ModelType model_type,
             const std::shared_ptr<Context> &model_context = nullptr)
```

从内存缓冲区加载模型，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_data`: 指向存储读入模型文件缓冲区的指针。
    - `data_size`: 缓冲区大小。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云侧推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Build

```cpp
Status Build(const void *model_data, size_t data_size, const void *weight_data, size_t weight_size, ModelType model_type,
             const std::shared_ptr<Context> &model_context = nullptr)
```

从内存缓冲区加载模型和权重数据，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_data`: 指向存储读入模型文件缓冲区的指针。
    - `data_size`: 模型缓冲区大小。
    - `weight_data`: 指向存储读入权重文件缓冲区的指针。
    - `weight_size`: 权重缓冲区大小。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云侧推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Build

```cpp
inline Status Build(const std::string &model_path, ModelType model_type,
             const std::shared_ptr<Context> &model_context = nullptr)
```

根据路径读取加载模型，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_path`: 模型文件路径。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云侧推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Build

```cpp
inline Status Build(const void *model_data, size_t data_size, ModelType model_type,
             const std::shared_ptr<Context> &model_context, const Key &dec_key,
             const std::string &dec_mode, const std::string &cropto_lib_path)
```

从内存缓冲区加载模型，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_data`: 指向存储读入模型文件缓冲区的指针。
    - `data_size`: 缓冲区大小。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云侧推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。
    - `dec_key`: 解密密钥，用于解密密文模型，密钥长度为16。
    - `dec_mode`: 解密模式，可选有`AES-GCM`。
    - `cropto_lib_path`: OpenSSL Crypto解密库路径。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Build

```cpp
inline Status Build(const std::string &model_path, ModelType model_type,
             const std::shared_ptr<Context> &model_context, const Key &dec_key,
             const std::string &dec_mode, const std::string &cropto_lib_path)
```

根据路径读取加载模型，并将模型编译至可在Device上运行的状态。

- 参数

    - `model_path`: 模型文件路径。
    - `model_type`: 模型文件类型，可选有`ModelType::kMindIR_Lite`、`ModelType::kMindIR`，分别对应`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出）。在端侧和云侧推理包中，端侧推理只支持`ms`模型推理，该入参值被忽略。云侧推理支持`ms`和`mindir`模型推理，需要将该参数设置为模型对应的选项值。云侧推理对`ms`模型的支持，将在未来的迭代中删除，推荐通过`mindir`模型进行云侧推理。
    - `model_context`: 模型[Context](#context)。
    - `dec_key`: 解密密钥，用于解密密文模型，密钥长度为16。
    - `dec_mode`: 解密模式，可选有`AES-GCM`。
    - `cropto_lib_path`: OpenSSL Crypto解密库路径。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

> `Build`之后对`model_context`的其他修改不再生效。

### Build

```cpp
Status Build(GraphCell graph, const std::shared_ptr<Context> &model_context = nullptr,
             const std::shared_ptr<TrainCfg> &train_cfg = nullptr)
```

将GraphCell存储的模型编译至可在Device上运行的状态。

- 参数

    - `graph`: `GraphCell`是`Cell`的一个派生，`Cell`目前没有开放使用。`GraphCell`可以由`Graph`构造，如`model.Build(GraphCell(graph), context)`。
    - `model_context`: 模型[Context](#context)。
    - `train_cfg`: train配置文件[TrainCfg](#traincfg)。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### BuildTransferLearning

```cpp
Status BuildTransferLearning(GraphCell backbone, GraphCell head, const std::shared_ptr<Context> &context,
                      const std::shared_ptr<TrainCfg> &train_cfg = nullptr)
```

构建一个迁移学习模型，其中主干权重是固定的，头部权重是可训练的。

- 参数

    - `backbone`: 静态、不可学习部分。
    - `head`: 可训练部分。
    - `model_context`: 模型[Context](#context)。
    - `train_cfg`: train配置文件[TrainCfg](#traincfg)。

- 返回值

  状态码。

### Resize

```cpp
Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims)
```

调整已编译模型的输入张量形状。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `dims`: 输入张量形状，按输入顺序排列的由形状组成的`vector`，模型会按顺序依次调整对应输入顺序的`inputs`张量形状。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### UpdateWeights

```cpp
Status UpdateWeights(const std::vector<MSTensor> &new_weights)
```

更新模型的权重Tensor的大小和内容。

- 参数

    - `new_weights`: 要更新的权重Tensor，可同时更新大小和内容。

- 返回值

  状态码。

### UpdateWeights

```cpp
Status UpdateWeights(const std::vector<std::vector<MSTensor>> &new_weights)
```

更新模型的权重的大小和内容。

- 参数

    - `new_weights`: 要更新的权重Tensor，可同时更新大小和内容。

- 返回值

  状态码。

### Predict

```cpp
Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs, const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

推理模型。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。
    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Predict

```cpp
Status Predict(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

推理模型。

- 参数

    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### RunStep

```cpp
Status RunStep(const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

单步训练模型。

- 参数

    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

  状态码。

### PredictWithPreprocess

```cpp
Status PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs,
                             const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

进行推理模型，并在推理前进行数据预处理。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。
    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

  状态码。

### Preprocess

```cpp
Status Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs)
```

若模型配置了数据预处理，对模型输入数据进行数据预处理。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。

- 返回值

  状态码。

### HasPreprocess

```cpp
bool HasPreprocess()
```

模型是否配置了数据预处理。

- 返回值

  模型是否配置了数据预处理。

### LoadConfig

```cpp
inline Status LoadConfig(const std::string &config_path)
```

根据路径读取配置文件。

- 参数

    - `config_path`: 配置文件路径。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

> 用户可以调用`LoadConfig`接口进行混合精度推理的设置，配置文件举例如下：
>
> [execution_plan]
>
> op_name1=data_type:float16
>
> op_name2=data_type:float32

### UpdateConfig

```cpp
inline Status UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config)
```

刷新配置，读文件相对比较费时，如果少部分配置发生变化可以通过该接口更新部分配置。

- 参数

    - `section`: 配置的章节名。
    - `config`: 要更新的配置对。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### GetInputs

```cpp
std::vector<MSTensor> GetInputs()
```

获取模型所有输入张量。

- 返回值

  包含模型所有输入张量的容器类型变量。

### GetInputByTensorName

```cpp
inline MSTensor GetInputByTensorName(const std::string &tensor_name)
```

获取模型指定名字的输入张量。

- 返回值

  指定名字的输入张量，如果该名字不存在则返回非法张量。

### GetGradients

```cpp
std::vector<MSTensor> GetGradients() const
```

获取所有Tensor的梯度。

- 返回值

  获取所有Tensor的梯度。

### ApplyGradients

```cpp
Status ApplyGradients(const std::vector<MSTensor> &gradients)
```

应用所有Tensor的梯度。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### GetFeatureMaps

```cpp
std::vector<MSTensor> GetFeatureMaps() const
```

获取模型的所有权重Tensors。

- 返回值

  获取模型的所有权重Tensor。

### GetTrainableParams

```cpp
std::vector<MSTensor> GetTrainableParams() const
```

获取optimizer中所有参与权重更新的MSTensor。

- 返回值

  optimizer中所有参与权重更新的MSTensor。

### UpdateFeatureMaps

```cpp
Status UpdateFeatureMaps(const std::vector<MSTensor> &new_weights)
```

更新模型的权重Tensor内容。

- 参数

    - `new_weights`: 要更新的权重Tensor。

- 返回值

  状态码。

### GetOptimizerParams

```cpp
std::vector<MSTensor> GetOptimizerParams() const
```

获取optimizer参数MSTensor。

- 返回值

  所有optimizer参数MSTensor。

### SetOptimizerParams

```cpp
Status SetOptimizerParams(const std::vector<MSTensor> &params)
```

更新optimizer参数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### SetupVirtualBatch

```cpp
Status SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f)
```

设置虚拟batch用于训练。

- 参数

    - `virtual_batch_multiplier`: 虚拟batch乘法器，当设置值小于1时，表示禁用虚拟batch。
    - `lr`: 学习率，默认为-1.0f。
    - `momentum`: 动量，默认为-1.0f。

- 返回值

  状态码。

### SetLearningRate

```cpp
Status SetLearningRate(float learning_rate)
```

设置学习率。

- 参数

    - `learning_rate`: 指定的学习率。

- 返回值

  状态码。

### GetLearningRate

```cpp
float GetLearningRate()
```

获取学习率。

- 返回值

  float类型，获取学习率。如果为0.0，表示没有找到优化器。

### InitMetrics

```cpp
Status InitMetrics(std::vector<Metrics *> metrics)
```

训练指标参数初始化。

- 参数

    - `metrics`: 训练指标参数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### GetMetrics

```cpp
std::vector<Metrics *> GetMetrics()
```

获取训练指标参数。

- 返回值

  训练指标参数。

### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs()
```

获取模型所有输出张量。

- 返回值

  包含模型所有输出张量的容器类型变量。

### GetOutputTensorNames

```cpp
inline std::vector<std::string> GetOutputTensorNames()
```

获取模型所有输出张量的名字。

- 返回值

  包含模型所有输出张量名字的容器类型变量。

### GetOutputByTensorName

```cpp
inline MSTensor GetOutputByTensorName(const std::string &tensor_name)
```

获取模型指定名字的输出张量。

- 返回值

  指定名字的输出张量，如果该名字不存在则返回非法张量。

### GetOutputsByNodeName

```cpp
inline std::vector<MSTensor> GetOutputsByNodeName(const std::string &node_name)
```

通过节点名获取模型的MSTensors输出张量。不建议使用，将在2.0版本废弃。

- 参数

    - `node_name`: 节点名称。

- 返回值

  包含在模型输出Tensor中的该节点输出Tensor的vector。

### BindGLTexture2DMemory

```cpp
  Status BindGLTexture2DMemory(const std::map<std::string, unsigned int> &inputGLTexture,
                               std::map<std::string, unsigned int> *outputGLTexture)
```

将OpenGL纹理数据与模型的输入和输出进行绑定。

- 参数

    - `inputGLTexture`: 模型输入的OpenGL纹理数据, key为输入Tensor的名称，value为OpenGL纹理。
    - `outputGLTexture`: 模型输出的OpenGL纹理数据，key为输出Tensor的名称，value为OpenGL纹理。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### CheckModelSupport

```cpp
static bool CheckModelSupport(enum DeviceType device_type, ModelType model_type)
```

检查设备是否支持该模型。

- 参数

    - `device_type`: 设备类型，例如`kMaliGPU`。
    - `model_type`: 模型类型，例如`MindIR`。

- 返回值

  状态码。

### SetTrainMode

```cpp
Status SetTrainMode(bool train)
```

session设置训练模式。

- 参数

    - `train`: 是否为训练模式。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### GetTrainMode

```cpp
bool GetTrainMode() const
```

获取session是否是训练模式。

- 返回值

  bool类型，表示是否是训练模式。

### impl

```cpp
const std::shared_ptr<ModelImpl> impl() const
```

获得模型的实现。

- 返回值

  获得模型的实现。

### GetModelInfo

```cpp
inline std::string GetModelInfo(const std::string &key)
```

获取模型的信息。

- 参数

    - `key`: 模型的key。

- 返回值

  模型的信息。

### Finalize

```cpp
Status Finalize()
```

模型终止。

- 返回值

  状态码。
