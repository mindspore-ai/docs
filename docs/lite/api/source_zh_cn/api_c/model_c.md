# model_c

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0rc1/docs/lite/api/source_zh_cn/api_c/model_c.md)

```C
#include<model_c.h>
```

Model定义了MindSpore中编译和运行的模型。

## 公有函数和数据类型

| function                                                                                                                                                                                     |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [MSModelHandle MSModelCreate()](#msmodelcreate)                                                                                                                                              |
| [void MSModelDestroy(MSModelHandle* model)](#msmodeldestroy)                                                                                                                                 |
| [size_t MSModelCalcWorkspaceSize(MSModelHandle model)](#msmodelcalcworkspacesize)                                                                                                                                                                                             |
| [void MSModelSetWorkspace(MSModelHandle model, void* workspace, size_t workspace_size)](#msmodelsetworkspace)                                                                                |
| [MSStatus MSModelBuild(MSModelHandle model, const void* model_data, size_t data_size, MSModelType model_type, const MSContextHandle model_context)](#msmodelbuild)                           |
| [MSStatus MSModelBuildFromFile(MSModelHandle model, const char* model_path, MSModelType model_type,const MSContextHandle model_context)](#msmodelbuildfromfile)                              |
| [MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo* shape_infos,size_t shape_info_num)](#msmodelresize)                                              |
| [MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray* outputs,const MSKernelCallBackC before, const MSKernelCallBackC after)](#msmodelpredict) |
| [MSStatus MSModelRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after)](#msmodelrunstep)                                                               |
| [MSStatus MSModelSetTrainMode(const MSModelHandle model, bool train)](#msmodelsettrainmode)                                                                                                  |
| [MSStatus MSModelExportWeight(const MSModelHandle model, const char* export_path)](#msmodelexportweight)                                                                                     |
| [MSTensorHandleArray MSModelGetInputs(const MSModelHandle model)](#msmodelgetinputs)                                                                                                         |
| [MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model)](#msmodelgetoutputs)                                                                                                       |
| [MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char* tensor_name)](#msmodelgetinputbytensorname)                                                               |
| [MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model, const char* tensor_name)](#msmodelgetoutputbytensorname)                                                             |
| [MSTrainCfgHandle MSTrainCfgCreate()](#mstraincfgcreate)                                                                                                                                     |
| [void MSTrainCfgDestroy(MSTrainCfgHandle *train_cfg)](#mstraincfgdestroy)                                                                                                                    |
| [char **MSTrainCfgGetLossName(MSTrainCfgHandle train_cfg, size_t *num)](#mstraincfggetlossname)                                                                                              |
| [void MSTrainCfgSetLossName(MSTrainCfgHandle train_cfg, const char **loss_name, size_t num)](#mstraincfgsetlossname)                                                                         |
| [MSOptimizationLevel MSTrainCfgGetOptimizationLevel(MSTrainCfgHandle train_cfg)](#mstraincfggetoptimizationlevel)                                                                            |
| [void MSTrainCfgSetOptimizationLevel(MSTrainCfgHandle train_cfg, MSOptimizationLevel level)](#mstraincfgsetoptimizationlevel)                                                                |
| [void MSStatus MSTrainModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type,const MSContextHandle model_context, const MSTrainCfgHandle train_cfg)](#mstrainmodelbuild)                                                                                                                                                                           |
| [MSStatus MSTrainModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,const MSContextHandle model_context, const MSTrainCfgHandle train_cfg)](#mstrainmodelbuildfromfile)                                                                                                                                                                   |
| [MSStatus MSRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after)](#msrunstep)                                                                         |
| [MSStatus MSModelSetLearningRate(MSModelHandle model, float learning_rate)](#msmodelsetlearningrate)                                                                                         |
| [float MSModelGetLearningRate(MSModelHandle model)](#msmodelgetlearningrate)                                                                                                                 |
| [MSTensorHandleArray MSModelGetWeights(MSModelHandle model)](#msmodelgetweights)                                                                                                             |
| [MSStatus MSModelUpdateWeights(MSModelHandle model, const MSTensorHandleArray new_weights)](#msmodelupdateweights)                                                                           |
| [bool MSModelGetTrainMode(MSModelHandle model)](#msmodelgettrainmode)                                                                                                                        |
| [MSStatus MSModelSetTrainMode(MSModelHandle model, bool train)](#msmodelsettrainmode)                                                                                                        |
| [MSStatus MSModelSetupVirtualBatch(MSModelHandle model, int virtual_batch_multiplier, float lr, float momentum)](#msmodelsetupvirtualbatch)                                                  |
| [MSStatus MSExportModel(MSModelHandle model, MSModelType model_type, const char *model_file,MSQuantizationType quantization_type, bool export_inference_only,char **output_tensor_name, size_t num)](#msexportmodel)                                                                                                                                                                               |
| [MSStatus MSExportModelBuffer(MSModelHandle model, MSModelType model_type, char **model_data, size_t *data_size,MSQuantizationType quantization_type, bool export_inference_only,char **output_tensor_name, size_t num)](#msexportmodelbuffer)                                                                                                                                                                         |
| [MSStatus MSExportWeightsCollaborateWithMicro(MSModelHandle model, MSModelType model_type,const char *weight_file, bool is_inference, bool enable_fp16,char **changeable_weights_name, size_t num)](#msexportweightscollaboratewithmicro)                                                                                                                                                         |

### 定义

#### MS_MAX_SHAPE_NUM

```C
#define MS_MAX_SHAPE_NUM 32
```

MSTensor最大支持的维度为`MS_MAX_SHAPE_NUM`。

### 公有函数

#### MSModelCreate

```C
MSModelHandle MSModelCreate()
```

创建一个模型对象，该选项仅MindSpore Lite有效。

- 返回值

  模型对象指针。

#### MSModelDestroy

```C
void MSModelDestroy(MSModelHandle* model)
```

销毁一个模型对象，该选项仅MindSpore Lite有效。

- 参数

    - `model`:指向模型对象指针的指针。

#### MSModelCalcWorkspaceSize

```C
size_t MSModelCalcWorkspaceSize(MSModelHandle model)
```

计算模型工作时所需内存空间大小，该选项仅对IoT有效。(该接口未实现)

- 参数
    - `model`: 指向模型对象的指针。

#### MSModelSetWorkspace

```C
void MSModelSetWorkspace(MSModelHandle model, void* workspace, size_t workspace_size)
```

设置模型的工作空间，该选项仅对IoT有效。(该接口暂未实现)

- 参数
    - `model`: 指向模型对象的指针。
    - `workspace`: 指向工作空间的指针。
    - `workspace_size`: 工作空间大小。

#### MSModelBuild

```C
MSStatus MSModelBuild(MSModelHandle model, const void* model_data, size_t data_size, MSModelType model_type, const MSContextHandle model_context)
```

从内存缓冲区加载并编译MindSpore模型，该选项仅MindSpore Lite有效。

- 参数

    - `model`: 指向模型对象的指针。
    - `model_data`: 内存中已经加载的模型数据地址。
    - `data_size`: 模型数据的长度。
    - `model_type`: 模型文件类型，具体见: [MSModelType](https://mindspore.cn/lite/api/zh-CN/r2.7.0rc1/api_c/types_c.html#msmodeltype)。
    - `model_context`: 模型的上下文环境，具体见: [Context](https://mindspore.cn/lite/api/zh-CN/r2.7.0rc1/api_c/context_c.html)。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelBuildFromFile

```C
MSStatus MSModelBuildFromFile(MSModelHandle model, const char* model_path, MSModelType model_type,
                                     const MSContextHandle model_context)
```

通过模型文件中加载并编译MindSpore模型，该选项仅MindSpore Lite有效。

- 参数

    - `model`: 指向模型对象的指针。
    - `model_path`: 模型文件路径。
    - `model_type`: 模型文件类型，具体见: [MSModelType](https://mindspore.cn/lite/api/zh-CN/r2.7.0rc1/api_c/types_c.html#msmodeltype)。
    - `model_context`: 模型的上下文环境，具体见: [Context](https://mindspore.cn/lite/api/zh-CN/r2.7.0rc1/api_c/context_c.html)。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelResize

```C
MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo* shape_infos,
                              size_t shape_info_num)
```

调整已编译模型的输入形状。

- 参数

    - `model`: 指向模型对象的指针。
    - `inputs`: 模型输入对应的张量数组结构体。
    - `shape_infos`: 输入形状信息数组，按模型输入顺序排列的由形状信息组成的数组，模型会按顺序依次调整张量形状。
    - `shape_info_num`: 形状信息数组的长度。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelPredict

```C
MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray* outputs,
                               const MSKernelCallBackC before, const MSKernelCallBackC after)
```

执行模型推理。

- 参数

    - `model`: 指向模型对象的指针。
    - `inputs`: 模型输入对应的张量数组结构体。
    - `outputs`: 函数输出，模型输出对应的张量数组结构体的指针。
    - `before`: 模型推理前执行的回调函数。
    - `after`: 模型推理后执行的回调函数。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelRunStep

```C
MSStatus MSModelRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after)
```

逐步运行模型，该选项仅对IoT有效。(该接口目前仅在Micro中使用)

- 参数

    - `model`: 指向模型对象的指针。
    - `before`: 模型运行前执行的回调函数。
    - `after`: 模型运行后执行的回调函数。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelSetTrainMode

```C
MSStatus MSModelSetTrainMode(const MSModelHandle model, bool train)
```

设置模型运行模式，该选项仅对IoT有效。(该接口目前仅在Micro中使用)

- 参数

    - `model`: 指向模型对象的指针。
    - `train`: True表示模型在训练模式下运行，否则为推理模式。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelExportWeight

```C
MSStatus MSModelExportWeight(const MSModelHandle model, const char* export_path)
```

将模型权重导出到二进制文件，该选项仅对IoT有效。(该接口目前仅在Micro中使用)

- 参数

    - `model`: 指向模型对象的指针。
    - `export_path`: 导出权重文件的路径。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelGetInputs

```C
MSTensorHandleArray MSModelGetInputs(const MSModelHandle model)
```

获取模型的输入张量数组结构体。

- 参数

    - `model`: 指向模型对象的指针。

- 返回值

  模型输入对应的张量数组结构体。

#### MSModelGetOutputs

```C
MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model)
```

获取模型的输出张量数组结构体。

- 参数

    - `model`: 指向模型对象的指针。

- 返回值

  模型输出对应的张量数组结构体。

#### MSModelGetInputByTensorName

```C
MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model,
                                            const char* tensor_name)
```

通过张量名获取模型的输入张量。

- 参数

    - `model`: 指向模型对象的指针。
    - `tensor_name`: 张量名称。

- 返回值

  tensor_name所对应的输入张量的张量指针，如果输出中没有该张量则返回空。

#### MSModelGetOutputByTensorName

```C
MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model,
                                            const char* tensor_name)
```

通过张量名获取MindSpore模型的输出张量。

- 参数

    - `model`: 指向模型对象的指针。
    - `tensor_name`: 张量名。

- 返回值

  tensor_name所对应的张量指针。

#### MSTrainCfgCreate

```C
MSTrainCfgHandle MSTrainCfgCreate()
```

创建一个训练配置对象，仅适用于训练。

- 参数

    - 无。

- 返回值

  训练配置对象句柄。

#### MSTrainCfgDestroy

```C
void MSTrainCfgDestroy(MSTrainCfgHandle *train_cfg)
```

销毁一个TrainCfg对象，仅适用于训练。

- 参数

    - `train_cfg`: 指向训练配置对象的指针。

- 返回值

  无。

#### MSTrainCfgGetLossName

```C
char **MSTrainCfgGetLossName(MSTrainCfgHandle train_cfg, size_t *num)
```

获取训练配置中指定loss位置编号的loss名称，仅适用于训练。

- 参数

    - `train_cfg`: 指向训练配置对象句柄。
    - `num`: 需要获得loss名称的位置编号。

- 返回值

  loss名称。

#### MSTrainCfgSetLossName

```C
void MSTrainCfgSetLossName(MSTrainCfgHandle train_cfg, const char **loss_name, size_t num)
```

用于指定训练配置中指定loss位置编号的loss名称，仅适用于训练。

- 参数

    - `train_cfg`: 指向训练配置对象句柄。
    - `loss_name`: 用户需要定义loss的名称。
    - `num`: 用户需要定义loss的位置编号。

- 返回值

  无。

#### MSTrainCfgGetOptimizationLevel

```C
MSOptimizationLevel MSTrainCfgGetOptimizationLevel(MSTrainCfgHandle train_cfg)
```

用于从训练配置中获得优化级别，仅适用于训练。

- 参数

    - `train_cfg`: 指向训练配置对象句柄。

- 返回值

  训练优化级别的句柄。

#### MSTrainCfgSetOptimizationLevel

```C
void MSTrainCfgSetOptimizationLevel(MSTrainCfgHandle train_cfg, MSOptimizationLevel level)
```

用于指定训练配置中优化级别，仅适用于训练。

- 参数

    - `train_cfg`: 指向训练配置对象句柄。
    - `level`: 训练优化级别对象句柄。

- 返回值

  无。

#### MSTrainModelBuild

```C
MSStatus MSTrainModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type,
                                  const MSContextHandle model_context, const MSTrainCfgHandle train_cfg)
```

从模型缓冲区构建可在设备上运行的训练模型，仅适用于训练。

- 参数

    - `mode`: 模型对象句柄。
    - `model_data`: 模型文件读取的缓存。
    - `data_size`: 模型文件缓存的字节数。
    - `model_type`: 模型文件的类型。
    - `model_context`: 模型执行期间上下文。
    - `train_cfg`: 训练使用的配置。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSTrainModelBuildFromFile

```C
MSStatus MSTrainModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,
                                          const MSContextHandle model_context, const MSTrainCfgHandle train_cfg)
```

从模型路径构建可在设备上运行的训练模型，仅适用于训练。

- 参数

    - `mode`: 模型对象句柄。
    - `model_path`: 模型文件的存储路径。
    - `model_type`: 模型文件的类型。
    - `model_context`: 模型执行期间上下文。
    - `train_cfg`: 训练使用的配置。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSRunStep

```C
MSStatus MSRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after)
```

单步模型训练，仅适用于训练。

- 参数

    - `mode`: 模型对象句柄。
    - `before`: 模型执行前回调函数。
    - `after`: 模型执行后回调函数。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelSetLearningRate

```C
MSStatus MSModelSetLearningRate(MSModelHandle model, float learning_rate)
```

设置模型学习率，仅适用于轻量训练。

- 参数

    - `mode`: 模型对象句柄。
    - `learning_rate`: 设置的模型学习率。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelGetLearningRate

```C
float MSModelGetLearningRate(MSModelHandle model)
```

获取模型学习率，仅适用于轻量训练。

- 参数

    - `mode`: 模型对象句柄。

- 返回值

  模型学习率。

#### MSModelGetWeights

```C
MSTensorHandleArray MSModelGetWeights(MSModelHandle model)
```

获取模型所有权重组成的tensor数组，仅适用于轻量训练。

- 参数

    - `mode`: 模型对象句柄。

- 返回值

  模型所有权重组成的tensor数组句柄。

#### MSModelUpdateWeights

```C
MSStatus MSModelUpdateWeights(MSModelHandle model, const MSTensorHandleArray new_weights)
```

更新模型所有权重，仅适用于轻量训练。

- 参数

    - `mode`: 模型对象句柄。
    - `new_weights`: 需要更新的模型权重组成的tensor数组句柄。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelGetTrainMode

```C
bool MSModelGetTrainMode(MSModelHandle model)
```

获得模型是否为训练模式。

- 参数

    - `mode`: 模型对象句柄。

- 返回值

  bool值，是否为训练模型。

#### MSModelSetTrainMode

```C
MSStatus MSModelSetTrainMode(MSModelHandle model, bool train)
```

设置模型是否为训练。

- 参数

    - `mode`: 模型对象句柄。
    - `train`: bool值，指定模型是否训练。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSModelSetupVirtualBatch

```C
MSStatus MSModelSetupVirtualBatch(MSModelHandle model, int virtual_batch_multiplier, float lr, float momentum)
```

配置虚拟批次训练，仅在训练时有效。

- 参数

    - `mode`: 模型对象句柄。
    - `virtual_batch_multiplier`: 虚拟批次系数，若设为小于1的数值则禁用此功能。
    - `lr`: 虚拟批次训练使用的学习率，设为-1时采用内部默认配置。
    - `momentum`: 虚拟批次训练中BatchNorm层使用的动量参数，设为-1时采用内部默认配置。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSExportModel

```C
MSStatus MSExportModel(MSModelHandle model, MSModelType model_type, const char *model_file,
                              MSQuantizationType quantization_type, bool export_inference_only,
                              char **output_tensor_name, size_t num)
```

导出训练模型，仅在训练时有效。

- 参数

    - `mode`: 模型对象句柄。
    - `model_type`: 模型文件类型。
    - `model_file`: 模型路径。
    - `quantization_type`: 模型量化类型。
    - `export_inference_only`: 是否仅导出推理模型。
    - `output_tensor_name`: 用于设置导出推理模型的输出张量名称，默认为空，此时导出完整推理模型。
    - `num`: 输出张量名称数量。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSExportModelBuffer

```C
MSStatus MSExportModelBuffer(MSModelHandle model, MSModelType model_type, char **model_data, size_t *data_size,
                                    MSQuantizationType quantization_type, bool export_inference_only,
                                    char **output_tensor_name, size_t num)
```

从内存缓存导出训练模型，仅在训练时有效。

- 参数

    - `mode`: 模型对象句柄。
    - `model_type`: 模型文件类型。
    - `model_data`: 模型的缓存数据。
    - `data_size`: 导出模型的缓存大小。
    - `quantization_type`: 模型量化类型。
    - `export_inference_only`: 是否仅导出推理模型。
    - `output_tensor_name`: 用于设置导出推理模型的输出张量名称，默认为空，此时导出完整推理模型。
    - `num`: 输出张量名称数量。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

#### MSExportWeightsCollaborateWithMicro

```C
MSStatus MSExportWeightsCollaborateWithMicro(MSModelHandle model, MSModelType model_type,
                                                    const char *weight_file, bool is_inference, bool enable_fp16,
                                                    char **changeable_weights_name, size_t num)
```

导出训练模型权重，仅在端侧micro训练时有效。

- 参数

    - `mode`: 模型对象句柄。
    - `model_type`: 模型文件类型。
    - `weight_file`: 导出权重文件路径。
    - `is_inference`: 是否从推理图模型导出权重，当前仅支持设置为`true`。
    - `enable_fp16`: 浮点权重是否以float16格式保存。
    - `changeable_weights_name`: 设置可动态改变形状的权重张量名称。
    - `num`: 可变权重张量名称的数量。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明成功。

### 公有数据类型

#### MSModelHandle

```C
typedef void* MSModelHandle;
```

模型对象指针。

#### MSTrainCfgHandle

```C
typedef void* MSTrainCfgHandle;
```

模型训练配置对象指针。

#### MSTensorHandleArray

```C
typedef struct MSTensorHandleArray {
  size_t handle_num;
  MSTensorHandle* handle_list;
} MSTensorHandleArray;
```

张量数组结构体

- 成员变量

    - `handle_num`: 张量数组长度。
    - `handle_list`: 张量数组。

#### MSShapeInfo

```C
#define MS_MAX_SHAPE_NUM 32
typedef struct MSShapeInfo {
  size_t shape_num;
  int64_t shape[MS_MAX_SHAPE_NUM];
} MSShapeInfo;
```

维度信息结构体，最大支持的维度为`MS_MAX_SHAPE_NUM`。

- 成员变量

    - `shape_num`: 维度数组长度。
    - `shape`: 维度数组。

#### MSCallBackParamC

```C
typedef struct MSCallBackParamC {
  char* node_name;
  char* node_type;
} MSCallBackParamC;
```

回调函数中存储的算子信息的参数。

- 成员变量
    - `node_name`: 算子名称。
    - `node_type`: 算子类型。

#### MSKernelCallBackC

```C
typedef bool (*MSKernelCallBackC)(const MSTensorHandleArray inputs,
                                  const MSTensorHandleArray outputs,
                                  const MSCallBackParamC kernel_Info);
```

回调函数指针类型。该函数指针是用于[MSModelPredict](#msmodelpredict)接口，是在算子执行前或执行后运行的回调函数。

