# model_c

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_c/model_c.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

```C
#include<model_c.h>
```

Model定义了MindSpore中编译和运行的模型。

## 公有函数和数据类型

| function                                                                                                                                                                                      |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MSModelHandle MSModelCreate()](#msmodelcreate)                                                                                                                                               |
| [void MSModelDestroy(MSModelHandle *model)](#msmodeldestroy)                                                                                                                                  |
| [void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size)](#msmodelsetworkspace)                                                                                 |
| [MSStatus MSModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type, const MSContextHandle model_context)](#msmodelbuild)                            |
| [MSStatus MSModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,const MSContextHandle model_context)](#msmodelbuildfromfile)                               |
| [MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos,size_t shape_info_num)](#msmodelresize)                                               |
| [MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,const MSKernelCallBackC before, const MSKernelCallBackC after)](#msmodelpredict) |
| [MSTensorHandleArray MSModelGetInputs(const MSModelHandle model)](#msmodelgetinputs)                                                                                                          |
| [MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model)](#msmodelgetoutputs)                                                                                                        |
| [MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char *tensor_name)](#msmodelgetinputbytensorname)                                                                |
| [MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model, const char *tensor_name)](#msmodelgetoutputbytensorname)                                                              |
| [ModelType](#msmodeltype)                                                                                                                                                                     |

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
void MSModelDestroy(MSModelHandle *model)
```

销毁一个模型对象，该选项仅MindSpore Lite有效。

- 参数

    - `model`:指向模型对象指针的指针。

#### MSModelSetWorkspace

```C
void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size)
```

设置模型的工作空间，该选项仅对IoT有效。(该接口暂未实现)

- 参数
    - `model`: 指向模型对象的指针。
    - `workspace`: 指向工作空间的指针。
    - `workspace_size`: 工作空间大小。

#### MSModelBuild

```C
MSStatus MSModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type, const MSContextHandle model_context)
```

从内存缓冲区加载并编译MindSpore模型。

- 参数

    - `model`: 指向模型对象的指针。
    - `model_data`: 内存中已经加载的模型数据地址。
    - `data_size`: 模型数据的长度。
    - `model_type`: 模型文件类型，具体见: [MSModelType](#MSModelType)。
    - `model_context`: 模型的上下文环境，具体见: [Context](./context_c.md)。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明创建成功。

#### MSModelBuildFromFile

```C
MSStatus MSModelBuildFromFile(MSModelHandle model, const char *model_path, MSModelType model_type,
                                     const MSContextHandle model_context)
```

通过模型文件中加载并编译MindSpore模型。

- 参数

    - `model`: 指向模型对象的指针。
    - `model_path`: 模型文件路径。
    - `model_type`: 模型文件类型，具体见: [MSModelType](#MSModelType)。
    - `model_context`: 模型的上下文环境，具体见: [Context](./context_c.md)。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明创建成功。

#### MSModelResize

```C
MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos,
                              size_t shape_info_num)
```

调整已编译模型的输入形状。

- 参数

    - `model`: 指向模型对象的指针。
    - `inputs`: 模型输入对应的张量数组结构体。
    - `shape_infos`: 输入形状信息数组，按模型输入顺序排列的由形状信息组成的数组，模型会按顺序依次调整张量形状。
    - `shape_info_num`: 形状信息数组的长度。

- 返回值

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明创建成功。

#### MSModelPredict

```C
MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
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

  枚举类型的状态码`MSStatus`，若返回`MSStatus::kMSStatusSuccess`则证明创建成功。

#### MSModelGetInputs

```C
MSTensorHandleArray MSModelGetInputs(const MSModelHandle model)
```

获取MindSpore模型的输入张量数组结构体。

- 参数

    - `model`: 指向模型对象的指针。

- 返回值

  张量数组结构体。

#### MSModelGetOutputs

```C
MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model)
```

获取MindSpore模型的输出张量数组结构体。

- 参数

    - `model`: 指向模型对象的指针。

- 返回值

  张量数组结构体。

#### MSModelGetInputByTensorName

```C
MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model,
                                            const char *tensor_name)
```

通过张量名获取MindSpore模型的输入张量。

- 参数

    - `model`: 指向模型对象的指针。
    - `tensor_name`: 张量名。

- 返回值

  tensor_name所对应的张量指针。

#### MSModelGetOutputByTensorName

```C
MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model,
                                            const char *tensor_name)
```

通过张量名获取MindSpore模型的输出张量。

- 参数

    - `model`: 指向模型对象的指针。
    - `tensor_name`: 张量名。

- 返回值

  tensor_name所对应的张量指针。

### 公有数据类型

#### MSModelHandle

```C
typedef void *MSModelHandle;
```

模型对象指针。

#### MSTensorHandleArray

```C
typedef struct MSTensorHandleArray {
  size_t handle_num;
  MSTensorHandle *handle_list;
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

模型维度信息，最大的维度为`MS_MAX_SHAPE_NUM`。

- 成员变量

    - `shape_num`: 维度数组长度。
    - `shape`: 维度数组。

#### MSCallBackParamC

```C
typedef struct MSCallBackParamC {
  char *node_name;
  char *node_type;
} MSCallBackParamC;
```

回调函数的参数与返回值。

- 成员变量
    - `node_name`: 节点名称参数。
    - `node_type`: 节点类型参数。

#### MSKernelCallBackC

```C
typedef bool (*MSKernelCallBackC)(const MSTensorHandleArray inputs,
                                  const MSTensorHandleArray outputs,
                                  const MSCallBackParamC kernel_Info);
```

回调函数指针类型。

#### MSModelType

```C
#include<types_c.h>
```

模型文件类型。

```C
typedef enum MSModelType {
  kMSModelTypeMindIR = 0,
  kMSModelTypeInvalid = 0xFFFFFFFF
} MSModelType;
```

| 类型定义            | 值         | 描述       |
| ------------------- | ---------- | ---------- |
| kMSModelTypeMindIR  | 0          | MindIR类型。 |
| kMSModelTypeInvalid | 0xFFFFFFFF | 非法类型。   |