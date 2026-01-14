# Class AscendDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore-lite/blob/master/include/api/context.h)&gt;

派生自[DeviceInfoContext](./classmindspore_DeviceInfoContext.md)，模型运行在Atlas 200/300/500推理产品、Atlas推理系列产品上的配置。

## 公有成员函数

| 函数                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------------------------|--------|--------|
| [enum DeviceType GetDeviceType() const override](#getdevicetype)                            |    √     |    √     |
| [void SetDeviceID(uint32_t device_id)](#setdeviceid)                                    |        √ |    √     |
| [uint32_t GetDeviceID() const](#getdeviceid)                                     |        √ |    √     |
| [void SetRankID(uint32_t rank_id)](#setrankid)                                   |     √ |    √     |
| [uint32_t GetRankID() const](#getrankid)                                         | √     |    √     |
| [inline void SetInsertOpConfigPath(const std::string &cfg_path)](#setinsertopconfigpath)          |        √ |    √     |
| [inline std::string GetInsertOpConfigPath() const](#getinsertopconfigpath)                              |        √ |    √     |
| [inline void SetInputFormat(const std::string &format)](#setinputformat)                   |        √ |    √     |
| [inline std::string GetInputFormat() const](#getinputformat)                                     |        √ |    √     |
| [inline void SetInputShape(const std::string &shape)](#setinputshape)                     |        √ |    √     |
| [inline std::string GetInputShape() const](#getinputshape)                                      |        √ |    √     |
| [void SetInputShapeMap(const std::map<int, std::vector\<int\>> &shape)](#setinputshapemap)             |        √ |    √     |
| [std::map<int, std::vector\<int\>> GetInputShapeMap() const](#getinputshapemap)                          |        √ |    √     |
| [void SetDynamicBatchSize(const std::vector\<size_t\> &dynamic_batch_size)](#setdynamicbatchsize)        |        √ |    √     |
| [inline std::string GetDynamicBatchSize() const](#getdynamicbatchsize)                                 |        √ |    √     |
| [inline void SetDynamicImageSize(const std::string &dynamic_image_size)](#setdynamicimagesize)         |        √ |    √     |
| [inline std::string GetDynamicImageSize() const](#getdynamicimagesize)                                 |        √ |    √     |
| [void SetOutputType(enum DataType output_type)](#setoutputtype)                    |        √ |    √     |
| [enum DataType GetOutputType() const](#getoutputtype)                                   |        √ |    √     |
| [inline void SetPrecisionMode(const std::string &precision_mode)](#setprecisionmode)        |        √ |    √     |
| [inline std::string GetPrecisionMode() const](#getprecisionmode)                                |        √ |    √     |
| [inline void SetOpSelectImplMode(const std::string &op_select_impl_mode)](#setopselectimplmode) |        √ |    √     |
| [inline std::string GetOpSelectImplMode() const](#getopselectimplmode)                                |        √ |    √     |
| [inline void SetFusionSwitchConfigPath(const std::string &cfg_path)](#setfusionswitchconfigpath) |        √ |    √     |
| [inline std::string GetFusionSwitchConfigPath() const](#getfusionswitchconfigpath)                   |        √ |    √     |
| [inline void SetBufferOptimizeMode(const std::string &buffer_optimize_mode)](#setbufferoptimizemode) |        √ |    √     |
| [inline std::string GetBufferOptimizeMode() const](#getbufferoptimizemode)                           |        √ |    √     |

### GetDeviceType

```cpp
enum DeviceType GetDeviceType() const override
```

- 返回值

  DeviceType::kAscend

### SetDeviceID

```cpp
void SetDeviceType(uint32_t device_id)
```

用于指定设备ID。

- 参数

    - `device_id`: 设备ID。

### GetDeviceID

```cpp
uint32_t GetDeviceID() const
```

- 返回值

  已配置的设备ID。

### SetRankID

```cpp
void SetRankID(uint32_t rank_id)
```

指定模型rank_id。

- 参数

    - `rank_id`: Rank ID。

### GetRankID

```cpp
uint32_t GetRankID() const
```

- 返回值

  获取模型的rank_id。

### SetInsertOpConfigPath

```cpp
inline void SetInsertOpConfigPath(const std::string &cfg_path)
```

模型插入[AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html)算子。

- 参数

    - `cfg_path`: [AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html)配置文件路径。

### GetInsertOpConfigPath

```cpp
inline std::string GetInsertOpConfigPath() const
```

- 返回值

  已配置的[AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html)。

### SetInputFormat

```cpp
inline void SetInputFormat(const std::string &format)
```

指定模型输入格式。

- 参数

    - `format`: 输入格式，可选值有 `"NCHW"`、`"NHWC"` 和 `"ND"`。

### GetInputFormat

```cpp
inline std::string GetInputFormat() const
```

- 返回值

  已配置的模型输入格式。

### SetInputShape

```cpp
inline void SetInputShape(const std::string &shape)
```

指定模型输入shape，为字符串形式，需指定输入名称，每个shape值由`,`隔开，不同输入由`;`隔开。

- 参数

    - `shape`: 如 `"input_op_name1:1,2,3,4;input_op_name2:4,3,2,1"`。

### GetInputShape

```cpp
inline std::string GetInputShape() const
```

- 返回值

  已配置的模型输入shape。

### SetInputShapeMap

```cpp
inline void SetInputShapeMap(const std::map<int, std::vector<int>> &shape)
```

指定模型输入shape。

- 参数

    - `shape`: map的key对应输入的下标，例如第一个输入对应下标0，第二个对应下标1。value对应输入shape，为数组形式。

### GetInputShapeMap

```cpp
std::map<int, std::vector<int>> GetInputShapeMap() const
```

- 返回值

  已配置模型输入shape。

### SetDynamicBatchSize

```cpp
void SetDynamicBatchSize(const std::vector<size_t> &dynamic_batch_size)
```

指定模型动态batch的档位，支持个数范围[2, 100]，为数组形式。

- 参数

    - `dynamic_batch_size`: 如`{1, 2}`。

### GetDynamicBatchSize

```cpp
inline std::string GetDynamicBatchSize() const
```

- 返回值

  已配置模型的动态batch。

### SetDynamicImageSize

```cpp
inline void SetDynamicImageSize(const std::string &dynamic_image_size)
```

指定模型动态分辨率的档位，支持个数范围[2, 100]，为字符串形式，每个shape值由`,`隔开，不同输入由`;`隔开。

- 参数

    - `dynamic_image_size`: 如`"64,64;128,128"`。

### GetDynamicImageSize

```cpp
inline std::string GetDynamicImageSize() const
```

- 返回值

  已配置模型的动态分辨率。

### SetOutputType

```cpp
void SetOutputType(enum DataType output_type)
```

指定模型输出type。

- 参数

    - `output_type`: 可选有 `"FP32"`、`"UINT8"` 或 `"FP16"`。

### GetOutputType

```cpp
enum DataType GetOutputType() const
```

- 返回值

  已配置模型的输出类型。

### SetPrecisionMode

```cpp
inline void SetPrecisionMode(const std::string &precision_mode)
```

配置模型精度模式。

- 参数

    - `precision_mode`: 可选有 `"enforce_fp16"`、`"preferred_fp32"`、`"enforce_origin"`、`"enforce_fp32"` 或者`"preferred_optimal"`，默认为 `"enforce_fp16"`。

### GetPrecisionMode

```cpp
inline std::string GetPrecisionMode() const
```

- 返回值

  已配置模型的精度模式。

### SetOpSelectImplMode

```cpp
inline void SetOpSelectImplMode(const std::string &op_select_impl_mode)
```

配置算子实现方式。

- 参数

    - `op_select_impl_mode`: 可选有 `"high_performance"` 和 `"high_precision"`，默认为 `"high_performance"`。

### GetOpSelectImplMode

```cpp
inline std::string GetOpSelectImplMode() const
```

- 返回值

  已配置的算子选择模式。

### SetFusionSwitchConfigPath

```cpp
inline void SetFusionSwitchConfigPath(const std::string &cfg_path)
```

配置融合开关。

- 参数

    - `cfg_path`: 融合开关配置文件，可指定关闭特定融合规则。

### GetFusionSwitchConfigPath

```cpp
inline std::string GetFusionSwitchConfigPath() const
```

- 返回值

  已配置的融合开关文件路径。

### SetBufferOptimizeMode

```cpp
inline void SetBufferOptimizeMode(const std::string &buffer_optimize_mode)
```

配置缓存优化模式。

- 参数

    - `buffer_optimize_mode`: 可选有 `"l1_optimize"`、`"l2_optimize"` 或 `"off_optimize"`，默认为 `"l2_optimize"`。

### GetBufferOptimizeMode

```cpp
inline std::string GetBufferOptimizeMode() const
```

- 返回值

  已配置的缓存优化模式。
