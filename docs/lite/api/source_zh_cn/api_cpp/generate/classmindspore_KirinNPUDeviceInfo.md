# Class KirinNPUDeviceInfo

\#include &lt;[context.h](https://atomgit.com/mindspore/mindspore-lite/blob/r2.9.0/include/api/context.h)&gt;

派生自[DeviceInfoContext](./classmindspore_DeviceInfoContext.md)，模型运行在NPU上的配置。

## 公有成员函数

|     函数     | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------ |-------|-------|
| [enum DeviceType GetDeviceType() const override](#getdevicetype) |    √   |    √   |
| [void SetEnableFP16(bool is_fp16)](#setenablefp16)                          |        ✕ |    √   |
| [bool GetEnableFP16() const](#getenablefp16)                                |       ✕ |    √   |
| [void SetFrequency(int frequency)](#setfrequency)      | ✕     |    √   |
| [int GetFrequency() const](#getfrequency)              | ✕     |    √   |

### GetDeviceType

```cpp
enum DeviceType GetDeviceType() const override
```

- 返回值

  DeviceType::kKirinNPU

### SetEnableFP16

```cpp
void SetEnableFP16(bool is_fp16)
```

用于指定是否以FP16精度进行推理。

- 参数

    - `is_fp16`: 是否以FP16精度进行推理。

### GetEnableFP16

```cpp
bool GetEnableFP16() const
```

- 返回值

  已配置的精度模式。

### SetFrequency

```cpp
void SetFrequency(int frequency)
```

用于指定NPU频率。

- 参数

    - `frequency`: 设置为1（低功耗）、2（均衡）、3（高性能）、4（极致性能），默认为3。

### GetFrequency

```cpp
int GetFrequency() const
```

- 返回值

  已配置的NPU频率模式。
