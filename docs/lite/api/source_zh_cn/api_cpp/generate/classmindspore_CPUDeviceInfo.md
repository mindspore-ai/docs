# Class CPUDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore-lite/blob/master/include/api/context.h)&gt;

派生自[DeviceInfoContext](./classmindspore_DeviceInfoContext.md)，模型运行在CPU上的配置。

## 公有成员函数

|     函数     | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------ | ---------|---------|
| [enum DeviceType GetDeviceType() const override](#getdevicetype) |        √ |        √ |
| [void SetEnableFP16(bool is_fp16)](#setenablefp16)      |        √ |        √ |
| [bool GetEnableFP16() const](#getenablefp16)            |        √ |        √ |

### GetDeviceType

```cpp
enum DeviceType GetDeviceType() const override
```

- 返回值

  DeviceType::kCPU

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
