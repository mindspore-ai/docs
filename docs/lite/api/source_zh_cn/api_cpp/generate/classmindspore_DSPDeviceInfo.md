# Class DSPDeviceInfo

\#include &lt;[context.h](https://gitee.com/mindspore/mindspore-lite/blob/master/include/api/context.h)&gt;

派生自[DeviceInfoContext](./classmindspore_DeviceInfoContext.md)，模型运行在DSP上的配置。

## 公有成员函数

|     函数     | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------ | ---------|---------|
| [enum DeviceType GetDeviceType() const override](#getdevicetype) |        x |        √ |
| [void SetDeviceID(uint32_t device_id)](#setdeviceid)             |        x |        √ |
| [uint32_t GetDeviceID() const](#getdeviceid)                     |        x |        √ |

### GetDeviceType

```cpp
enum DeviceType GetDeviceType() const override
```

- 返回值

  DeviceType::kDSP

### SetDeviceID

```cpp
void SetDeviceID(uint32_t device_id)
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
