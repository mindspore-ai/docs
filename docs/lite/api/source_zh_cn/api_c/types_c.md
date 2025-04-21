# types_c

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_c/types_c.md)

```C
#include<types_c.h>
```

该文的件定义了一些枚举类型的数据类型。

## MSModelType

模型文件类型。

```C
typedef enum MSModelType {
  kMSModelTypeMindIR = 0,
  kMSModelTypeInvalid = 0xFFFFFFFF
} MSModelType;
```

| 类型定义            | 值         | 描述         |
| ------------------- | ---------- | ------------ |
| kMSModelTypeMindIR  | 0          | MindIR类型。 |
| kMSModelTypeInvalid | 0xFFFFFFFF | 非法类型。   |

## MSDeviceType

设备类型。

```C
typedef enum MSDeviceType {
  kMSDeviceTypeCPU = 0,
  kMSDeviceTypeGPU,
  kMSDeviceTypeKirinNPU,
  kMSDeviceTypeInvalid = 100,
} MSDeviceType;
```

| 定义                  | 值  | 描述          |
| --------------------- | --- | ------------- |
| kMSDeviceTypeCPU      | 0   | 设备类型是CPU |
| kMSDeviceTypeGPU      | 1   | 设备类型是GPU |
| kMSDeviceTypeKirinNPU | 2   | 设备类型是NPU |
| kMSDeviceTypeInvalid  | 100 | 设备类型非法  |

