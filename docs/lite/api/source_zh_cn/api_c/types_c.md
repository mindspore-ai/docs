# types_c

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_c/types_c.md)

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

## MSOptimizationLevel

模型优化级别。

```C
typedef enum MSOptimizationLevel {
  kMSKO0 = 0,
  kMSKO2 = 2,
  kMSKO3 = 3,
  kMSKAUTO = 4,
  kMSKOPTIMIZATIONTYPE = 0xFFFFFFFF
} MSOptimizationLevel;
```

| 定义                   | 值         | 描述                                                     |
| --------------------- | ---------- | ------------------------------------------------------- |
| kMSKO0                | 0          | 不进行优化                                                |
| kMSKO2                | 2          | 将网络转换为float16精度，保留BatchNorm和损失函数为float32精度  |
| kMSKO3                | 3          | 将整个网络（包括BatchNorm）转换为float16精度                  |
| kMSKAUTO              | 4          | 根据设备类型自动化选择优化策略                                |
| kMSKOPTIMIZATIONTYPE  | 0xFFFFFFFF | 非法类型                                                  |

## MSQuantizationType

模型优化级别。

```C
typedef enum MSQuantizationType {
  kMSNO_QUANT = 0,
  kMSWEIGHT_QUANT = 1,
  kMSFULL_QUANT = 2,
  kMSUNKNOWN_QUANT_TYPE = 0xFFFFFFFF
} MSQuantizationType;
```

| 定义                   | 值         | 描述                 |
| --------------------- | ---------- | ------------------- |
| kMSNO_QUANT           | 0          | 不进行量化            |
| kMSWEIGHT_QUANT       | 1          | 权重量化             |
| kMSFULL_QUANT         | 2          | 权量化               |
| kMSKOPTIMIZATIONTYPE  | 0xFFFFFFFF | 非法类型             |