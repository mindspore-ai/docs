# AscendDeviceInfo

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0rc1/docs/lite/api/source_zh_cn/api_java/ascenddeviceinfo.md)

```java
import com.mindspore.config.AscendDeviceInfo;
```

用于配置MindSpore Lite Ascend设备选项。

## 公有成员函数

| function                                   | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------------------------------------ |--------|--------|
| [int getDeviceID)](#getDeviceID) | √      | ✕     |
| [int getDeviceType](#getDeviceType) | √      | ✕     |
| [String getProvider](#getProvider) | √      | ✕     |
| [void setProvider(String provider)](#setProvider) | √      | ✕     |
| [void setDeviceID(int deviceId)](#setDeviceID) | √ | ✕ |
| [int getRankID()](#getRankID) | √      | ✕     |
| [void setRankID(int rankId)](#setRankID) | √      | ✕     |
| [String getInsertOpConfigPath()](#getInsertOpConfigPath) | √      | ✕     |
| [void setInsertOpConfigPath(String insertOpConfigPath)](#setInsertOpConfigPath) | √      | ✕     |
| [String getInputFormat()](#getInputFormat) | √ | ✕ |
| [void setInputFormat(String inputFormat)](#setInputFormat) | √      | ✕     |
| [String getInputShape()](#getInputShape) | √      | ✕     |
| [void setInputShape(String inputShape)](#setInputShape) | √      | ✕     |
| [HashMap<Integer, ArrayList<Integer>> getInputShapeMap()](#getInputShapeMap) | √      | ✕     |
| [void setInputShapeMap(HashMap<Integer, ArrayList<Integer>> inputShapeMap)](#setInputShapeMap) | √      | ✕     |
| [ArrayList<Integer> getDynamicBatchSize()](#getDynamicBatchSize) | √      | ✕     |
| [void setDynamicBatchSize(ArrayList<Integer> dynamicBatchSize)](#setDynamicBatchSize) | √      | ✕     |
| [String getDynamicImageSize()](#getDynamicImageSize) | √      | ✕     |
| [void setDynamicImageSize(String dynamicImageSize)](#setDynamicImageSize) | √      | ✕     |
| [int getOutputType()](#getOutputType) | √ | ✕ |
| [void setOutputType(int outputType)](#setOutputType) | √      | ✕     |
| [String getPrecisionMode()](#getPrecisionMode) | √ | ✕ |
| [void setPrecisionMode(String precisionMode)](#setPrecisionMode) | √ | ✕ |
| [String getOpSelectImplMode()](#getOpSelectImplMode) | √ | ✕ |
| [void setOpSelectImplMode(String opSelectImplMode)](#setOpSelectImplMode) | √ | ✕ |
| [String getFusionSwitchConfigPath()](#getFusionSwitchConfigPath) | √ | ✕ |
| [void setFusionSwitchConfigPath(String fusionSwitchConfigPath)](#setFusionSwitchConfigPath) | √ | ✕ |
| [String getBufferOptimizeMode()](#getBufferOptimizeMode) | √ | ✕ |
| [void setBufferOptimizeMode(String bufferOptimizeMode)](#setBufferOptimizeMode) | √ | ✕ |

## getDeviceID

```java
public int getDeviceID()
```

获取昇腾设备ID。

- 返回值

  设备ID。

## getDeviceType

```java
public int getDeviceType()
```

获取昇腾设备类型。

- 返回值

  设备类型。

## getProvider

```java
public String getProvider()
```

获取模型后端类型。

- 返回值

  模型后端类型。

## setProvider

```java
public void setProvider(String provider)
```

设置模型后端类型。

- 参数
- `provider`: 要设置的模型后端类型。

## setDeviceID

```java
public void setDeviceID(int deviceId)
```

设置昇腾设备ID。

- 参数
- `deviceId`: 要设置的设备ID。

## getRankID

```java
public int getRankID()
```

获取集群中分布式模型的逻辑ID。

- 返回值

  逻辑ID。

## setRankID

```java
public void setRankID(int rankId)
```

设置集群中分布式模型的逻辑ID。

- 参数

- `rankId`: 要设置的逻辑ID。

## getInsertOpConfigPath

```java
public String getInsertOpConfigPath()
```

获取AIPP配置文件路径。

- 返回值

  配置文件路径。

## setInsertOpConfigPath

```java
public void setInsertOpConfigPath(String insertOpConfigPath)
```

设置AIPP配置文件路径。

- 参数

    - `insertOpConfigPath`: 要设置的AIPP配置文件路径。

## getInputFormat

```java
public String getInputFormat()
```

获取模型的输入格式。

- 返回值

    输入格式。

## setInputFormat

```java
public void setInputFormat(String inputFormat)
```

设置模型的输入格式。可选"NCHW"，"NHWC"和"ND"。

- 参数

- `inputFormat`: 要设置的输入格式。

## getInputShape

```java
public String getInputShape()
```

获取模型的输入形状。

- 返回值

  输入形状。

## setInputShape

```java
public void setInputShape(String inputShape)
```

设置模型的输入形状。例如"input_op_name1:1,2,3,4;input_op_name2:4,3,2,1;"。

- 参数
- `inputShape`: 模型输入形状。

## getInputShapeMap

```java
public HashMap<Integer, ArrayList<Integer>> getInputShapeMap()
```

获取模型的输入形状映射。

- 返回值

  输入形状映射。

## setInputShapeMap

```java
public void setInputShapeMap(HashMap<Integer, ArrayList<Integer>> inputShapeMap)
```

设置模型的输入形状。例如{{0, {1,2,3,4}}, {1, {4,3,2,1}}}。

- 参数
- `inputShapeMap`: 要设置的输入形状映射。

## getDynamicBatchSize

```java
public ArrayList<Integer> getDynamicBatchSize()
```

获取模型的动态batch大小。

- 返回值

  动态batch大小。

## setDynamicBatchSize

```java
public void setDynamicBatchSize(ArrayList<Integer> dynamicBatchSize)
```

设置模型的动态batch大小。取值范围为 2 到 100。例如 {1, 2} 表示batch大小配置为 1 和 2。

- 参数
- `dynamicBatchSize`: 要设置的动态batch大小。

## getDynamicImageSize

```java
public String getDynamicImageSize()
```

获取模型的动态图像尺寸。

- 返回值

  动态图像尺寸。

## setDynamicImageSize

```java
public void setDynamicImageSize(String dynamicImageSize)
```

设置模型的动态图像尺寸。

- 参数
- `dynamicImageSize`: 要设置的动态图像尺寸。

## getOutputType

```java
public int getOutputType()
```

获取模型的输出类型。

- 返回值

  输出类型。

## setOutputType

```java
public void setOutputType(int outputType)
```

设置模型的输出类型。可以是 DataType.kNumberTypeFloat32、DataType.kNumberTypeUInt8 或 DataType.kNumberTypeFloat16。

- 参数
- `outputType`: 要设置的输出类型。

## getPrecisionMode

```java
public String getPrecisionMode()
```

获取模型的精度模式。

- 返回值

  精度模式。

## setPrecisionMode

```java
public void setPrecisionMode(String precisionMode)
```

设置模型的精度模式。可选"enforce_fp16"、"preferred_fp32"、"enforce_origin"、"enforce_fp32" 和 "preferred_optimal"，"enforce_fp16" 为默认值。

- 参数
- `precisionMode`: 要设置的精度模式。

## getOpSelectImplMode

```java
public String getOpSelectImplMode()
```

获取模型的算子选择实现方式。

- 返回值

  算子实现方式。

## setOpSelectImplMode

```java
public void setOpSelectImplMode(String opSelectImplMode)
```

设置模型的算子选择实现方式。可选“high_performance”和“high_precision”，默认设置为“high_performace”。

- 参数
- `opSelectImplMode`: 算子实现方式。

## getFusionSwitchConfigPath

```java
public String getFusionSwitchConfigPath()
```

获取融合开关配置文件路径，它控制要关闭哪些融合pass。

- 返回值

  融合开关配置文件路径。

## setFusionSwitchConfigPath

```java
public void setFusionSwitchConfigPath(String fusionSwitchConfigPath)
```

设置融合开关配置文件路径。

- 参数
- `fusionSwitchConfigPath`: 要设置的融合开关配置文件路径。

## getBufferOptimizeMode

```java
public String getBufferOptimizeMode()
```

获取缓冲区优化模式。可选“l1_optimize”、“l2_optimize”或“off_optimize”。“l2_optimize”设置为默认值。

- 返回值

  缓冲区优化模式。

## setBufferOptimizeMode

```java
public void setBufferOptimizeMode(String bufferOptimizeMode)
```

设置缓冲区优化模式。

- 参数
- `bufferOptimizeMode`: 要设置的缓冲区优化模式。