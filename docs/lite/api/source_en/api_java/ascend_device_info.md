# AscendDeviceInfo

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/feature/atomgit/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/ascend_device_info.md)

```java
import com.mindspore.config.AscendDeviceInfo;
```

The AscendDeviceInfo class is used to configure MindSpore Lite Ascend device options.

## Public Member Functions

| function                                   | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------------------------ |--------|--------|
| [int getDeviceID()](#getdeviceid) | √      | ✕     |
| [int getDeviceType](#getdevicetype) | √      | ✕     |
| [String getProvider](#getprovider) | √      | ✕     |
| [void setProvider(String provider)](#setprovider) | √      | ✕     |
| [void setDeviceID(int deviceId)](#setdeviceid) | √ | ✕ |
| [int getRankID()](#getrankid) | √      | ✕     |
| [void setRankID(int rankId)](#setrankid) | √      | ✕     |
| [String getInsertOpConfigPath()](#getinsertopconfigpath) | √      | ✕     |
| [void setInsertOpConfigPath(String insertOpConfigPath)](#setinsertopconfigpath) | √      | ✕     |
| [String getInputFormat()](#getinputformat) | √ | ✕ |
| [void setInputFormat(String inputFormat)](#setinputformat) | √      | ✕     |
| [String getInputShape()](#getinputshape) | √      | ✕     |
| [void setInputShape(String inputShape)](#setinputshape) | √      | ✕     |
| [HashMap<Integer, ArrayList<Integer>> getInputShapeMap()](#getinputshapemap) | √      | ✕     |
| [void setInputShapeMap(HashMap<Integer, ArrayList<Integer>> inputShapeMap)](#setinputshapemap) | √      | ✕     |
| [ArrayList<Integer> getDynamicBatchSize()](#getdynamicbatchsize) | √      | ✕     |
| [void setDynamicBatchSize(ArrayList<Integer> dynamicBatchSize)](#setdynamicbatchsize) | √      | ✕     |
| [String getDynamicImageSize()](#getdynamicimagesize) | √      | ✕     |
| [void setDynamicImageSize(String dynamicImageSize)](#setdynamicimagesize) | √      | ✕     |
| [int getOutputType()](#getoutputtype) | √ | ✕ |
| [void setOutputType(int outputType)](#setoutputtype) | √      | ✕     |
| [String getPrecisionMode()](#getprecisionmode) | √ | ✕ |
| [void setPrecisionMode(String precisionMode)](#setprecisionmode) | √ | ✕ |
| [String getOpSelectImplMode()](#getopselectimplmode) | √ | ✕ |
| [void setOpSelectImplMode(String opSelectImplMode)](#setopselectimplmode) | √ | ✕ |
| [String getFusionSwitchConfigPath()](#getfusionswitchconfigpath) | √ | ✕ |
| [void setFusionSwitchConfigPath(String fusionSwitchConfigPath)](#setfusionswitchconfigpath) | √ | ✕ |
| [String getBufferOptimizeMode()](#getbufferoptimizemode) | √ | ✕ |
| [void setBufferOptimizeMode(String bufferOptimizeMode)](#setbufferoptimizemode) | √ | ✕ |

## getDeviceID

```java
public int getDeviceID()
```

Get the Ascend device ID.

- Returns

  the deviceId.

## getDeviceType

```java
public int getDeviceType()
```

Get the device type.

- Returns

  the deviceType.

## getProvider

```java
public String getProvider()
```

Get Ascend device provider.

- Returns

  the provider.

## setProvider

```java
public void setProvider(String provider)
```

Set Ascend device provider.

- Parameters
- `provider`: provider the provider to set.

## setDeviceID

```java
public void setDeviceID(int deviceId)
```

Set the Ascend device ID.

- Parameters
- `deviceId`: deviceId the deviceId to set.

## getRankID

```java
public int getRankID()
```

Get the logical ID of a distributed model in the cluster.

- Returns

  the rankId.

## setRankID

```java
public void setRankID(int rankId)
```

Set the logical ID of a distributed model in the cluster.

- Parameters

- `rankId`: rankId the rankId to set.

## getInsertOpConfigPath

```java
public String getInsertOpConfigPath()
```

Get the AIPP configuration file path.

- Returns

  the insertOpConfigPath.

## setInsertOpConfigPath

```java
public void setInsertOpConfigPath(String insertOpConfigPath)
```

Set the AIPP configuration file path.

- Parameters

    - `insertOpConfigPath`: insertOpConfigPath AIPP configuration file path.

## getInputFormat

```java
public String getInputFormat()
```

Get the input format of the model.

- Returns

    the inputFormat.

## setInputFormat

```java
public void setInputFormat(String inputFormat)
```

Set the input format of the model. Optional "NCHW", "NHWC", and "ND"

- Parameters

- `inputFormat`: inputFormat.

## getInputShape

```java
public String getInputShape()
```

Get the input shape of the model.

- Returns

  the inputShape.

## setInputShape

```java
public void setInputShape(String inputShape)
```

Set the input shape of the model. e.g. "input_op_name1:1,2,3,4;input_op_name2:4,3,2,1;".

- Parameters
- `inputShape`: inputShape Model input shape.

## getInputShapeMap

```java
public HashMap<Integer, ArrayList<Integer>> getInputShapeMap()
```

Get the input shape mapping of the model.

- Returns

  the inputShapeMap.

## setInputShapeMap

```java
public void setInputShapeMap(HashMap<Integer, ArrayList<Integer>> inputShapeMap)
```

Model input shape. e.g. {{0, {1,2,3,4}}, {1, {4,3,2,1}}}.

- Parameters
- `inputShapeMap`: inputShapeMap the inputShapeMap to set.

## getDynamicBatchSize

```java
public ArrayList<Integer> getDynamicBatchSize()
```

Get the dynamic batch sizes of the model.

- Returns

  the dynamicBatchSize.

## setDynamicBatchSize

```java
public void setDynamicBatchSize(ArrayList<Integer> dynamicBatchSize)
```

Dynamic batch sizes of model inputs. Ranges from 2 to 100. e.g. {1, 2} means batch size 1 and 2 are configured.

- Parameters
- `dynamicBatchSize`: dynamicBatchSize the dynamicBatchSize to set.

## getDynamicImageSize

```java
public String getDynamicImageSize()
```

Get the dynamic image sizes of the model.

- Returns

  the dynamicImageSize.

## setDynamicImageSize

```java
public void setDynamicImageSize(String dynamicImageSize)
```

Set the dynamic image sizes of the model.

- Parameters
- `dynamicImageSize`: dynamicImageSize the dynamicImageSize to set.

## getOutputType

```java
public int getOutputType()
```

Get the output type of the model.

- Returns

  the outputType.

## setOutputType

```java
public void setOutputType(int outputType)
```

Set the type of model outputs. Can be DataType.kNumberTypeFloat32, DataType.kNumberTypeUInt8, or DataType.kNumberTypeFloat16.

- Parameters
- `outputType`: outputType the outputType to set.

## getPrecisionMode

```java
public String getPrecisionMode()
```

Get the precision mode of the model.

- Returns

  the precisionMode.

## setPrecisionMode

```java
public void setPrecisionMode(String precisionMode)
```

Set the precision mode. Optional "enforce_fp16", "preferred_fp32", "enforce_origin", "enforce_fp32", and "preferred_optimal". "enforce_fp16" is set as default.

- Parameters
- `precisionMode`: precisionMode.

## getOpSelectImplMode

```java
public String getOpSelectImplMode()
```

Get the operator selection implementation mode of the model.

- Returns

  the opSelectImplMode.

## setOpSelectImplMode

```java
public void setOpSelectImplMode(String opSelectImplMode)
```

Set the operator selection implementation mode of the model. Optional "high_performance" and "high_precision". "high_performance" is set as default.

- Parameters
- `opSelectImplMode`: opSelectImplMode.

## getFusionSwitchConfigPath

```java
public String getFusionSwitchConfigPath()
```

Get fusion switch config file path. It controls which fusion passes to be turned off.

- Returns

  the fusionSwitchConfigPath.

## setFusionSwitchConfigPath

```java
public void setFusionSwitchConfigPath(String fusionSwitchConfigPath)
```

Set fusion switch config file path.

- Parameters
- `fusionSwitchConfigPath`: fusionSwitchConfigPath the fusionSwitchConfigPath to set.

## getBufferOptimizeMode

```java
public String getBufferOptimizeMode()
```

Get the buffer optimize mode. Optional "l1_optimize", "l2_optimize", or "off_optimize". "l2_optimize" is set as default.

- Returns

  the bufferOptimizeMode.

## setBufferOptimizeMode

```java
public void setBufferOptimizeMode(String bufferOptimizeMode)
```

Set the buffer optimization mode of the model.

- Parameters
- `bufferOptimizeMode`: bufferOptimizeMode the bufferOptimizeMode to set.