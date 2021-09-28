# MSTensor

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/mstensor.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

```java
import com.mindspore.lite.MSTensor;
```

MSTensor defined tensor in MindSpore Lite.

## Public Member Functions

| function                                   |
| ------------------------------------------ |
| [int[] getShape()](#getshape)             |
| [int getDataType()](#getdatatype)        |
| [byte[] getByteData()](#getbytedata)     |
| [float[] getFloatData()](#getfloatdata)  |
| [int[] getIntData()](#getintdata)        |
| [long[] getLongData()](#getlongdata)     |
| [void setData(byte[] data)](#setdata)     |
| [void setData(ByteBuffer data)](#setdata) |
| [long size()](#size)                       |
| [int elementsNum()](#elementsnum)         |
| [void free()](#free)                       |
| [String tensorName()](#tensorname)         |
| [DataType](#datatype)                      |

## getShape

```java
public int[] getShape()
```

Get the shape of the MindSpore Lite MSTensor.

- Returns

  A array of int as the shape of the MindSpore Lite MSTensor.

## getDataType

```java
public int getDataType()
```

DataType is defined in [com.mindspore.lite.DataType](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/DataType.java).

- Returns

  The MindSpore Lite data type of the MindSpore Lite MSTensor class.

## getByteData

```java
public byte[] getByteData()
```

Get output data of MSTensor, the data type is byte.

- Returns

  The byte array containing all MSTensor output data.

## getFloatData

```java
public float[] getFloatData()
```

Get output data of MSTensor, the data type is float.

- Returns

  The float array containing all MSTensor output data.

## getIntData

```java
public int[] getIntData()
```

Get output data of MSTensor, the data type is int.

- Returns

  The int array containing all MSTensor output data.

## getLongData

```java
public long[] getLongData()
```

Get output data of MSTensor, the data type is long.

- Returns

  The long array containing all MSTensor output data.

## setData

```java
public void setData(byte[] data)
```

Set the input data of MSTensor.

- Parameters

    - `data`: Input data of byte[] type.

```java
public void setData(ByteBuffer data)
```

Set the input data of MSTensor.

- Parameters

    - `data`: Input data of ByteBuffer type.

## size

```java
public long size()
```

Get the size of the data in MSTensor in bytes.

- Returns

  The size of the data in MSTensor in bytes.

## elementsNum

```java
public int elementsNum()
```

Get the number of elements in MSTensor.

- Returns

  The number of elements in MSTensor.

## free

```java
public void free()
```

Free all temporary memory in MindSpore Lite MSTensor.

## tensorName

```java
public String tensorName()
```

Get tensor name.

- Returns

  Tensor name.

## DataType

```java
import com.mindspore.lite.DataType;
```

Define tensor data type.

### Public member variable

```java
public static final int kNumberTypeBool = 30;
public static final int kNumberTypeInt = 31;
public static final int kNumberTypeInt8 = 32;
public static final int kNumberTypeInt16 = 33;
public static final int kNumberTypeInt32 = 34;
public static final int kNumberTypeInt64 = 35;
public static final int kNumberTypeUInt = 36;
public static final int kNumberTypeUInt8 = 37;
public static final int kNumberTypeUInt16 = 38;
public static final int kNumberTypeUint32 = 39;
public static final int kNumberTypeUInt64 = 40;
public static final int kNumberTypeFloat = 41;
public static final int kNumberTypeFloat16 = 42;
public static final int kNumberTypeFloat32 = 43;
public static final int kNumberTypeFloat64 = 44;
```
