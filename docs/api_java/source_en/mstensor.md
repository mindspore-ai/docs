# MSTensor

<a href="https://gitee.com/mindspore/docs/blob/r1.2/docs/api_java/source_en/mstensor.md" target="_blank"><img src="./_static/logo_source.png"></a>

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

DataType is defined in [com.mindspore.lite.DataType](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/DataType.java).

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
