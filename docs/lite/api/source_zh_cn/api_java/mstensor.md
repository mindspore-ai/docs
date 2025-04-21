# MSTensor

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_java/mstensor.md)

```java
import com.mindspore.MSTensor;
```

MSTensor定义了MindSpore中的张量。

## 公有成员函数

| function                                   | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------------------------------------ |--------|--------|
| [MSTensor createTensor(String tensorName, int dataType, int[] tensorShape, ByteBuffer buffer)](#createtensor)             | √      | √      |
| [MSTensor createTensor(String tensorName, Object obj)](#createtensor)             | √      | √      |
| [int[] getShape()](#getshape)             | √      | √      |
| [int getDataType()](#getdatatype)        | √      | √      |
| [byte[] getByteData()](#getbytedata)     | √      | √      |
| [float[] getFloatData()](#getfloatdata)  | √      | √      |
| [int[] getIntData()](#getintdata)        | √      | √      |
| [long[] getLongData()](#getlongdata)     | √      | √      |
| [void setData(byte[] data)](#setdata)     | √      | √      |
| [void setData(float[] data)](#setdata)     | √      | √      |
| [void setData(int[] data)](#setdata)     | √      | √      |
| [void setData(long[] data)](#setdata)     | √      | √      |
| [void setData(ByteBuffer data)](#setdata) | √      | √      |
| [long size()](#size)                       | √      | √      |
| [int elementsNum()](#elementsnum)         | √      | √      |
| [void free()](#free)                       | √      | √      |
| [String tensorName()](#tensorname)         | √      | √      |
| [DataType](#datatype)                      | √      | √      |

## createTensor

```java
public static MSTensor createTensor(String tensorName, int dataType, int[] tensorShape, ByteBuffer buffer)
```

生成MindSpore MSTensor。

- 参数

    - `tensorName`: 张量名称。
    - `dataType`: 张量数据类型。
    - `tensorShape`: 张量形状。
    - `buffer`: 张量数据。

- 返回值

  MindSpore MSTensor。

```java
public static MSTensor createTensor(String tensorName, Object obj)
```

生成MindSpore MSTensor。

- 参数

    - `tensorName`: 张量名称。
    - `obj`: java的Array对象或者一个标量值，支持的数据类型：float、double、int、long、boolean。

- 返回值

  MindSpore MSTensor。

## getShape

```java
public int[] getShape()
```

获取MindSpore MSTensor的形状。

- 返回值

  一个包含MindSpore MSTensor形状数值的整型数组。

## getDataType

```java
public int getDataType()
```

DataType在[com.mindspore.DataType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DataType.java)中定义。

- 返回值

  MindSpore MSTensor类的MindSpore DataType。

## getByteData

```java
public byte[] getByteData()
```

获得MSTensor的输出数据，数据类型为byte类型。

- 返回值

  包含所有MSTensor输出数据的byte类型数组。

## getFloatData

```java
public float[] getFloatData()
```

获得MSTensor的输出数据，数据类型为float类型。

- 返回值

  包含所有MSTensor输出数据的float类型数组。

## getIntData

```java
public int[] getIntData()
```

获得MSTensor的输出数据，数据类型为int类型。

- 返回值

  包含所有MSTensor输出数据的int类型数组。

## getLongData

```java
public long[] getLongData()
```

获得MSTensor的输出数据，数据类型为long类型。

- 返回值

  包含所有MSTensor输出数据的long类型数组。

## setData

```java
public void setData(byte[] data)
```

设定MSTensor的输入数据。

- 参数

    - `data`: byte[]类型的输入数据。

```java
public void setData(float[] data)
```

设定MSTensor的输入数据。

- 参数

    - `data`: float[]类型的输入数据。

```java
public void setData(int[] data)
```

设定MSTensor的输入数据。

- 参数

    - `data`: int[]类型的输入数据。

```java
public void setData(long[] data)
```

设定MSTensor的输入数据。

- 参数

    - `data`: long[]类型的输入数据。

```java
public void setData(ByteBuffer data)
```

设定MSTensor的输入数据。

- 参数

    - `data`: ByteBuffer类型的输入数据。

## size

```java
public long size()
```

获取MSTensor中的数据的字节数大小。

- 返回值

  MSTensor中的数据的字节数大小。

## elementsNum

```java
public int elementsNum()
```

获取MSTensor中的元素个数。

- 返回值

  MSTensor中的元素个数。

## free

```java
public void free()
```

释放MSTensor运行过程中动态分配的内存。

## tensorName

```java
public String tensorName()
```

返回tensor的名称。

- 返回值

  tensor的名称。

## DataType

```java
import com.mindspore.config.DataType;
```

DataType定义了MindSpore中的张量的数据类型。

### 公有成员变量

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
