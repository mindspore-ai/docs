# MSTensor

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_java/source_zh_cn/mstensor.md)

```java
import com.mindspore.lite.MSTensor;
```

MSTensor定义了MindSpore Lite中的张量。

## 公有成员函数

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

获取MindSpore Lite MSTensor的形状。

- 返回值

  一个包含MindSpore Lite MSTensor形状数值的整型数组。

## getDataType

```java
public int getDataType()
```

DataType在[com.mindspore.lite.DataType](https://gitee.com/mindspore/mindspore/blob/r1.2/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/DataType.java)中定义。

- 返回值

  MindSpore Lite MSTensor类的MindSpore Lite DataType。

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
