# MSTensor

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/lite/api/source_zh_cn/api_java/mstensor.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

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
| [String tensorName()](#tensorname)         |
| [DataType](#datatype)                      |

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

DataType在[com.mindspore.lite.DataType](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/java/java/common/src/main/java/com/mindspore/lite/DataType.java)中定义。

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

## tensorName

```java
public String tensorName()
```

返回tensor的名称。

- 返回值

  tensor的名称。

## DataType

```java
import com.mindspore.lite.DataType;
```

DataType定义了MindSpore Lite中的张量的数据类型。

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
