# tensor_c

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_c/tensor_c.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

```C
#include<tensor_c.h>
```

tensor_c.h提供了控制张量的接口，借助该接口，用户可以创建、销毁张量，也可以获取或者修改张量的属性。

## 公有函数

| function                                                                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MSTensorHandle MSTensorCreate(const char *name, MSDataType type, const int64_t *shape, size_t shape_num, const void *data, size_t data_len)](#mstensorcreate) |
| [void MSTensorDestroy(MSTensorHandle *tensor)](#mstensordestroy)                                                                                               |
| [MSTensorHandle MSTensorClone(MSTensorHandle tensor)](#mstensorclone)                                                                                          |
| [void MSTensorSetName(MSTensorHandle tensor, const char *name)](#mstensorsetname)                                                                              |
| [const char *MSTensorGetName(const MSTensorHandle tensor)](#mstensorgetname)                                                                                   |
| [void MSTensorSetDataType(MSTensorHandle tensor, MSDataType type)](#mstensorsetdatatype)                                                                       |
| [MSDataType MSTensorGetDataType(const MSTensorHandle tensor)](#mstensorgetdatatype)                                                                            |
| [void MSTensorSetShape(MSTensorHandle tensor, const int64_t *shape, size_t shape_num)](#mstensorsetshape)                                                      |
| [const int64_t *MSTensorGetShape(const MSTensorHandle tensor, size_t *shape_num)](#mstensorgetshape)                                                           |
| [void MSTensorSetFormat(MSTensorHandle tensor, MSFormat format)](#mstensorsetformat)                                                                           |
| [MSFormat MSTensorGetFormat(const MSTensorHandle tensor)](#mstensorgetformat)                                                                                  |
| [void MSTensorSetData(MSTensorHandle tensor, void *data)](#mstensorsetdata)                                                                                    |
| [const void *MSTensorGetData(const MSTensorHandle tensor)](#mstensorgetdata)                                                                                   |
| [void *MSTensorGetMutableData(const MSTensorHandle tensor)](#mstensorgetmutabledata)                                                                           |
| [int64_t MSTensorGetElementNum(const MSTensorHandle tensor)](#mstensorgetelementnum)                                                                           |
| [size_t MSTensorGetDataSize(const MSTensorHandle tensor)](#mstensorgetdatasize)                                                                                |

### 公有函数

#### MSTensorCreate

```C
MSTensorHandle MSTensorCreate(const char *name, MSDataType type, const
                              int64_t *shape, size_t shape_num,
                              const void *data, size_t data_len)
```

生成MindSpore MSTensor。

- 参数

    - `name`: 张量名称。
    - `type`: 张量数据类型。
    - `shape`: 张量的维度数组。
    - `shape_num`: 张量维度数组长度。
    - `data`: 指向数据的指针。
    - `data_len`: 数据的长度。

- 返回值

  指向MindSpore MSTensor(后文简称：MSTensor)的指针。

#### MSTensorDestroy

```c
void MSTensorDestroy(MSTensorHandle *tensor)
```

销毁MSTensor对象。若参数tensor为空或者tensor指向的内存为空则不会做任何操作。

- 参数
    - `tensor`: 指向MSTensor指针的指针。

#### MSTensorClone

```c
MSTensorHandle MSTensorClone(MSTensorHandle tensor)
```

深拷贝一个MSTensor。

- 参数
    - `tensor`: 指向待拷贝MSTensor的指针。

- 返回值

  指向新MSTensor的指针。

#### MSTensorSetName

```c
void MSTensorSetName(MSTensorHandle tensor, const char *name)
```

设置MSTensor的名称。所有参数参数不能为空，若为空则不会做任何操作，并在日志中输出错误信息。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `name`: 张量名称。

#### MSTensorGetName

```c
const char *MSTensorGetName(const MSTensorHandle tensor)
```

获取MSTensor的名称。

- 参数
    - `tensor`: 指向MSTensor的指针。

#### MSTensorSetDataType

```C
void MSTensorSetDataType(MSTensorHandle tensor, MSDataType type)
```

设置MSTensor的数据类型。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `type`: 张量的数据类型。

获取MindSpore MSTensor的形状。

- 返回值

  一个包含MindSpore MSTensor形状数值的整型数组。

#### MSTensorGetDataType

```C
MSDataType MSTensorGetDataType(const MSTensorHandle tensor)
```

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

  MindSpore MSTensor类的MindSpore DataType。

#### MSTensorSetShape

```C
void MSTensorSetShape(MSTensorHandle tensor, const int64_t *shape, size_t shape_num)
```

获取MindSpore MSTensor的形状。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `shape`: 维度信息数组。
    - `shape_num`: 维度信息数组的长度。

#### MSTensorGetShape

```C
const int64_t *MSTensorGetShape(const MSTensorHandle tensor, size_t *shape_num)
```

获取MindSpore MSTensor的形状。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `shape_num`: 维度信息数组的长度。

- 返回值

  一个包含MindSpore MSTensor形状数值的整型数组。

#### MSTensorSetFormat

```C
void MSTensorSetFormat(MSTensorHandle tensor, MSFormat format)
```

获取MindSpore MSTensor的数据排列。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `format`: 张量的数据排列，具体见[MSFormat](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/format_c.html#msformat)。

- 返回值

  一个包含MindSpore MSTensor形状数值的整型数组。

#### MSTensorGetFormat

```C
MSFormat MSTensorGetFormat(const MSTensorHandle tensor)
```

获取张量的数据排列。

- 返回值

  张量的数据排列，具体见[MSFormat](https://www.mindspore.cn/lite/api/zh-CN/master/api_c/format_c.html#msformat)。

#### MSTensorSetData

```C
void MSTensorSetData(MSTensorHandle tensor, void *data)
```

设置张量的数据。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `data`: 指向数据的指针。

#### MSTensorGetData

```C
const void *MSTensorGetData(const MSTensorHandle tensor)
```

获取设置张量的数据。

- 返回值

  MSTensor的数据指针。

#### MSTensorGetMutableData

```C
void *MSTensorGetMutableData(const MSTensorHandle tensor)
```

获取可变的MsTensor的数据。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

    MSTensor的数据指针。

#### MSTensorGetElementNum

```C
int64_t MSTensorGetElementNum(const MSTensorHandle tensor)
```

获取MSTensor的元素个数。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

    MSTensor的元素个数。

#### MSTensorGetDataSize

```C
size_t MSTensorGetDataSize(const MSTensorHandle tensor)
```

获取MSTensor中的数据的字节数大小。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

  MSTensor中的数据的字节数大小。

#### DataType

```C
typedef enum MSDataType {
  kMSDataTypeUnknown = 0,
  kMSDataTypeObjectTypeString = 12,
  kMSDataTypeObjectTypeList = 13,
  kMSDataTypeObjectTypeTuple = 14,
  kMSDataTypeObjectTypeTensor = 17,
  kMSDataTypeNumberTypeBegin = 29,
  kMSDataTypeNumberTypeBool = 30,
  kMSDataTypeNumberTypeInt8 = 32,
  kMSDataTypeNumberTypeInt16 = 33,
  kMSDataTypeNumberTypeInt32 = 34,
  kMSDataTypeNumberTypeInt64 = 35,
  kMSDataTypeNumberTypeUInt8 = 37,
  kMSDataTypeNumberTypeUInt16 = 38,
  kMSDataTypeNumberTypeUInt32 = 39,
  kMSDataTypeNumberTypeUInt64 = 40,
  kMSDataTypeNumberTypeFloat16 = 42,
  kMSDataTypeNumberTypeFloat32 = 43,
  kMSDataTypeNumberTypeFloat64 = 44,
  kMSDataTypeNumberTypeEnd = 46,
  kMSDataTypeInvalid = INT32_MAX,
} MSDataType;
```

以下表格描述了MindSpore MSTensor保存的数据支持的类型。

 **enum**类型变量。

| 类型定义              | 值        | 描述                     |
| --------------------- | --------- | ------------------------ |
| kTypeUnknown          | 0         | 表示未知的数据类型。       |
| kObjectTypeString     | 12        | 表示String数据类型。     |
| kObjectTypeList       | 13        | 表示List数据类型。       |
| kObjectTypeTuple      | 14        | 表示Tuple数据类型。      |
| kObjectTypeTensorType | 17        | 表示TensorList数据类型。 |
| kNumberTypeBegin      | 29        | 表示Number类型的起始。   |
| kNumberTypeBool       | 30        | 表示Bool数据类型。       |
| kNumberTypeInt8       | 32        | 表示Int8数据类型。       |
| kNumberTypeInt16      | 33        | 表示Int16数据类型。      |
| kNumberTypeInt32      | 34        | 表示Int32数据类型。      |
| kNumberTypeInt64      | 35        | 表示Int64数据类型。      |
| kNumberTypeUInt       | 36        | 表示UInt数据类型。       |
| kNumberTypeUInt8      | 37        | 表示UInt8数据类型。      |
| kNumberTypeUInt16     | 38        | 表示UInt16数据类型。     |
| kNumberTypeUInt32     | 39        | 表示UInt32数据类型。     |
| kNumberTypeUInt64     | 40        | 表示UInt64数据类型。     |
| kNumberTypeFloat16    | 42        | 表示Float16数据类型。    |
| kNumberTypeFloat32    | 43        | 表示Float32数据类型。    |
| kNumberTypeFloat64    | 44        | 表示Float64数据类型。    |
| kNumberTypeEnd        | 46        | 表示Number类型的结尾。   |
| kInvalidType          | INT32_MAX | 表示无效的数据类型。     |

