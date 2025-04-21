# tensor_c

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_c/tensor_c.md)

```C
#include<tensor_c.h>
```

tensor_c.h提供了控制MindSpore MSTensor(后文简称：MSTensor)的接口，借助该接口，用户可以创建、销毁张量，也可以获取或者修改张量的属性。

## 公有函数

| function                                                                                                                                                       |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [MSTensorHandle MSTensorCreate(const char* name, MSDataType type, const int64_t* shape, size_t shape_num, const void* data, size_t data_len)](#mstensorcreate) |
| [void MSTensorDestroy(MSTensorHandle* tensor)](#mstensordestroy)                                                                                               |
| [MSTensorHandle MSTensorClone(MSTensorHandle tensor)](#mstensorclone)                                                                                          |
| [void MSTensorSetName(MSTensorHandle tensor, const char* name)](#mstensorsetname)                                                                              |
| [const char* MSTensorGetName(const MSTensorHandle tensor)](#mstensorgetname)                                                                                   |
| [void MSTensorSetDataType(MSTensorHandle tensor, MSDataType type)](#mstensorsetdatatype)                                                                       |
| [MSDataType MSTensorGetDataType(const MSTensorHandle tensor)](#mstensorgetdatatype)                                                                            |
| [void MSTensorSetShape(MSTensorHandle tensor, const int64_t* shape, size_t shape_num)](#mstensorsetshape)                                                      |
| [const int64_t* MSTensorGetShape(const MSTensorHandle tensor, size_t* shape_num)](#mstensorgetshape)                                                           |
| [void MSTensorSetFormat(MSTensorHandle tensor, MSFormat format)](#mstensorsetformat)                                                                           |
| [MSFormat MSTensorGetFormat(const MSTensorHandle tensor)](#mstensorgetformat)                                                                                  |
| [void MSTensorSetData(MSTensorHandle tensor, void* data)](#mstensorsetdata)                                                                                    |
| [const void* MSTensorGetData(const MSTensorHandle tensor)](#mstensorgetdata)                                                                                   |
| [void* MSTensorGetMutableData(const MSTensorHandle tensor)](#mstensorgetmutabledata)                                                                           |
| [int64_t MSTensorGetElementNum(const MSTensorHandle tensor)](#mstensorgetelementnum)                                                                           |
| [size_t MSTensorGetDataSize(const MSTensorHandle tensor)](#mstensorgetdatasize)                                                                                |

### MSTensorCreate

```C
MSTensorHandle MSTensorCreate(const char* name, MSDataType type, const
                              int64_t* shape, size_t shape_num,
                              const void* data, size_t data_len)
```

创建一个MSTensor。

- 参数

    - `name`: 张量名称。
    - `type`: 张量数据类型。
    - `shape`: 张量的维度数组。
    - `shape_num`: 张量维度数组长度。
    - `data`: 指向数据的指针。
    - `data_len`: 数据的长度。

- 返回值

  指向MSTensor的指针。

### MSTensorDestroy

```c
void MSTensorDestroy(MSTensorHandle* tensor)
```

销毁MSTensor对象。若参数tensor为空或者tensor指向的内存为空则不会做任何操作。

- 参数
    - `tensor`: 指向MSTensor指针的指针。

### MSTensorClone

```c
MSTensorHandle MSTensorClone(MSTensorHandle tensor)
```

深拷贝一个MSTensor。

- 参数
    - `tensor`: 指向待拷贝MSTensor的指针。

- 返回值

  指向新MSTensor的指针。

### MSTensorSetName

```c
void MSTensorSetName(MSTensorHandle tensor, const char* name)
```

设置MSTensor的名称。所有参数不能为空，若为空则不会做任何操作，并在日志中输出错误信息。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `name`: 张量名称。

### MSTensorGetName

```c
const char* MSTensorGetName(const MSTensorHandle tensor)
```

获取MSTensor的名称。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

  MSTensor的名称。

### MSTensorSetDataType

```C
void MSTensorSetDataType(MSTensorHandle tensor, MSDataType type)
```

设置MSTensor的数据类型。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `type`: 张量的数据类型。

### MSTensorGetDataType

```C
MSDataType MSTensorGetDataType(const MSTensorHandle tensor)
```

获取MSTensor的数据类型，具体数据类型见[MSDataType](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_c/data_type_c.html#msdatatype)。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

  MSTensor的数据类型。

### MSTensorSetShape

```C
void MSTensorSetShape(MSTensorHandle tensor, const int64_t* shape, size_t shape_num)
```

设置MSTensor的形状。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `shape`: 维度信息数组。
    - `shape_num`: 维度信息数组的长度。

### MSTensorGetShape

```C
const int64_t* MSTensorGetShape(const MSTensorHandle tensor, size_t* shape_num)
```

获取MSTensor的形状。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `shape_num`: 维度信息数组的长度。

- 返回值

  一个包含MSTensor形状数值的整型数组。

### MSTensorSetFormat

```C
void MSTensorSetFormat(MSTensorHandle tensor, MSFormat format)
```

设置MSTensor的数据排列。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `format`: 张量的数据排列，具体见[MSFormat](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_c/format_c.html#msformat)。

### MSTensorGetFormat

```C
MSFormat MSTensorGetFormat(const MSTensorHandle tensor)
```

获取MSTensor的数据排列。

- 返回值

  张量的数据排列，具体见[MSFormat](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_c/format_c.html#msformat)。

### MSTensorSetData

```C
void MSTensorSetData(MSTensorHandle tensor, void* data)
```

设置MSTensor的数据。

- 参数
    - `tensor`: 指向MSTensor的指针。
    - `data`: 指向数据的指针。

### MSTensorGetData

```C
const void* MSTensorGetData(const MSTensorHandle tensor)
```

获取设置张量的数据。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

  指向MSTensor的数据指针。

### MSTensorGetMutableData

```C
void* MSTensorGetMutableData(const MSTensorHandle tensor)
```

获取可变的MSTensor的数据。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

    指向MSTensor的可变数据的指针。

### MSTensorGetElementNum

```C
int64_t MSTensorGetElementNum(const MSTensorHandle tensor)
```

获取MSTensor的元素个数。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

    MSTensor的元素个数。

### MSTensorGetDataSize

```C
size_t MSTensorGetDataSize(const MSTensorHandle tensor)
```

获取MSTensor中的数据的字节数大小。

- 参数
    - `tensor`: 指向MSTensor的指针。

- 返回值

  MSTensor中的数据的字节数大小。
