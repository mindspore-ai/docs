# data_type_c

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_c/data_type_c.md)

```C
#include<data_type_c.h>
```

以下表格描述了MSTensor保存的数据支持的类型。

## MSDataType

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

| 类型定义                     | 值        | 描述                     |
| ---------------------------- | --------- | ------------------------ |
| kMSDataTypeUnknown           | 0         | 表示未知的数据类型。     |
| kMSDataTypeObjectTypeString  | 12        | 表示string数据类型。     |
| kMSDataTypeObjectTypeList    | 13        | 表示list数据类型。       |
| kMSDataTypeObjectTypeTuple   | 14        | 表示tuple数据类型。      |
| kMSDataTypeObjectTypeTensor  | 17        | 表示TensorList数据类型。 |
| kMSDataTypeNumberTypeBegin   | 29        | 表示number类型的起始。   |
| kMSDataTypeNumberTypeBool    | 30        | 表示bool数据类型。       |
| kMSDataTypeNumberTypeInt8    | 32        | 表示int8数据类型。       |
| kMSDataTypeNumberTypeInt16   | 33        | 表示int16数据类型。      |
| kMSDataTypeNumberTypeInt32   | 34        | 表示int32数据类型。      |
| kMSDataTypeNumberTypeInt64   | 35        | 表示int64数据类型。      |
| kMSDataTypeNumberTypeUInt    | 36        | 表示uint数据类型。       |
| kMSDataTypeNumberTypeUInt8   | 37        | 表示uint8数据类型。      |
| kMSDataTypeNumberTypeUInt16  | 38        | 表示uint16数据类型。     |
| kMSDataTypeNumberTypeUInt32  | 39        | 表示uint32数据类型。     |
| kMSDataTypeNumberTypeUInt64  | 40        | 表示uint64数据类型。     |
| kMSDataTypeNumberTypeFloat16 | 42        | 表示float16数据类型。    |
| kMSDataTypeNumberTypeFloat32 | 43        | 表示float32数据类型。    |
| kMSDataTypeNumberTypeFloat64 | 44        | 表示float64数据类型。    |
| kMSDataTypeNumberTypeEnd     | 46        | 表示number类型的结尾。   |
| kMSDataTypeInvalid           | INT32_MAX | 表示无效的数据类型。      |

