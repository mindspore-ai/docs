# data_type_c

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/lite/api/source_zh_cn/api_c/data_type_c.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

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
| kMSDataTypeObjectTypeString  | 12        | 表示String数据类型。     |
| kMSDataTypeObjectTypeList    | 13        | 表示List数据类型。       |
| kMSDataTypeObjectTypeTuple   | 14        | 表示Tuple数据类型。      |
| kMSDataTypeObjectTypeTensor  | 17        | 表示TensorList数据类型。 |
| kMSDataTypeNumberTypeBegin   | 29        | 表示Number类型的起始。   |
| kMSDataTypeNumberTypeBool    | 30        | 表示Bool数据类型。       |
| kMSDataTypeNumberTypeInt8    | 32        | 表示Int8数据类型。       |
| kMSDataTypeNumberTypeInt16   | 33        | 表示Int16数据类型。      |
| kMSDataTypeNumberTypeInt32   | 34        | 表示Int32数据类型。      |
| kMSDataTypeNumberTypeInt64   | 35        | 表示Int64数据类型。      |
| kMSDataTypeNumberTypeUInt    | 36        | 表示UInt数据类型。       |
| kMSDataTypeNumberTypeUInt8   | 37        | 表示UInt8数据类型。      |
| kMSDataTypeNumberTypeUInt16  | 38        | 表示UInt16数据类型。     |
| kMSDataTypeNumberTypeUInt32  | 39        | 表示UInt32数据类型。     |
| kMSDataTypeNumberTypeUInt64  | 40        | 表示UInt64数据类型。     |
| kMSDataTypeNumberTypeFloat16 | 42        | 表示Float16数据类型。    |
| kMSDataTypeNumberTypeFloat32 | 43        | 表示Float32数据类型。    |
| kMSDataTypeNumberTypeFloat64 | 44        | 表示Float64数据类型。    |
| kMSDataTypeNumberTypeEnd     | 46        | 表示Number类型的结尾。   |
| kMSDataTypeInvalid           | INT32_MAX | 表示无效的数据类型。      |

