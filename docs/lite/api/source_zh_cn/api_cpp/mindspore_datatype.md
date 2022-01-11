# mindspore::DataType

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/lite/api/source_zh_cn/api_cpp/mindspore_datatype.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

以下表格描述了MindSpore MSTensor保存的数据支持的类型。

## DataType

 **enum**类型变量。

| 类型定义 | 值 | 描述 |
| --- | --- | --- |
|kTypeUnknown | 0 | 表示未知的数据类型 |
|kObjectTypeString| 12 | 表示String数据类型。 |
|kObjectTypeList| 13 | 表示List数据类型。 |
|kObjectTypeTuple| 14 | 表示Tuple数据类型。 |
|kObjectTypeTensorType| 17 | 表示TensorList数据类型。 |
|kNumberTypeBegin| 29 | 表示Number类型的起始。 |
|kNumberTypeBool| 30 | 表示Bool数据类型。 |
|kNumberTypeInt8| 32 | 表示Int8数据类型。 |
|kNumberTypeInt16| 33 | 表示Int16数据类型。 |
|kNumberTypeInt32| 34 | 表示Int32数据类型。 |
|kNumberTypeInt64| 35 | 表示Int64数据类型。 |
|kNumberTypeUInt| 36 | 表示UInt数据类型。 |
|kNumberTypeUInt8| 37 | 表示UInt8数据类型。 |
|kNumberTypeUInt16| 38 | 表示UInt16数据类型。 |
|kNumberTypeUInt32| 39 | 表示UInt32数据类型。 |
|kNumberTypeUInt64| 40 | 表示UInt64数据类型。 |
|kNumberTypeFloat16| 42 | 表示Float16数据类型。 |
|kNumberTypeFloat32| 43 | 表示Float32数据类型。 |
|kNumberTypeFloat64| 44 | 表示Float64数据类型。|
|kNumberTypeEnd| 46 | 表示Number类型的结尾。 |
|kInvalidType | INT32_MAX | 表示无效的数据类型 |
