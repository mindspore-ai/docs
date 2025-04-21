# mindspore::DataType

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_cpp/mindspore_datatype.md)

以下表格描述了MindSpore MSTensor保存的数据支持的类型。

## DataType

 **enum**类型变量。

| 类型定义 | 值 | 描述 |
| --- | --- | --- |
|kTypeUnknown | 0 | 表示未知的数据类型。 |
|kObjectTypeString| 12 | 表示string数据类型。 |
|kObjectTypeList| 13 | 表示list数据类型。 |
|kObjectTypeTuple| 14 | 表示tuple数据类型。 |
|kObjectTypeTensorType| 17 | 表示TensorList数据类型。 |
|kNumberTypeBegin| 29 | 表示number类型的起始。 |
|kNumberTypeBool| 30 | 表示bool数据类型。 |
|kNumberTypeInt8| 32 | 表示int8数据类型。 |
|kNumberTypeInt16| 33 | 表示int16数据类型。 |
|kNumberTypeInt32| 34 | 表示int32数据类型。 |
|kNumberTypeInt64| 35 | 表示int64数据类型。 |
|kNumberTypeUInt| 36 | 表示uint数据类型。 |
|kNumberTypeUInt8| 37 | 表示uint8数据类型。 |
|kNumberTypeUInt16| 38 | 表示uint16数据类型。 |
|kNumberTypeUInt32| 39 | 表示uint32数据类型。 |
|kNumberTypeUInt64| 40 | 表示uint64数据类型。 |
|kNumberTypeFloat16| 42 | 表示float16数据类型。 |
|kNumberTypeFloat32| 43 | 表示float32数据类型。 |
|kNumberTypeFloat64| 44 | 表示float64数据类型。|
|kNumberTypeEnd| 46 | 表示number类型的结尾。 |
|kInvalidType | INT32_MAX | 表示无效的数据类型。 |
