# 错误码及元类型

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_cpp/source_zh_cn/errorcode_and_metatype.md)

## master

以下表格描述了MindSpore Lite中支持的错误码和元类型。

### ErrorCode

| 定义 | 值 | 描述 |
| --- | --- | --- |
| RET_OK | 0 | 执行成功。 |
| RET_ERROR | -1 | 通用错误码。 |
| RET_NULL_PTR | -2 | 返回空指针。 |
| RET_PARAM_INVALID | -3 | 无效参数。 |
| RET_NO_CHANGE | -4 | 无改变。 |
| RET_SUCCESS_EXIT | -5 | 无错误退出。 |
| RET_MEMORY_FAILED | -6 | 创建内存失败。 |
| RET_NOT_SUPPORT | -7 | 尚未支持。 |
| RET_THREAD_POOL_ERROR | -8 | 线程池内部错误。 |
| RET_OUT_OF_TENSOR_RANGE | -100 | 输出检查越界。 |
| RET_INPUT_TENSOR_ERROR | -101 | 输入检查越界。 |
| RET_REENTRANT_ERROR | -102 | 存在运行中的执行器。 |
| RET_GRAPH_FILE_ERR | -200 | 图文件识别失败。 |
| RET_NOT_FIND_OP | -300 | 无法找到算子。 |
| RET_INVALID_OP_NAME | -301 | 无效算子名。 |
| RET_INVALID_OP_ATTR | -302 | 无效算子属性。 |
| RET_OP_EXECUTE_FAILURE | -303 | 算子执行失败。 |
| RET_FORMAT_ERR | -400 | 张量格式检查失败。 |
| RET_INFER_ERR | -500 | 维度推理失败。 |
| RET_INFER_INVALID | -501 | 无效的维度推理。 |
| RET_INPUT_PARAM_INVALID | -600 | 无效的用户输入参数。 |

### MetaType

 **enum**类型变量。

| 类型定义 | 值 | 描述 |
| --- | --- | --- |
|kObjectTypeString| 12 | 表示String数据类型。 |
|kObjectTypeTensorType| 17 | 表示TensorList数据类型。 |
|kNumberTypeBegin| 29 | 表示Number类型的起始。 |
|kNumberTypeBool| 30 | 表示Bool数据类型。 |
|kNumberTypeInt| 31 | 表示Int数据类型。 |
|kNumberTypeInt8| 32 | 表示Int8数据类型。 |
|kNumberTypeInt16| 33 | 表示Int16数据类型。 |
|kNumberTypeInt32| 34 | 表示Int32数据类型。 |
|kNumberTypeInt64| 35 | 表示Int64数据类型。 |
|kNumberTypeUInt| 36 | 表示UInt数据类型。 |
|kNumberTypeUInt8| 37 | 表示UInt8数据类型。 |
|kNumberTypeUInt16| 38 | 表示UInt16数据类型。 |
|kNumberTypeUInt32| 39 | 表示UInt32数据类型。 |
|kNumberTypeUInt64| 40 | 表示UInt64数据类型。 |
|kNumberTypeFloat| 41 | 表示Float数据类型。 |
|kNumberTypeFloat16| 42 | 表示Float16数据类型。 |
|kNumberTypeFloat32| 43 | 表示Float32数据类型。 |
|kNumberTypeFloat64| 44 | 表示Float64数据类型。|
|kNumberTypeEnd| 45 | 表示Number类型的结尾。 |

### 函数接口

```cpp
std::string GetErrorInfo(STATUS error_code)
```

获取错误码描述信息。

- 参数

    - `error_code`: 需获取描述信息的错误码。

- 返回值

    错误码描述信息字符串。
