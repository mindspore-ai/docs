# ErrorCode and MetaType

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_cpp/source_en/errorcode_and_metatype.md)

## master

Description of error code and meta type supported in MindSpore Lite.

### ErrorCode

| Definition  | Value | Description |
| ---  | --- | --- |
| RET_OK | 0 | No error occurs. |
| RET_ERROR | -1 | Common error code. |
| RET_NULL_PTR | -2 | NULL pointer returned. |
| RET_PARAM_INVALID | -3 | Invalid parameter. |
| RET_NO_CHANGE | -4 | No change. |
| RET_SUCCESS_EXIT | -5 | No error but exit. |
| RET_MEMORY_FAILED | -6 | Fail to create memory. |
| RET_NOT_SUPPORT | -7 | Fail to support. |
| RET_THREAD_POOL_ERROR | -8 | Error in thread pool. |
| RET_OUT_OF_TENSOR_RANGE | -100 | Failed to check range. |
| RET_INPUT_TENSOR_ERROR | -101 | Failed to check input tensor. |
| RET_REENTRANT_ERROR | -102 | Exist executor running. |
| RET_GRAPH_FILE_ERR | -200 | Failed to verify graph file. |
| RET_NOT_FIND_OP | -300 | Failed to find operator. |
| RET_INVALID_OP_NAME | -301 | Invalid operator name. |
| RET_INVALID_OP_ATTR | -302 | Invalid operator attr. |
| RET_OP_EXECUTE_FAILURE | -303 | Failed to execute the operator. |
| RET_FORMAT_ERR | -400 | Failed to check the tensor format. |
| RET_INFER_ERR | -500 | Failed to infer shape. |
| RET_INFER_INVALID | -501 | Invalid infer shape before runtime. |
| RET_INPUT_PARAM_INVALID | -600 | Invalid input param by user. |

### MetaType

An **enum** type.

| Type Definition | Value | Description |
| --- | --- | --- |
|kObjectTypeString| 12 | Indicating a data type of String. |
|kObjectTypeTensorType| 17 | Indicating a data type of TensorList. |
|kNumberTypeBegin| 29 | The beginning of number type. |
|kNumberTypeBool| 30 | Indicating a data type of bool. |
|kNumberTypeInt| 31 | Indicating a data type of int. |
|kNumberTypeInt8| 32 | Indicating a data type of int8. |
|kNumberTypeInt16| 33 | Indicating a data type of int16. |
|kNumberTypeInt32| 34 | Indicating a data type of int32. |
|kNumberTypeInt64| 35 | Indicating a data type of int64. |
|kNumberTypeUInt| 36 | Indicating a data type of unit. |
|kNumberTypeUInt8| 37 | Indicating a data type of unit8. |
|kNumberTypeUInt16| 38 | Indicating a data type of uint16. |
|kNumberTypeUInt32| 39 | Indicating a data type of uint32. |
|kNumberTypeUInt64| 40 | Indicating a data type of uint64. |
|kNumberTypeFloat| 41 | Indicating a data type of float. |
|kNumberTypeFloat16| 42 | Indicating a data type of float16. |
|kNumberTypeFloat32| 43 | Indicating a data type of float32. |
|kNumberTypeFloat64| 44 | Indicating a data type of float64.|
|kNumberTypeEnd| 45 | The end of number type. |

### Function

```cpp
std::string GetErrorInfo(STATUS error_code)
```

Function to obtain description of errorcode.

- Parameters

    - `error_code`: Define which errorcode info to obtain.

- Returns

    String of errorcode info.
