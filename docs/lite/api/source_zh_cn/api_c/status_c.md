# status_c

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/api/source_zh_cn/api_c/status_c.md)

```C
#include<status_c.h>
```

status_c.h提供了MindSpore Lite运行时的状态码。

## MSCompCode

MindSpore不同组件的代码。

```C
enum MSCompCode {
  kMSCompCodeCore = 0x00000000u,
  kMSCompCodeMD = 0x10000000u,
  kMSCompCodeME = 0x20000000u,
  kMSCompCodeMC = 0x30000000u,
  kMSCompCodeLite = 0xF0000000u,
};
```

| 定义            | 值          | 描述                             |
| --------------- | ----------- | -------------------------------- |
| kMSCompCodeCore | 0x00000000u | MindSpore Core的代码。           |
| kMSCompCodeMD   | 0x10000000u | MindSpore MindData的代码。       |
| kMSCompCodeME   | 0x20000000u | MindSpore MindExpression的代码。 |
| kMSCompCodeMC   | 0x30000000u | MindSpore的代码。                |
| kMSCompCodeLite | 0xF0000000u | MindSpore Lite的代码。           |

## MSStatus

MindSpore的状态码。

```C
typedef enum MSStatus {
  kMSStatusSuccess = 0,
  kMSStatusCoreFailed = kMSCompCodeCore | 0x1,
  kMSStatusLiteError = kMSCompCodeLite | (0x0FFFFFFF & -1),
  kMSStatusLiteNullptr = kMSCompCodeLite | (0x0FFFFFFF & -2),
  kMSStatusLiteParamInvalid = kMSCompCodeLite | (0x0FFFFFFF & -3),
  kMSStatusLiteNoChange = kMSCompCodeLite | (0x0FFFFFFF & -4),
  kMSStatusLiteSuccessExit = kMSCompCodeLite | (0x0FFFFFFF & -5),
  kMSStatusLiteMemoryFailed = kMSCompCodeLite | (0x0FFFFFFF & -6),
  kMSStatusLiteNotSupport = kMSCompCodeLite | (0x0FFFFFFF & -7),
  kMSStatusLiteThreadPoolError = kMSCompCodeLite | (0x0FFFFFFF & -8),
  kMSStatusLiteUninitializedObj = kMSCompCodeLite | (0x0FFFFFFF & -9),
  kMSStatusLiteFileError = kMSCompCodeLite | (0x0FFFFFFF & -10),
  kMSStatusLiteServiceDeny = kMSCompCodeLite | (0x0FFFFFFF & -11),
  kMSStatusLiteModelRebuild = kMSCompCodeLite | (0x0FFFFFFF & -12),
  kMSStatusLiteOutOfTensorRange = kMSCompCodeLite | (0x0FFFFFFF & -100),
  kMSStatusLiteInputTensorError = kMSCompCodeLite | (0x0FFFFFFF & -101),
  kMSStatusLiteReentrantError = kMSCompCodeLite | (0x0FFFFFFF & -102),
  kMSStatusLiteGraphFileError = kMSCompCodeLite | (0x0FFFFFFF & -200),
  kMSStatusLiteNotFindOp = kMSCompCodeLite | (0x0FFFFFFF & -300),
  kMSStatusLiteInvalidOpName = kMSCompCodeLite | (0x0FFFFFFF & -301),
  kMSStatusLiteInvalidOpAttr = kMSCompCodeLite | (0x0FFFFFFF & -302),
  kMSStatusLiteOpExecuteFailure = kMSCompCodeLite | (0x0FFFFFFF & -303),
  kMSStatusLiteFormatError = kMSCompCodeLite | (0x0FFFFFFF & -400),
  kMSStatusLiteInferError = kMSCompCodeLite | (0x0FFFFFFF & -500),
  kMSStatusLiteInferInvalid = kMSCompCodeLite | (0x0FFFFFFF & -501),
  kMSStatusLiteInputParamInvalid = kMSCompCodeLite | (0x0FFFFFFF & -600),
} MSStatus;
```

| 定义                | 值              | 描述 |
| ------------------- | --------------- | ---- |
| kMSStatusSuccess    | 0               | 通用的成功状态码。 |
| kMSStatusCoreFailed | kMSCompCodeCore \| 0x1 | MindSpore Core 失败状态码。 |
| kMSStatusLiteError | kMSCompCodeLite \| (0x0FFFFFFF & -1) |MindSpore Lite 异常状态码。|
| kMSStatusLiteNullptr | kMSCompCodeLite \| (0x0FFFFFFF & -2) |MindSpore Lite 空指针状态码。|
| kMSStatusLiteParamInvalid | kMSCompCodeLite \| (0x0FFFFFFF & -3) |MindSpore Lite 参数异常状态码。|
| kMSStatusLiteNoChange | kMSCompCodeLite \| (0x0FFFFFFF & -4) |MindSpore Lite 未改变状态码。|
| kMSStatusLiteSuccessExit | kMSCompCodeLite \| (0x0FFFFFFF & -5) |MindSpore Lite 没有错误但是退出的状态码。|
| kMSStatusLiteMemoryFailed | kMSCompCodeLite \| (0x0FFFFFFF & -6) |MindSpore Lite 内存分配失败的状态码。|
| kMSStatusLiteNotSupport | kMSCompCodeLite \| (0x0FFFFFFF & -7) |MindSpore Lite 功能未支持的状态码。|
| kMSStatusLiteThreadPoolError | kMSCompCodeLite \| (0x0FFFFFFF & -8) |MindSpore Lite 线程池异常状态码。|
| kMSStatusLiteUninitializedObj | kMSCompCodeLite \| (0x0FFFFFFF & -9) |MindSpore Lite 未初始化状态码。|
| kMSStatusLiteFileError | kMSCompCodeLite \| (0x0FFFFFFF & -10) |MindSpore Lite 无效文件状态码。|
| kMSStatusLiteServiceDeny | kMSCompCodeLite \| (0x0FFFFFFF & -11) |MindSpore Lite 拒绝服务状态码。|
| kMSStatusLiteModelRebuild | kMSCompCodeLite \| (0x0FFFFFFF & -12) |MindSpore Lite 模型重复构建状态码。|
| kMSStatusLiteOutOfTensorRange | kMSCompCodeLite \| (0x0FFFFFFF & -100) |MindSpore Lite 张量溢出错误的状态码。|
| kMSStatusLiteInputTensorError | kMSCompCodeLite \| (0x0FFFFFFF & -101) |MindSpore Lite 输入张量异常的状态码。|
| kMSStatusLiteReentrantError | kMSCompCodeLite \| (0x0FFFFFFF & -102) |MindSpore Lite 重入异常的状态码。|
| kMSStatusLiteGraphFileError | kMSCompCodeLite \| (0x0FFFFFFF & -200) |MindSpore Lite 文件异常状态码。|
| kMSStatusLiteNotFindOp | kMSCompCodeLite \| (0x0FFFFFFF & -300) |MindSpore Lite 未找到算子的状态码。|
| kMSStatusLiteInvalidOpName | kMSCompCodeLite \| (0x0FFFFFFF & -301) |MindSpore Lite 无效算子状态码。|
| kMSStatusLiteInvalidOpAttr | kMSCompCodeLite \| (0x0FFFFFFF & -302) |MindSpore Lite 无效算子超参数状态码。|
| kMSStatusLiteOpExecuteFailure | kMSCompCodeLite \| (0x0FFFFFFF & -303) |MindSpore Lite 算子执行失败的状态码。|
| kMSStatusLiteFormatError | kMSCompCodeLite \| (0x0FFFFFFF & -400) |MindSpore Lite 张量格式异常状态码。|
| kMSStatusLiteInferError | kMSCompCodeLite \| (0x0FFFFFFF & -500) |MindSpore Lite 形状推理异常状态码。|
| kMSStatusLiteInferInvalid | kMSCompCodeLite \| (0x0FFFFFFF & -501) |MindSpore Lite 无效的形状推理的状态码。|
| kMSStatusLiteInputParamInvalid | kMSCompCodeLite \| (0x0FFFFFFF & -600) |MindSpore Lite 用户输入的参数无效状态码。|

