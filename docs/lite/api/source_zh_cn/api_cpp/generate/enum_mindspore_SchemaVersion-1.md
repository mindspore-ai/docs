# Enum SchemaVersion

\#include &lt;[delegate.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/delegate.h)&gt;

定义了MindSpore Lite执行在线推理时模型文件的版本。

```cpp
typedef enum {
  SCHEMA_INVALID = -1, /**< invalid version */
  SCHEMA_CUR,          /**< current version for ms model defined in model.fbs*/
  SCHEMA_V0,           /**< previous version for ms model defined in model_v0.fbs*/
} SchemaVersion;
```
