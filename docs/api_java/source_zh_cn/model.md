# Model

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/api_java/source_zh_cn/model.md)

```java
import com.mindspore.lite.Model;
```

Model定义了MindSpore Lite中的模型，便于计算图管理。

## 公有成员函数

| function                                                     |
| ------------------------------------------------------------ |
| [boolean loadModel(Context context, String modelName)](#loadmodel) |
| [boolean loadModel(String modelPath)](#loadmodel)           |
| [void freeBuffer()](#freebuffer)                            |
| [void free()](#free)                                         |

## loadModel

```java
public boolean loadModel(Context context, String modelName)
```

导入Assets中的MindSpore Lite模型。

- 参数

    - `context`: Android中的Context上下文
    - `modelName`: 模型文件名称

- 返回值

  是否导入成功

```java
public boolean loadModel(String modelPath)
```

导入modelPath中的ms模型。

- 参数

    - `modelPath`: 模型文件路径

- 返回值

  是否导入成功

## freeBuffer

```java
public void freeBuffer()
```

释放MindSpore Lite Model中的MetaGraph，用于减小运行时的内存。释放后该Model就不能再进行图编译了。

## free

```java
public void free()
```

释放Model运行过程中动态分配的内存。
