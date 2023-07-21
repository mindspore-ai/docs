# mindspore::session

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_zh_cn/session.md)

## LiteSession

\#include &lt;[lite_session.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/lite_session.h)&gt;

LiteSession定义了MindSpore Lite中的会话，用于进行Model的编译和前向推理。

### 构造函数和析构函数

#### LiteSession

```cpp
LiteSession()
```

MindSpore Lite LiteSession的构造函数，使用默认参数。

#### ~LiteSession

```cpp
~LiteSession()
```

MindSpore Lite LiteSession的析构函数。

### 公有成员函数

#### BindThread

```cpp
virtual void BindThread(bool if_bind)
```

尝试将线程池中的线程绑定到指定的cpu内核，或从指定的cpu内核进行解绑。

- 参数

    - `if_bind`: 定义了对线程进行绑定或解绑。

#### CompileGraph

```cpp
virtual int CompileGraph(lite::Model *model)
```

编译MindSpore Lite模型。

> CompileGraph必须在RunGraph方法之前调用。

- 参数

    - `model`: 定义了需要被编译的模型。  

- 返回值

    STATUS ，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/errorcode.h)中定义。

#### GetInputs

```cpp
virtual std::vector <tensor::MSTensor *> GetInputs() const
```

获取MindSpore Lite模型的MSTensors输入。

- 返回值

    MindSpore Lite MSTensor向量。

#### GetInputsByTensorName

```cpp
mindspore::tensor::MSTensor *GetInputsByTensorName(const std::string &name) const
```

通过tensor名获取MindSpore Lite模型的MSTensors输入。

- 参数

    - `name`: 定义了tensor名。

- 返回值

    MindSpore Lite MSTensor。

#### RunGraph

```cpp
virtual int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr)
```

运行带有回调函数的会话。
> RunGraph必须在CompileGraph方法之后调用。

- 参数

    - `before`: 一个[**KernelCallBack**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/mindspore.html#kernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。

    - `after`: 一个[**KernelCallBack**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.1/mindspore.html#kernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

    STATUS ，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/errorcode.h)中定义。

#### GetOutputsByNodeName

```cpp
virtual std::vector <tensor::MSTensor *> GetOutputsByNodeName(const std::string &node_name) const
```

通过节点名获取MindSpore Lite模型的MSTensors输出。

- 参数

    - `node_name`: 定义了节点名。

- 返回值

    MindSpore Lite MSTensor向量。

#### GetOutputs

```cpp
virtual std::unordered_map <std::string, mindspore::tensor::MSTensor *> GetOutputs() const
```

获取与张量名相关联的MindSpore Lite模型的MSTensors输出。

- 返回值

    包含输出张量名和MindSpore Lite MSTensor的容器类型变量。

#### GetOutputTensorNames

```cpp
virtual std::vector <std::string> GetOutputTensorNames() const
```

获取由当前会话所编译的模型的输出张量名。

- 返回值

    字符串向量，其中包含了按顺序排列的输出张量名。

#### GetOutputByTensorName

```cpp
virtual mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const
```

通过张量名获取MindSpore Lite模型的MSTensors输出。

- 参数

    - `tensor_name`: 定义了张量名。

- 返回值

    指向MindSpore Lite MSTensor的指针。

#### Resize

```cpp
virtual int Resize(const std::vector <tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims)
```

调整输入的形状。

- 参数

    - `inputs`: 模型对应的所有输入。
    - `dims`: 输入对应的新的shape，顺序注意要与inputs一致。

- 返回值

    STATUS ，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/errorcode.h)中定义。

### 静态公有成员函数

#### CreateSession

```cpp
static LiteSession *CreateSession(const lite::Context *context)
```

用于创建一个LiteSession指针的静态方法。

- 参数

    - `context`: 定义了所要创建的session的上下文。

- 返回值

    指向MindSpore Lite LiteSession的指针。

```cpp
static LiteSession *CreateSession(const char *model_buf, size_t size, const lite::Context *context);
```

用于创建一个LiteSession指针的静态方法。返回的Lite Session指针已经完成了model_buf的读入和图编译。

- 参数

    - `model_buf`: 定义了读取模型文件的缓存区。

    - `size`: 定义了模型缓存区的字节数。

    - `context`: 定义了所要创建的session的上下文。

- 返回值

    指向MindSpore Lite LiteSession的指针。

## TrainSession

\#include &lt;[lite_session.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/lite_session.h)&gt;

继承于类 LiteSession，用于训练模型。

### 析构函数

#### ~TrainSession

```cpp
virtual ~TrainSession() = default;
```

虚析构函数。

### 公有成员函数

#### CreateSession

```cpp
static TrainSession *CreateSession(const char *model_buf, size_t size, lite::Context *context, bool train_mode = false);
```

基于已有MS模型创建一个用于训练会话的静态对象指针。

- 参数

    - `model_buf`: 指向包含模型文件的缓冲区指针。
    - `size`: 缓冲区长度。
    - `context`: 指向目标会话的指针。
    - `train_mode`: 训练模式，默认值为false。

- 返回值

    返回一个指向训练会话的静态对象指针。

```cpp
static TrainSession *CreateSession(const std::string &filename, lite::Context *context, bool train_mode = false);
```

基于已有模型创建一个用于训练会话的静态对象指针。

- 参数

    - `filename`: 指向文件名称。
    - `context`: 指向会话指针。
    - `train_mode`: 训练模式。

- 返回值

    返回一个指向训练会话的静态对象指针。

#### ExportToBuf

```cpp
virtual void *ExportToBuf(char *buf, size_t *len) const = 0;
```

创建一个指向缓冲区的字符指针。

- 参数

    - `buf`: 指向模型导出的目标缓冲区的指针，如果指针为空则自动分配一块内存。
    - `len`: 指向预分配缓冲区大小的指针。

- 返回值

    返回一个指向存储导出模型缓冲区的字符指针。

#### SaveToFile

```cpp
virtual int SaveToFile(const std::string &filename) const = 0;
```

保存已训练模型。

- 参数

    - `filename`: 已训练模型的文件名。

- 返回值

    0 表示保存成功，-1 表示保存失败。

#### Train

```cpp
virtual int Train() = 0;
```

设置为训练模式。

- 返回值

    返回执行结果状态代码，状态码参见 " errorcode.h "。

#### IsTrain

```cpp
bool IsTrain() { return train_mode_ == true; }
```

检查当前模型是否为训练模式。

- 返回值

    返回 true 或 false，即当前模型是否为训练模式。

#### Eval

```cpp
virtual int Eval() = 0;
```

设置为验证模式。

- 返回值

    返回执行结果状态代码，状态码参见 " errorcode.h "。

#### IsEval

```cpp
bool IsEval() { return train_mode_ == false; }
```

检查当前模型是否为验证模式。

- 返回值

    返回 true 或 false，即当前模型是否为验证模式。
