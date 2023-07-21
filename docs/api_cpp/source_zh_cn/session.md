# mindspore::session

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/api_cpp/source_zh_cn/session.md)

\#include &lt;[lite_session.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/lite_session.h)&gt;

## LiteSession

LiteSession定义了MindSpore Lite中的会话，用于进行Model的编译和前向推理。

### 构造函数和析构函数

```cpp
LiteSession()
```

MindSpore Lite LiteSession的构造函数，使用默认参数。

```cpp
~LiteSession()
```

MindSpore Lite LiteSession的析构函数。

### 公有成员函数

```cpp
virtual void BindThread(bool if_bind)
```

尝试将线程池中的线程绑定到指定的cpu内核，或从指定的cpu内核进行解绑。

- 参数

    - `if_bind`: 定义了对线程进行绑定或解绑。

```cpp
virtual int CompileGraph(lite::Model *model)
```

编译MindSpore Lite模型。

> CompileGraph必须在RunGraph方法之前调用。

- 参数

    - `model`: 定义了需要被编译的模型。

- 返回值

    STATUS ，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/errorcode.h)中定义。

```cpp
virtual std::vector <tensor::MSTensor *> GetInputs() const
```

获取MindSpore Lite模型的MSTensors输入。

- 返回值

    MindSpore Lite MSTensor向量。

```cpp
std::vector <tensor::MSTensor *> GetInputsByName(const std::string &node_name) const
```

通过节点名获取MindSpore Lite模型的MSTensors输入。

- 参数

    - `node_name`: 定义了节点名。

- 返回值

    MindSpore Lite MSTensor向量。

```cpp
virtual int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr)
```

运行带有回调函数的会话。
> RunGraph必须在CompileGraph方法之后调用。

- 参数

    - `before`: 一个[**KernelCallBack**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.0/session.html#kernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。

    - `after`: 一个[**KernelCallBack**](https://www.mindspore.cn/doc/api_cpp/zh-CN/r1.0/session.html#kernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

    STATUS ，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/errorcode.h)中定义。

```cpp
virtual std::vector <tensor::MSTensor *> GetOutputsByNodeName(const std::string &node_name) const
```

通过节点名获取MindSpore Lite模型的MSTensors输出。

- 参数

    - `node_name`: 定义了节点名。

- 返回值

    MindSpore Lite MSTensor向量。

```cpp
virtual std::unordered_map <std::string, mindspore::tensor::MSTensor *> GetOutputs() const
```

获取与张量名相关联的MindSpore Lite模型的MSTensors输出。

- 返回值

    包含输出张量名和MindSpore Lite MSTensor的容器类型变量。

```cpp
virtual std::vector <std::string> GetOutputTensorNames() const
```

获取由当前会话所编译的模型的输出张量名。

- 返回值

    字符串向量，其中包含了按顺序排列的输出张量名。

```cpp
virtual mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const
```

通过张量名获取MindSpore Lite模型的MSTensors输出。

- 参数

    - `tensor_name`: 定义了张量名。

- 返回值

    指向MindSpore Lite MSTensor的指针。

```cpp
virtual int Resize(const std::vector <tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims)
```

调整输入的形状。

- 参数

    - `inputs`: 模型对应的所有输入。
    - `dims`: 输入对应的新的shape，顺序注意要与inputs一致。

- 返回值

    STATUS ，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.0/mindspore/lite/include/errorcode.h)中定义。

### 静态公有成员函数

```cpp
static LiteSession *CreateSession(lite::Context *context)
```

用于创建一个LiteSession指针的静态方法。

- 参数

    - `context`: 定义了所要创建的session的上下文。

- 返回值

    指向MindSpore Lite LiteSession的指针。

## KernelCallBack

```cpp
using KernelCallBack = std::function<bool(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs, const CallBackParam &opInfo)>
```

一个函数包装器。KernelCallBack 定义了指向回调函数的指针。

## CallBackParam

一个结构体。CallBackParam定义了回调函数的输入参数。
**属性**

```cpp
name_callback_param
```

**string** 类型变量。节点名参数。

```cpp
type_callback_param
```

**string** 类型变量。节点类型参数。
