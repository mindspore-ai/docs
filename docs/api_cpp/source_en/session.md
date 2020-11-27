# mindspore::session

<a href="https://gitee.com/mindspore/docs/blob/master/docs/api_cpp/source_en/session.md" target="_blank"><img src="./_static/logo_source.png"></a>

\#include &lt;[lite_session.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/lite_session.h)&gt;

## LiteSession

LiteSession defines session in MindSpore Lite for compiling Model and forwarding model.

### Constructors & Destructors

#### LiteSession

```cpp
LiteSession()
```

Constructor of MindSpore Lite LiteSession using default value for parameters.

#### ~LiteSession

```cpp
~LiteSession()
```

Destructor of MindSpore Lite LiteSession.

### Public Member Functions

#### BindThread

```cpp
virtual void BindThread(bool if_bind)
```

Attempt to bind or unbind threads in the thread pool to or from the specified cpu core.

- Parameters

    - `if_bind`: Define whether to bind or unbind threads.

#### CompileGraph

```cpp
virtual int CompileGraph(lite::Model *model)
```

Compile MindSpore Lite model.

> CompileGraph should be called before RunGraph.

- Parameters

    - `model`: Define the model to be compiled.

- Returns

    STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/errorcode.h).

#### GetInputs

```cpp
virtual std::vector <tensor::MSTensor *> GetInputs() const
```

Get input MindSpore Lite MSTensors of model.

- Returns

    The vector of MindSpore Lite MSTensor.

#### GetInputsByName

```cpp
mindspore::tensor::MSTensor *GetInputsByName(const std::string &name) const
```

Get input MindSpore Lite MSTensors of model by tensor name.

- Parameters

    - `name`: Define tensor name.

- Returns

    MindSpore Lite MSTensor.

#### RunGraph

```cpp
virtual int RunGraph(const KernelCallBack &before = nullptr, const KernelCallBack &after = nullptr)
```

Run session with callback.
> RunGraph should be called after CompileGraph.

- Parameters

    - `before`: A [**KernelCallBack**](https://www.mindspore.cn/doc/api_cpp/en/master/mindspore.html#kernelcallback) function. Define a callback function to be called before running each node.

    - `after`: A [**KernelCallBack**](https://www.mindspore.cn/doc/api_cpp/en/master/mindspore.html#kernelcallback) function. Define a callback function to be called after running each node.

- Returns

    STATUS as an error code of running graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/errorcode.h).

#### GetOutputsByNodeName

```cpp
virtual std::vector <tensor::MSTensor *> GetOutputsByNodeName(const std::string &node_name) const
```

Get output MindSpore Lite MSTensors of model by node name.

- Parameters

    - `node_name`: Define node name.

- Returns

    The vector of MindSpore Lite MSTensor.

#### GetOutputs

```cpp
virtual std::unordered_map <std::string, mindspore::tensor::MSTensor *> GetOutputs() const
```

Get output MindSpore Lite MSTensors of model mapped by tensor name.

- Returns

    The map of output tensor name and MindSpore Lite MSTensor.

#### GetOutputTensorNames

```cpp
virtual std::vector <std::string> GetOutputTensorNames() const
```

Get name of output tensors of model compiled by this session.

- Returns

    The vector of string as output tensor names in order.

#### GetOutputByTensorName

```cpp
virtual mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const
```

Get output MindSpore Lite MSTensors of model by tensor name.

- Parameters

    - `tensor_name`: Define tensor name.

- Returns

    Pointer of MindSpore Lite MSTensor.

#### GetOutputByTensorName

```cpp
virtual mindspore::tensor::MSTensor *GetOutputByTensorName(const std::string &tensor_name) const
```

Get output MindSpore Lite MSTensors of model by tensor name.

- Parameters

    - `tensor_name`: Define tensor name.

- Returns

  Pointer of MindSpore Lite MSTensor.

#### Resize

```cpp
virtual int Resize(const std::vector <tensor::MSTensor *> &inputs, const std::vector<std::vector<int>> &dims)

```

Resize inputs shape.

- Parameters

    - `inputs`: Model inputs.
    - `dims`: Define the new inputs shape.

- Returns

    STATUS as an error code of resize inputs, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/master/mindspore/lite/include/errorcode.h).

### Static Public Member Functions

#### CreateSession

```cpp
static LiteSession *CreateSession(const lite::Context *context)
```

Static method to create a LiteSession pointer.

- Parameters

    - `context`: Define the context of session to be created.

- Returns

    Pointer of MindSpore Lite LiteSession.

```cpp
static LiteSession *CreateSession(const char *model_buf, size_t size, const lite::Context *context);
```

Static method to create a LiteSession pointer which has already compiled a model.

- Parameters

    - `model_buf`: Define the buffer read from a model file.

    - `size`: variable. Define bytes number of model buffer.

    - `context`: Define the context of session to be created.

- Returns

    Pointer of MindSpore Lite LiteSession.
