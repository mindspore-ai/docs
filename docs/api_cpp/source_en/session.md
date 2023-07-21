# mindspore::session

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/api_cpp/source_en/session.md)

## LiteSession

\#include &lt;[lite_session.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/lite_session.h)&gt;

LiteSession defines sessions in MindSpore Lite for compiling Model and forwarding inference.

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

Attempts to bind threads in the thread pool to the specified CPU core or unbind threads from the core.

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

    STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/errorcode.h).

#### GetInputs

```cpp
virtual std::vector <tensor::MSTensor *> GetInputs() const
```

Get input MindSpore Lite MSTensors of model.

- Returns

    The vector of MindSpore Lite MSTensor.

#### GetInputsByTensorName

```cpp
mindspore::tensor::MSTensor *GetInputsByTensorName(const std::string &name) const
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

    - `before`: A [**KernelCallBack**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/mindspore.html#kernelcallback) function. Define a callback function to be called before running each node.

    - `after`: A [**KernelCallBack**](https://www.mindspore.cn/doc/api_cpp/en/r1.1/mindspore.html#kernelcallback) function. Define a callback function to be called after running each node.

- Returns

    STATUS as an error code of running graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/errorcode.h).

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

Get the MSTensors output of the MindSpore Lite model mapped by tensor name.

- Returns

    The map of output tensor name and MindSpore Lite MSTensor.

#### GetOutputTensorNames

```cpp
virtual std::vector <std::string> GetOutputTensorNames() const
```

Get name of output tensors of model compiled by this session.

- Returns

    A string variable, contains the output tensors’ names in order.

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

    - `dims`: defines the new inputs shape. Its order should be consistent with inputs.

- Returns

    STATUS as an error code of resize inputs, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/errorcode.h).

### Static Public Member Functions

#### CreateSession

```cpp
static LiteSession *CreateSession(const lite::Context *context)
```

Static method to create a LiteSession pointer.

- Parameters

    - `context`: Define the context of session to be created.

- Returns

    Pointer that points to MindSpore Lite MSTensor.

```cpp
static LiteSession *CreateSession(const char *model_buf, size_t size, const lite::Context *context);
```

Static method to create a LiteSession pointer. The returned LiteSession pointer has already read model_buf and completed graph compilation.

- Parameters

    - `model_buf`: Define the buffer read from a model file.

    - `size`: variable. Define the byte number of model buffer.

    - `context`: Define the context of session to be created.

- Returns

    Pointer that points to MindSpore Lite LiteSession.

## TrainSession

\#include &lt;[lite_session.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/lite_session.h)&gt;

Inherited from LiteSession, TrainSession defines the class that allows training the MindSpore model.

### Constructors & Destructors

#### ~TrainSession

```cpp
virtual ~TrainSession() = default;
```

Static method to create a TrainSession object.

### Public Member Functions

#### CreateSession

```cpp
static TrainSession *CreateSession(const char *model_buf, size_t size, lite::Context *context, bool train_mode = false);
```

Static method to create a TrainSession object.

- Parameters

    - `model_buf`: A buffer that was read from a MS model file.

    - `size`: Length of the buffer.

    - `context`: Defines the context of the session to be created.

    - `train_mode`: Training mode to initialize Session with.

- Returns

    Pointer that points to MindSpore Lite TrainSession.

```cpp
static TrainSession *CreateSession(const std::string &filename, lite::Context *context, bool train_mode = false);
```

Static method to create a TrainSession object.

- Parameters

    - `filename`: Filename to read flatbuffer from.

    - `context`: Defines the context of the session to be created.

    - `train_mode`: Training mode to initialize Session with.

- Returns

    Pointer that points to MindSpore Lite TrainSession.

#### ExportToBuf

```cpp
virtual void *ExportToBuf(char *buf, size_t *len) const = 0;
```

Export the trained model into a buffer.

- Parameters

    - `buf`: The buffer to be exported into. If equal to nullptr, `buf` will be allocated.

    - `len`: Size of the pre-allocated buffer, and the returned size of the exported buffer.

- Returns

    Pointer that points to MindSpore Lite TrainSession.

#### SaveToFile

```cpp
virtual int SaveToFile(const std::string &filename) const = 0;
```

Save the trained model into a flatbuffer file.

- Parameters

    - `filename`: Name of the file to save flatbuffer.

- Returns

    0 represents success or -1 in case of error.

#### Train

```cpp
virtual int Train() = 0;
```

Set model to train mode.

- Returns

    STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/errorcode.h)

#### IsTrain

```cpp
bool IsTrain() { return train_mode_ == true; }
```

Checks whether the current model is under the train mode.

- Returns

    Boolean indication if model is in train mode.

#### Eval

```cpp
virtual int Eval() = 0;
```

Set model to eval mode.

- Returns

    STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.1/mindspore/lite/include/errorcode.h).

#### IsEval

```cpp
bool IsEval() { return train_mode_ == false; }
```

Check mode of model.

- Returns

    boolean indication if model is in eval mode.
