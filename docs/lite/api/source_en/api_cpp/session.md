# mindspore::session

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/api/source_en/api_cpp/session.md)

## LiteSession

\#include &lt;[lite_session.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/lite_session.h)&gt;

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

    STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h).

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

    - `before`: A [**KernelCallBack**](https://www.mindspore.cn/lite/api/en/r1.3/api_cpp/mindspore.html#kernelcallback) function. Define a callback function to be called before running each node.

    - `after`: A [**KernelCallBack**](https://www.mindspore.cn/lite/api/en/r1.3/api_cpp/mindspore.html#kernelcallback) function. Define a callback function to be called after running each node.

- Returns

    STATUS as an error code of running graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h).

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

    STATUS as an error code of resize inputs, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h).

#### Train

```cpp
virtual int Train() = 0;
```

Set model to train mode.

- Returns

    STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h).

#### IsTrain

```cpp
bool IsTrain() { return train_mode_ == true; }
```

Check whether the current model is under the train mode.

- Returns

    Boolean indication if model is in train mode.

#### Eval

```cpp
virtual int Eval() = 0;
```

Set model to eval mode.

- Returns

    STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h).

#### IsEval

```cpp
bool IsEval() { return train_mode_ == false; }
```

Check mode of model.

- Returns

    Boolean indication if model is in eval mode.

#### SetLearningRate

```cpp
virtual int SetLearningRate(float learning_rate) = 0;
```

Set the learning rate for the current model.

- Returns

    0 represents success or -1 in case of error.

#### GetLearningRate

```cpp
virtual float GetLearningRate() = 0;
```

Get the learning rate of the current model.

- Returns

    The learning rate of the current model, default is 0.0.

#### SetupVirtualBatch

```cpp
virtual int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) = 0;
```

Customize the virtual batch size, in order to reduce memory consumption.

- Parameters

    - `virtual_batch_multiplier`: virtual batch number.
    - `lr`: learning rate.
    - `momentum`: momentum.

- Returns

    0 represents success or -1 in case of error.

#### GetPredictions

```cpp
virtual std::vector<tensor::MSTensor *> GetPredictions() const = 0;
```

Get the predicting result of the trained model.

- Returns

    Return the pointer vector of prediction results.

#### Export

```cpp
virtual int (const std::string &file_name, lite::ModelType model_type = lite::MT_TRAIN,
                     lite::QuantizationType quant_type = lite::QT_DEFAULT, lite::FormatType format= lite::FT_FLATBUFFERS) const = 0;
```

Save the trained model into a flatbuffer file.

- Parameters

    - `filename`: Filename of the file to save buffer.
    - `model_type`: Model save Type train or inference.
    - `quant_type`: Quant type of Model.
    - `format`: Model save.

- Returns

    0 represents success or -1 in case of error.

#### GetFeatureMaps

```cpp
 virtual std::vector<tensor::MSTensor *> GetFeatureMaps() const = 0;
```

Get the model feature map.

- Returns

    feature map list

#### UpdateFeatureMaps

```cpp
 virtual int UpdateFeatureMaps(const std::vector<tensor::MSTensor *> &features) = 0;
```

Update model feature map.

- Parameters

    - `features`: new features.

- Returns

   STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h).

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

    - `size`: Define the byte number of model buffer.

    - `context`: Define the context of session to be created.

- Returns

    Pointer that points to MindSpore Lite LiteSession.

## TrainSession

\#include &lt;[train_session.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/train/train_session.h)&gt;

TrainSession defines sessions in MindSpore Lite for compiling Model and training.

### Constructors & Destructors

#### TrainSession

```cpp
TrainSession()
```

Constructor of MindSpore Lite TrainSession using default value for parameters.

#### ~TrainSession

```cpp
~TrainSession()
```

Destructor of MindSpore Lite TrainSession.

### Public Member Functions

#### CreateTransferSession

```cpp
static TrainSession *CreateTransferSession(const std::string &filename_backbone, const std::string &filename_head, const lite::Context *context, bool train_mode = false, const lite::TrainCfg *cfg = nullptr);
```

Static method that creates the object pointer that points to the transfer learning training session.

- Parameters

    - `filename_backbone`: File name of the backbone network.
    - `filename_head`:  File name of the head network.
    - `context`:  Pointer that points to the target session.
    - `train_mode`: Training mode to initialize the Session.
    - `cfg`: Config of train session.

- Returns

    Pointer that points to MindSpore Lite TrainSession.

#### CreateTrainSession

```cpp
static LiteSession *CreateTrainSession(const std::string &filename, const lite::Context *context, bool train_mode = false, const lite::TrainCfg *cfg = nullptr);
```

Static method to create a TrainSession object.

- Parameters

    - `filename`: Train model file name.
    - `context`: Pointer that points to the target session.
    - `train_mode`: Training mode to initialize Session.
    - `cfg`: Config of train session.

- Returns

    Pointer that points to MindSpore Lite TrainSession.

## TrainLoop

\#include &lt;[ltrain_loop.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/train/train_loop.h)&gt;

Inherited from Session and used for reducing the RAM consumption during model training, user can set hyper-parameters and customized data preprocessing function.

### Constructors & Destructors

#### ~TrainLoop

```cpp
virtual ~TrainLoop() = default;
```

Destructor function.

### Public Member Functions

#### CreateTrainLoop

```cpp
static TrainLoop *CreateTrainLoop(session::TrainSession *train_session, lite::Context *context, int batch_size = -1);
```

A static method of creating TrainLoop pointer.

- Parameters

    - `train_session`: Pointer that points to the CreateSession or CreateTransferSession object.
    - `context`: Pointer that points to a context.
    - `batch_size`: Batch size number.

- Returns

    Pointer that points to the TrainLoop object .

#### Reset

```cpp
virtual int Reset() = 0;
```

Reset the epoch to 0.

- Returns

    0 means resetting successfully while -1 means failed.

#### train_session

```cpp
virtual session::TrainSession *train_session() = 0;
```

Get the object of the current TrainSession.

- Returns

    Pointer that points to the object of TrainSession.

#### Init

```cpp
virtual int Init(std::vector<mindspore::session::Metrics *> metrics) = 0;
```

Initialize the model evaluation matrix.

- Parameters

    - `metrics`: Pointer vector of the model evaluating matrix.

- Returns

    0 means initializing successfully while -1 means failed.

#### GetMetrics

```cpp
virtual std::vector<mindspore::session::Metrics *> GetMetrics() = 0;
```

Get the model evaluation matrix.

- Returns

    Pointer vector of the model evaluation matrix.

#### SetKernelCallBack

```cpp
virtual int SetKernelCallBack(const KernelCallBack &before, const KernelCallBack &after) = 0;
```

Set the callback function during training.

- Parameters

    - `before`: Callback pointer before execution.
    - `after`: Callback pointer after execution.

- Returns

    0 means setting successfully while -1 means failed.

#### Train

```cpp
virtual int Train(int epochs, mindspore::dataset::Dataset *dataset, std::vector<TrainLoopCallBack *> cbs, LoadDataFunc load_func = nullptr)= 0;
```

Execute training.

- Parameters

    - `epochs`: Training epoch number.
    - `dataset`: Pointer that points to the MindData object.
    - `cbs`: Object pointer vector.
    - `load_func`: Class template function object.

- Returns

    0 means training successfully while -1 means failed.

#### Eval

```cpp
virtual int Eval(mindspore::dataset::Dataset *dataset, std::vector<TrainLoopCallBack *> cbs, LoadDataFunc load_func = nullptr, int max_steps = INT_MAX) = 0;
```

Execute evaluating.

- Parameters

    - `dataset`: Pointer that points to the DataSet object.
    - `cbs`: Object pointer vector.
    - `load_func`: Class template function object.
    - `max_steps`: Eval epoch number.

- Returns

    0 means evaluating successfully while -1 means failed.

## TrainLoopCallback

\#include &lt;[ltrain_loop_callback.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/train/train_loop_callback.h)&gt;

Execute the callback functions during the model training.

### Constructors & Destructors

#### ~TrainLoopCallback

```cpp
virtual ~TrainLoopCallback() = default;
```

Destructor function.

### Public Member Functions

#### Begin

```cpp
virtual void Begin(const TrainLoopCallBackData &cb_data) {}
```

The method is called once before the network is executed.

- Parameters

    - `cb_data`: cb_data info about current execution.

#### End

```cpp
virtual void End(const TrainLoopCallBackData &cb_data) {}
```

The method is called once after the network executed.

- Parameters

    - `cb_data`: cb_data info about current execution.

#### EpochBegin

```cpp
virtual void EpochBegin(const TrainLoopCallBackData &cb_data) {}
```

The method is called at the beginning of each epoch.

- Parameters

    - `cb_data`: cb_data info about current execution.

#### EpochEnd

```cpp
virtual int EpochEnd(const TrainLoopCallBackData &cb_data) { return RET_CONTINUE; }
```

The method is called at the end of each epoch.

- Parameters

    - `cb_data`: cb_data info about current execution.

- Returns
    STATUS as an error code of compiling graph, STATUS is defined in [errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h).

#### StepBegin

```cpp
virtual void StepBegin(const TrainLoopCallBackData &cb_data) {}
```

The method is called at the beginning of each step.

- Parameters

    - `cb_data`: cb_data info about current execution.

#### StepEnd

```cpp
virtual void StepEnd(const TrainLoopCallBackData &cb_data) {}
```

The method is called after each step has finished.

- Parameters

    - `cb_data`: cb_data info about current execution.

## Metrics

\#include &lt;[metrics.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/train/metrics.h)&gt;

Evaluation metrics of the training model.

### Constructors & Destructors

#### ~Metrics

```cpp
virtual ~Metrics() = default;
```

Destructor function.

### Public Member Functions

#### Clear

```cpp
virtual void Clear() {}
```

Reset the member variables `total_accuracy_` and `total_steps_` to 0.

#### Eval

```cpp
virtual float Eval() {}
```

Evaluation the model.

#### Update

```cpp
virtual void Update(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs) = 0;
```

Update the member variables `total_accuracy_` and `total_steps_`.
