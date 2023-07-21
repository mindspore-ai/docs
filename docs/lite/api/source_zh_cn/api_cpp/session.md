# mindspore::session

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/lite/api/source_zh_cn/api_cpp/session.md)

## LiteSession

\#include &lt;[lite_session.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/lite_session.h)&gt;

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

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

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

    - `before`: 一个[**KernelCallBack**](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/mindspore.html#kernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。

    - `after`: 一个[**KernelCallBack**](https://www.mindspore.cn/lite/api/zh-CN/r1.3/api_cpp/mindspore.html#kernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

    STATUS ，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

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

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### Train

```cpp
virtual int Train() = 0;
```

设置为训练模式。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### IsTrain

```cpp
bool IsTrain() { return train_mode_ == true; }
```

检查当前模型是否为训练模式。

- 返回值

    true 或 false，即当前模型是否为训练模式。

#### Eval

```cpp
virtual int Eval() = 0;
```

设置为验证模式。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### IsEval

```cpp
bool IsEval() { return train_mode_ == false; }
```

检查当前模型是否为验证模式。

- 返回值

    true 或 false，即当前模型是否为验证模式。

#### SetLearningRate

```cpp
virtual int SetLearningRate(float learning_rate) = 0;
```

为当前模型设置学习率。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### GetLearningRate

```cpp
virtual float GetLearningRate() = 0;
```

获取当前模型的学习率。

- 返回值

    当前模型的学习率， 如果未设置优化器则返回0.0。

#### SetupVirtualBatch

```cpp
virtual int SetupVirtualBatch(int virtual_batch_multiplier, float lr = -1.0f, float momentum = -1.0f) = 0;
```

用户自定义虚拟批次数,，用于减少内存消耗。

- 参数

    - `virtual_batch_multiplier`: 自定义虚拟批次数。
    - `lr`: 自定义学习率。
    - `momentum`: 自定义动量。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### GetPredictions

```cpp
virtual std::vector<tensor::MSTensor *> GetPredictions() const = 0;
```

获取训练模型的预测结果。

- 返回值

    预测结果张量指针数组。

#### Export

```cpp
virtual int Export(const std::string &file_name, lite::ModelType model_type = lite::MT_TRAIN,
                     lite::QuantizationType quant_type = lite::QT_DEFAULT, lite::FormatType format= lite::FT_FLATBUFFERS) const = 0;
```

保存已训练模型。

- 参数

    - `filename`: 保存模型的文件名。
    - `model_type`: 训练或推理。
    - `quant_type`: 量化类型。
    - `format`: 保存模型格式。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### GetFeatureMaps

```cpp
 virtual std::vector<tensor::MSTensor *> GetFeatureMaps() const = 0;
```

获取训练模型权重。

- 返回值

    权重列表。

#### UpdateFeatureMaps

```cpp
 virtual int UpdateFeatureMaps(const std::vector<tensor::MSTensor *> &features) = 0;
```

更新训练模型权重。

- 参数

    - `features`: 新的权重列表。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

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

\#include &lt;[train_session.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/train/train_session.h)&gt;

TrainSession定义了MindSpore Lite 训练过程中的会话，用于进行Model的编译和训练。

### 构造函数和析构函数

#### TrainSession

```cpp
TrainSession()
```

MindSpore Lite TrainSession的构造函数，使用默认参数。

#### ~TrainSession

```cpp
~TrainSession()
```

MindSpore Lite TrainSession的析构函数。

### 公有成员函数

#### CreateTransferSession

```cpp
static TrainSession *CreateTransferSession(const std::string &filename_backbone, const std::string &filename_head, const lite::Context *context, bool train_mode = false, const lite::TrainCfg *cfg = nullptr);
```

创建迁移学习训练会话指针的静态方法。

- 参数

    - `filename_backbone`: 主干网络的名称。
    - `filename_head`: 顶层网络的名称。
    - `context`: 指向目标会话的指针。
    - `train_mode`: 是否开启训练模式。
    - `cfg`: 训练相关配置。

- 返回值

    指向训练会话的指针。

#### CreateTrainSession

```cpp
static LiteSession *CreateTrainSession(const std::string &filename, const lite::Context *context, bool train_mode = false, const lite::TrainCfg *cfg = nullptr);
```

创建训练会话指针的静态方法。

- 参数

    - `filename`: 指向文件名称。
    - `context`: 指向会话指针
    - `train_mode`: 是否开启训练模式。
    - `cfg`: 训练相关配置。

- 返回值

    指向训练会话的指针。

## TrainLoop

\#include &lt;[ltrain_loop.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/train/train_loop.h)&gt;

继承于Session，可设置训练参数和数据预处理函数，用于减少模型训练的资源消耗。

### 构造函数和析构函数

#### ~TrainLoop

```cpp
virtual ~TrainLoop() = default;
```

虚析构函数。

### 公有成员函数

#### CreateTrainLoop

```cpp
static TrainLoop *CreateTrainLoop(session::TrainSession *train_session, lite::Context *context, int batch_size = -1);
```

创建迭代训练指针的静态方法。

- 参数

    - `model_filename`: 模型文件名。
    - `context`: 指向目标会话的指针。
    - `batch_size`: 批次数。

- 返回值

    指向迭代训练对象的指针。

#### Reset

```cpp
virtual int Reset() = 0;
```

重置迭代次数为0。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### train_session

```cpp
virtual session::TrainSession *train_session() = 0;
```

获取TrainSession会话对象。

- 返回值

    指向训练会话对象的指针。

#### Init

```cpp
virtual int Init(std::vector<mindspore::session::Metrics *> metrics) = 0;
```

初始化模型评估矩阵。

- 参数

    - `metrics`: 模型评估矩阵指针数组。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### GetMetrics

```cpp
virtual std::vector<mindspore::session::Metrics *> GetMetrics() = 0;
```

获取模型评估矩阵。

- 返回值

    模型评估矩阵指针数组。

#### SetKernelCallBack

```cpp
virtual int SetKernelCallBack(const KernelCallBack &before, const KernelCallBack &after) = 0;
```

设置运行时回调函数。

- 参数

    - `before`: 执行前回调。
    - `after`: 执行后回调。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### Train

```cpp
virtual int Train(int epochs, mindspore::dataset::Dataset *dataset, std::vector<TrainLoopCallBack *> cbs, LoadDataFunc load_func = nullptr)= 0;
```

执行迭代训练。

- 参数

    - `epochs`: 迭代次数。
    - `dataset`: 指向MindData类对象的指针。
    - `cbs`: 对象指针数组。
    - `load_func`: 类模板函数对象。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### Eval

```cpp
virtual int Eval(mindspore::dataset::Dataset *dataset, std::vector<TrainLoopCallBack *> cbs, LoadDataFunc load_func = nullptr, int max_steps = INT_MAX) = 0;
```

执行推理。

- 参数

    - `dataset`: 指向MindData类对象的指针。
    - `cbs`: 对象指针数组。
    - `load_func`: 类模板函数对象。
    - `max_steps`: 重复迭代次数。

- 返回值

    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

## TrainLoopCallback

\#include &lt;[ltrain_loop_callback.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/train/train_loop_callback.h)&gt;

在模型训练中执行回调函数。

### 构造函数和析构函数

#### ~TrainLoopCallback

```cpp
virtual ~TrainLoopCallback() = default;
```

析构函数。

### Public Member Functions

#### Begin

```cpp
virtual void Begin(const TrainLoopCallBackData &cb_data) {}
```

在模型训练前执行。

- 参数

    - `cb_data`: 回调函数对象。

#### End

```cpp
virtual void End(const TrainLoopCallBackData &cb_data) {}
```

在模型训练后执行回调。

- 参数

    - `cb_data`: 回调函数对象。

#### EpochBegin

```cpp
virtual void EpochBegin(const TrainLoopCallBackData &cb_data) {}
```

每次迭代开始前执行回调。

- 参数

    - `cb_data`: 回调函数对象。

#### EpochEnd

```cpp
virtual int EpochEnd(const TrainLoopCallBackData &cb_data) { return RET_CONTINUE; }
```

每次迭代结束后执行回调。

- 参数

    - `cb_data`: 回调函数对象。

- 返回
    STATUS，即编译图的错误码。STATUS在[errorcode.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/errorcode.h)中定义。

#### StepBegin

```cpp
virtual void StepBegin(const TrainLoopCallBackData &cb_data) {}
```

每一步开始前执行回调。

- 参数

    - `cb_data`: 回调函数对象。

#### StepEnd

```cpp
virtual void StepEnd(const TrainLoopCallBackData &cb_data) {}
```

每一步开始后执行回调。

- 参数

    - `cb_data`: 回调函数对象。

## Metrics

\#include &lt;[metrics.h](https://gitee.com/mindspore/mindspore/blob/r1.3/mindspore/lite/include/train/metrics.h)&gt;

训练模型评估矩阵类

### 构造函数和析构函数

#### ~Metrics

```cpp
virtual ~Metrics() = default;
```

析构函数。

### Public Member Functions

#### Clear

```cpp
virtual void Clear() {}
```

将成员变量`total_accuracy_`和`total_steps_`置为零。

#### Eval

```cpp
virtual float Eval() {}
```

评估模型。

#### Update

```cpp
virtual void Update(std::vector<tensor::MSTensor *> inputs, std::vector<tensor::MSTensor *> outputs) = 0;
```

更新成员变量`total_accuracy_`和`total_steps_`的值。
