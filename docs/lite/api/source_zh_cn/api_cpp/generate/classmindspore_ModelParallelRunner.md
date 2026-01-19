# Class ModelParallelRunner

\#include &lt;[model_parallel_runner.h](https://gitee.com/mindspore/mindspore-lite/blob/master/include/api/model_parallel_runner.h)&gt;

ModelParallelRunner定义了MindSpore的多个Model以及并发策略，便于多个Model的调度与管理。

## 构造函数和析构函数

```cpp
ModelParallelRunner()
~ModelParallelRunner()
```

## 公有成员函数

| 函数                   | 云侧推理是否支持 | 端侧推理是否支持 |
|-------------------------------------------------------------|---------|---------|
| [inline Status Init(const std::string &model_path, const std::shared_ptr\<RunnerConfig\> &runner_config = nullptr)](#init)     |    √    |    ✕    |
| [Status Init(const void *model_data, const size_t data_size, const std::shared_ptr\<RunnerConfig\> &runner_config = nullptr)](#init-1)     |    √    |    ✕    |
| [std::vector\<MSTensor\> GetInputs()](#getinputs)     |    √    |    ✕    |
| [std::vector\<MSTensor\> GetOutputs()](#getoutputs)     |    √    |    ✕    |
| [Status Predict(const std::vector\<MSTensor\> &inputs, std::vector\<MSTensor\> *outputs, const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)](#predict)     |    √    |    ✕    |

### Init

```cpp
inline Status Init(const std::string &model_path, const std::shared_ptr<RunnerConfig> &runner_config = nullptr)
```

根据路径读取加载模型，生成一个或者多个模型，并将所有模型编译至可在Device上运行的状态。该接口支持传入`ms`模型（`converter_lite`工具导出）和`mindir`模型（MindSpore导出或`converter_lite`工具导出），但对`ms`模型的支持，将在未来的迭代中删除，推荐使用`mindir`模型进行推理。当使用`ms`模型进行推理时，请保持模型的后缀名为`.ms`，否则无法识别。

- 参数

    - `model_path`: 模型文件路径。
    - `runner_config`: 一个[RunnerConfig](#runnerconfig)类。定义了并发推理模型的配置参数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Init

```cpp
Status Init(const void *model_data, const size_t data_size, const std::shared_ptr<RunnerConfig> &runner_config = nullptr)
```

根据模型文件数据，生成一个或者多个模型，并将所有模型编译至可在Device上运行的状态。该接口仅支持传入`mindir`模型文件数据。

- 参数

    - `model_data`: 模型文件数据。
    - `data_size`: 模型文件数据大小。
    - `runner_config`: 一个[RunnerConfig](#runnerconfig)类。定义了并发推理模型的配置参数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### Predict

```cpp
Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr)
```

并发推理模型。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。
    - `before`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之前调用的回调函数。
    - `after`: 一个[**MSKernelCallBack**](#mskernelcallback) 结构体。定义了运行每个节点之后调用的回调函数。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### GetInputs

```cpp
std::vector<MSTensor> GetInputs()
```

获取模型所有输入张量。

- 返回值

  包含模型所有输入张量的容器类型变量。

### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs()
```

获取模型所有输出张量。

- 返回值

  包含模型所有输出张量的容器类型变量。
