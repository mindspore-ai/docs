# Class ModelExecutor

\#include &lt;[multi_model_runner.h](https://atomgit.com/mindspore/mindspore-lite/blob/master/include/api/multi_model_runner.h)&gt;

ModelExecutor定义了对Model的封装，用于调度多个Model的推理。

## 构造函数

```c++
ModelExecutor()
```

```c++
ModelExecutor(const std::vector<std::shared_ptr<ModelImpl>> &models, const std::vector<std::string> &executor_input_names,
                const std::vector<std::string> &executor_output_names,
                const std::vector<std::vector<std::string>> &subgraph_input_names,
                const std::vector<std::vector<MSTensor>> &model_output_tensors)
```

- 参数

    - `models`: 一个由ModelImplPtr组成的向量，用于在ModelExecutor中推理。
    - `executor_input_names`: 由string组成的向量，当前ModelExecutor的输入名。
    - `executor_output_names`: 由string组成的向量，当前ModelExecutor的输出名。
    - `subgraph_input_names`: 由string组成的向量，当前ModelExecutor中所有模型的输入以及输出名。
    - `model_output_tensors`: 由MSTensor组成的向量，模型的输出tensor。

## 析构函数

```c++
~ModelExecutor()
```

## 公有成员函数

| 函数                                                                                                                                                                                                                 | 云侧推理是否支持 | 端侧推理是否支持 |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|---------|
| [Status Predict(const std::vector\<MSTensor\> &inputs, std::vector\<MSTensor\> *outputs)](#predict)     |    √    |    ✕   |
| [std::vector\<MSTensor\> GetInputs() const](#getinputs)     |    √    |    ✕   |
| [std::vector\<MSTensor\> GetOutputs() const](#getoutputs)     |    √    |    ✕   |

### Predict

```cpp
Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs)
```

ModelExecutor的推理接口。

- 参数

    - `inputs`: 模型输入按顺序排列的`vector`。
    - `outputs`: 输出参数，按顺序排列的`vector`的指针，模型输出会按顺序填入该容器。

- 返回值

  状态码类`Status`对象，可以使用其公有函数`StatusCode`或`ToString`函数来获取具体错误码及错误信息。

### GetInputs

```cpp
std::vector<MSTensor> GetInputs() const
```

获取模型所有输入张量。

- 返回值

  包含模型所有输入张量的容器类型变量。

### GetOutputs

```cpp
std::vector<MSTensor> GetOutputs() const
```

获取模型所有输出张量。

- 返回值

  包含模型所有输出张量的容器类型变量。
