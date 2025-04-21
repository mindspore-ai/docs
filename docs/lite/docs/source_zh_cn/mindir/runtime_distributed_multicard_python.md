# 使用Python接口执行Ascend后端多卡/多芯推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/runtime_distributed_multicard_python.md)

## 概述

在单机多卡、单卡多芯场景下，为了充分发挥设备性能，需要让芯片或者不同卡直接进行并行推理。在Ascend环境下，这种场景更加常见，例如Atlas 300I Duo推理卡具有单卡双芯的规格，天然更适合并行处理。本教程介绍如何使用[Python接口](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite.html)在Ascend后端环境下执行MindSpore Lite多卡/多芯推理。推理核心流程与[云侧单卡推理](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_python.html)流程大致相同，可以相互参考。

MindSpore Lite云侧分布式推理仅支持在Linux环境部署运行，本教程支持的设备类型为Atlas训练系列产品，并且以 **Atlas 300I Duo推理卡 + 开源Stable Diffusion（SD） ONNX模型** 作为案例。本案例调用Python的multiprocess库创建子进程，用户与主进程进行交互，主进程通过管道与子进程进行交互。

## 准备工作

1. 下载云侧Ascend后端多卡/多芯推理Python[示例代码](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_parallel_python)，后文将该目录称为示例代码目录。

2. 下载MindSpore Lite云侧推理安装包[mindspore-lite-{version}-linux-{arch}.whl](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/use/downloads.html)，存放至示例代码目录，并通过`pip`工具安装。

3. 通过[converter_lite工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter_tool.html)将ONNX模型转换为MindSpore的MINDIR格式模型，其中batch_size的大小设置为原模型的一半，用于分配到双芯或双卡上并行推理，达到与原模型一致的输出结果（若采用更多张卡并行，则需要batch size可以整除所用卡数，转换时设置为整除后的结果）。例如对于batch size = 2的模型，转换时设置其为1，表示双芯并行推理时每个子进程推理单个batch。对于SD2.1模型，可以采用如下的转换命令：

    ```shell
    ./converter_lite --modelFile=model.onnx --fmk=ONNX --outputFile=SD2.1_size512_bs1 --inputShape="sample:1,4,64,64;timestep:1;encoder_hidden_states:1,77,1024" --optimize=ascend_oriented
    ```

更多converter工具的使用方法以及模型转换时可配置的优化点，可参考[模型转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/converter.html)页面。

后续章节将结合代码讲述MindSpore Lite云侧分布式推理主要步骤，参考[示例代码](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_parallel_python)。

## 推理流程

此处的推理流程对应的是示例代码 `mslite_engine.py` 的类 `MSLitePredict`。

### 创建上下文配置

上下文配置保存了所需基本配置参数，主要包括设置设备类型为`Ascend`，以及指定设备ID从而为模型分配执行的芯片或卡。如下示例代码演示如何通过[Context](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context)创建上下文：

```python
# init and set context
context = mslite.Context()
context.target = ["ascend"]
context.ascend.device_id = device_id
```

### 模型创建、加载与编译

与[MindSpore Lite云侧单卡推理](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_python.html)一致，Ascend后端多卡/多芯推理的主入口是[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model)接口，可进行模型加载、编译和执行。创建[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model)并调用[Model.build_from_file](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file)接口来实现模型加载与模型编译，示例代码如下：

```python
# create Model and build Model
model = mslite.Model()
model.build_from_file(mindir_file, mslite.ModelType.MINDIR, context)
```

### 模型输入数据填充

首先，使用[Model.get_inputs](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.get_inputs)方法获取所有输入[Tensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Tensor.html#mindspore_lite.Tensor)，利用相关接口将Host数据填入。示例代码如下：

```python
# set model input
inputs = model.get_inputs()
for i, _input in enumerate(inputs):
    _input.set_data_from_numpy(input_data[i])
```

### 推理执行

调用[Model.predict](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict)接口执行推理，示例代码如下：

```python
# execute inference
outputs = model.predict(inputs)
```

## 多卡/多芯并行

示例代码`parallel_predict_utils.py`采用multiprocessing的Process和Pipe方法来实现子进程的并行启动以及同步等待操作。函数`predict_process_func`定义了子进程执行的动作，其初始化上一节介绍的`MSLitePredict`类，并调用其推理方法来完成模型的推理。主进程所在的类为`ParallelPredictUtils`。具体的并行过程如下。

1. 主进程中调用multiprocessing.Process生成两个子进程，主进程向子进程传入模型路径和 `device_id`，子进程进行模型初始化和编译，完成后通过管道向主进程发送编译成功（`BUILD_FINISH`）/失败（`ERROR`）消息；

2. 收到两个子进程的build成功消息后，主进程通过管道向子进程传入开始推理的消息（`PREDICT`），并传入推理输入数据；

3. 子进程进行并行推理，将推理完成消息（`REPLY`）和推理结果放入管道；

4. 主进程接收管道中两个子进程的推理结果；

5. 重复过程2-4，直到主进程发送退出消息（`EXIT`），子进程收到该消息后中断管道监听，正常退出。

我们可以把上述用于管道通信的消息定义为全局变量：

```python
# define message id
ERROR = 0
PREDICT = 1
REPLY = 2
BUILD_FINISH = 3
EXIT = 4
```

其中主进程初始化代码如下：

```python
# main process init
def __init__(self, model_path, device_num):
    self.model_path = model_path
    self.device_num = device_num
    self.pipe_parent_end = []
    self.pipe_child_end = []
    self.process = []
    for _ in range(device_num):
        pipe_end_0, pipe_end_1 = Pipe()
        self.pipe_parent_end.append(pipe_end_0)
        self.pipe_child_end.append(pipe_end_1)
```

主进程创建、调度子进程示例代码如下：

```python
# create subprocess
process = Process(target=predict_process_func,
                  args=(self.pipe_child_end[device_id], self.model_path, device_id,))
process.start()

# send PREDICT command to subprocess and receive result
result = []
for i in range(self.device_num):
    self.pipe_parent_end[i].send((PREDICT, inputs[i]))

for i in range(self.device_num):
    msg_type, msg = self.pipe_parent_end[i].recv()
    if msg_type == ERROR:
        raise RuntimeError(f"Failed to call predict, exception occur: {msg}")
    assert msg_type == REPLY
    result.append(msg)
return result
```

主进程结束子进程并等待同步示例代码如下：

```python
# finalize subprocess
for i in range(self.device_num):
    self.pipe_parent_end[i].send((EXIT, ""))
for i in range(self.device_num):
    self.process[i].join()
```

子进程初始化加载模型示例代码如下：

```python
# subprocess init and build model using MSLitePredict
try:
    predict_worker = MSLitePredict(model_path, device_id)
    pipe_child_end.send((BUILD_FINISH, ""))
except Exception as e:
    logging.exception(e)
    pipe_child_end.send((ERROR, str(e)))
    raise
```

子进程推理示例代码如下：

```python
# model predict in subprocess
msg_type, msg = pipe_child_end.recv()
if msg_type == EXIT:
    print(f"Receive exit message {msg} and start to exit", flush=True)
    break
if msg_type != PREDICT:
    raise RuntimeError(f"Expect to receive EXIT or PREDICT message for child process!")
inputs = msg
result = predict_worker.run(inputs)
pipe_child_end.send((REPLY, result))
```

## 执行双卡/双芯并行样例

配置示例代码目录下`main.py`中的模型路径`mindir_path`为模型转换得到的mindir路径（参考准备工作一节），执行`main.py`，即可实现单卡双芯/双卡并行推理。样例代码中对输入数据按照batch维度进行了切分，即把输入的batch_size=2的数据切分成两份单batch数据，如下所示：

```python
# data slice
input_data_0 = [sample[0], timestep, encoder_hidden_states[0]]
input_data_1 = [sample[1], timestep, encoder_hidden_states[1]]
input_data_all = [input_data_0, input_data_1]
```

推理完成后，会打印总推理时间`Total predict time = XXX`，单位是秒。若推理成功，整个执行过程会打印如下信息。推理结果保存在变量`predict_result`中，包含两个单batch模型的推理结果，拼接后与一个batch_size=2的模型结果一致。

```shell
Success to build model 0
Success to build model 1
Total predict time =  XXX
Receive exit message  and start to exit
Receive exit message  and start to exit
=========== success ===========
```