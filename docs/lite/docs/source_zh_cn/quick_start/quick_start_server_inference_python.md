# 体验Python极简并发推理Demo

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/lite/docs/source_zh_cn/quick_start/quick_start_server_inference_python.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

本教程提供了MindSpore Lite执行并发推理的示例程序。通过创建并发推理配置，并发Runner加载与编译，设置并发推理任务，执行并发推理的方式，演示了[Python接口](https://mindspore.cn/lite/api/zh-CN/r2.0.0-alpha/mindspore_lite.html)进行服务端并发推理的基本流程，用户能够快速了解MindSpore Lite执行并发推理相关API的使用。相关代码放置在[mindspore/lite/examples/quick_start_server_inference_python](https://gitee.com/mindspore/mindspore/tree/r2.0.0-alpha/mindspore/lite/examples/quick_start_server_inference_python)目录。

本教程模拟的使用场景为：当服务器同时收到多台客户端请求的推理任务时，利用支持并发推理的[ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/r2.0.0-alpha/mindspore_lite/mindspore_lite.ModelParallelRunner.html)接口，同时进行多个推理任务的推理，并将推理结果返还给客户端。

下面以Ubuntu 18.04为例，介绍了在Linux X86操作系统配合CPU硬件平台下如何使用Python极简并发推理Demo：

- 一键安装并发推理相关模型文件、MindSpore Lite及其所需的依赖，详情参见[一键安装](#一键安装)小节。

- 执行Python极简并发推理Demo，详情参见[执行Demo](#执行demo)小节。

- Python极简并发推理Demo内容说明，详情参见[Demo内容说明](#demo内容说明)小节。

## 一键安装

本环节以全新的Ubuntu 18.04为例，介绍在CPU环境的Linux-x86_64系统上，通过pip安装Python3.7版本的MindSpore Lite。

进入到[mindspore/lite/examples/quick_start_server_inference_python](https://gitee.com/mindspore/mindspore/tree/r2.0.0-alpha/mindspore/lite/examples/quick_start_server_inference_python)目录下，以安装1.9.0版本的MindSpore Lite为例，执行`lite-server-cpu-pip.sh`脚本进行一键式安装。安装脚本会下载并发推理所需的模型和输入数据文件、安装MindSpore_Lite所需的依赖，以及下载并安装支持并发推理功能的MindSpore Lite。

注：此命令可设置安装的MindSpore Lite版本，由于从MindSpore Lite 1.8.0版本开始支持Python接口，因此版本不能设置低于1.8.0，可设置的版本详情参见[下载MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/r2.0.0-alpha/use/downloads.html)提供的版本。

  ```bash
  MINDSPORE_LITE_VERSION=1.9.0 bash ./lite-server-cpu-pip.sh
  ```

> 若MobileNetV2模型下载失败，请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_server_inference_python/model`目录。
>
> 若input.bin输入数据文件下载失败，请手动下载相关输入数据文件[input.bin](https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin)，并将其拷贝到`mindspore/lite/examples/quick_start_server_inference_python/model`目录。
>
> 若使用脚本下载MindSpore Lite并发推理框架失败，请手动下载对应硬件平台为CPU、操作系统为Linux-x86_64的[MindSpore Lite 并发模型推理框架](https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/lite/release/linux/x86_64/server/mindspore_lite-1.9.0-cp37-cp37m-linux_x86_64.whl)或对应硬件平台为CPU、操作系统为Linux-aarch64的[MindSpore Lite 并发模型推理框架](https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0/MindSpore/lite/release/linux/aarch64/server/mindspore_lite-1.9.0-cp37-cp37m-linux_aarch64.whl)，用户可以使用`uname -m`命令在终端上查询操作系统。并将其拷贝到`mindspore/lite/examples/quick_start_server_inference_python`目录下。
>
> 若需要使用Python3.7以上版本对应的MindSpore Lite，请在本地[编译](https://www.mindspore.cn/lite/docs/zh-CN/r2.0.0-alpha/use/build.html)，注意Python API模块编译依赖：Python >= 3.7.0、NumPy >= 1.17.0、wheel >= 0.32.0。注意要生成支持并发推理的MindSpore Lite安装包，编译前需要配置环境变量：export MSLITE_ENABLE_SERVER_INFERENCE=on。编译成功后，将`output/`目录下生成的Whl安装包拷贝到`mindspore/lite/examples/quick_start_server_inference_python`目录下。
>
> 若`mindspore/lite/examples/quick_start_server_inference_python`目录下不存在MindSpore Lite安装包，则一键安装脚本将会卸载当前已安装的MindSpore Lite后，从华为镜像下载并安装MindSpore Lite。否则，若目录下存在MindSpore Lite安装包，则会优先安装该安装包。
>
> 通过手动下载并且将文件放到指定位置后，需要再次执行lite-server-cpu-pip.sh脚本才能完成一键安装。

执行成功将会显示如下结果，模型文件和输入数据文件可在`mindspore/lite/examples/quick_start_server_inference_python/model`目录下找到。

```text
Successfully installed mindspore-lite-1.9.0
```

## 执行Demo

一键安装后，进入[mindspore/lite/examples/quick_start_server_inference_python](https://gitee.com/mindspore/mindspore/tree/r2.0.0-alpha/mindspore/lite/examples/quick_start_server_inference_python)目录，并执行以下命令，体验MindSpore Lite并发推理MobileNetV2模型。

```bash
python quick_start_server_inferece_python.py
```

执行完成后将能得到如下结果。在多线程执行并发推理任务过程中，打印并发推理中的单次推理耗时和推理结果，结束线程后打印并发推理总耗时。

推理结果说明：

- 模拟5台客户端同时向服务器发送并发推理任务的请求，`parallel id`表示客户端id。

- 模拟每个客户端向服务器发送2个推理任务的请求，`task index`表示任务的序号。

- `run once time`表示每个客户端单次请求的推理任务的推理时长。

- 接下来显示每个客户端单次请求的推理任务的推理结果，包括输出Tensor的名称、输出Tensor的数据大小，输出Tensor的元素数量以及前5个数据。

- `total run time`表示服务器完成所有并发推理任务的总耗时。

    ```text
    parallel id:  0  | task index:  1  | run once time:  0.024082660675048828  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  2  | task index:  1  | run once time:  0.029989957809448242  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  1  | task index:  1  | run once time:  0.03409552574157715  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  3  | task index:  1  | run once time:  0.04005265235900879  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  4  | task index:  1  | run once time:  0.04981422424316406  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  0  | task index:  2  | run once time:  0.028667926788330078  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  2  | task index:  2  | run once time:  0.03010392189025879  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  3  | task index:  2  | run once time:  0.030695676803588867  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  1  | task index:  2  | run once time:  0.04117941856384277  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    parallel id:  4  | task index:  2  | run once time:  0.028671741485595703  s
    tensor name is:Softmax-65 tensor size is:4004 tensor elements num is:1001
    output data is: 1.02271215e-05 9.92699e-06 1.6968432e-05 6.8573616e-05 9.731416e-05
    total run time:  0.08787751197814941  s
    ```

## Demo内容说明

使用MindSpore Lite执行并发推理主要包括以下步骤：

1. [关键变量说明](#关键变量说明)：说明并发推理中用到的关键变量。
2. [创建并发推理配置](#创建并发推理配置)：创建并发推理配置选项[RunnerConfig](https://www.mindspore.cn/lite/api/zh-CN/r2.0.0-alpha/mindspore_lite/mindspore_lite.RunnerConfig.html)，保存需要的一些基本配置参数，用于执行并发推理的初始化。
3. [并发Runner加载与编译](#并发runner加载与编译)：执行并发推理之前，需要调用`ModelParallelRunner`的[init](https://www.mindspore.cn/lite/api/zh-CN/r2.0.0-alpha/mindspore_lite/mindspore_lite.ModelParallelRunner.html#mindspore_lite.ModelParallelRunner.init)接口进行并发Runner的初始化，主要进行模型读取，加载`RunnerConfig`配置，创建并发，以及子图切分、算子选型调度。可将`ModelParallelRunner`理解为支持并发推理的`model`。该阶段会耗费较多时间，所以建议`ModelParallelRunner`初始化一次，多次执行并发推理。
4. [设置并发推理任务](#设置并发推理任务)：创建多线程，绑定并发推理任务。推理任务包括向`输入Tensor`中填充数据、使用`ModelParallelRunner`的[predict](https://www.mindspore.cn/lite/api/zh-CN/r2.0.0-alpha/mindspore_lite/mindspore_lite.ModelParallelRunner.html#mindspore_lite.ModelParallelRunner.predict)接口进行并发推理和通过`输出Tensor`得到推理结果。
5. [执行并发推理](#执行并发推理)：启动多线程，执行配置好的并发推理任务。执行过程中，打印并发推理中的单次推理耗时和推理结果，结束线程后打印并发推理总耗时。

更多Python接口的高级用法与示例，请参考[Python API](https://www.mindspore.cn/lite/api/zh-CN/r2.0.0-alpha/mindspore_lite.html)。

![img](../images/server_inference.png)

### 关键变量说明

- `THREAD_NUM`：单个worker的线程数量。`WORKERS_NUM * THREAD_NUM`应该小于机器核心数量。

- `WORKERS_NUM`：在服务器端，指定在一个`ModelParallelRunner`中的workers的数量，即执行并发推理的单元。若想对比并发推理和非并发推理的推理时长差异，可以将`WORKERS_NUM`设置为1进行对比。

- `PARALLEL_NUM`：并发数量，即同时在发送推理任务请求的客户端数量。

- `TASK_NUM`：任务数量，即单个客户端发送的推理任务请求的数量。

    ```python
    import time
    from threading import Thread
    import numpy as np
    import mindspore_lite as mslite

    # the number of threads of one worker.
    # WORKERS_NUM * THREAD_NUM should not exceed the number of cores of the machine.
    THREAD_NUM = 1

    # In parallel inference, the number of workers in one `ModelParallelRunner` in server.
    # If you prepare to compare the time difference between parallel inference and serial inference,
    # you can set WORKERS_NUM = 1 as serial inference.
    WORKERS_NUM = 3

    # Simulate 5 clients, and each client sends 2 inference tasks to the server at the same time.
    PARALLEL_NUM = 5
    TASK_NUM = 2
    ```

### 创建并发推理配置

创建并发推理配置`RunnerConfig`。由于本教程演示的是在CPU设备上执行推理的场景，因此需要将创建的CPU设备硬件信息加入上下文`Conterxt`中，再将`Conterxt`加入`RunnerConfig`中。

```python
# Init RunnerConfig and context, and add CPU device info
cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
context = mslite.Context(thread_num=THREAD_NUM, inter_op_parallel_num=THREAD_NUM)
context.append_device_info(cpu_device_info)
parallel_runner_config = mslite.RunnerConfig(context=context, workers_num=WORKERS_NUM)
```

### 并发Runner加载与编译

调用`ModelParallelRunner`的[init](https://www.mindspore.cn/lite/api/zh-CN/r2.0.0-alpha/mindspore_lite/mindspore_lite.ModelParallelRunner.html#mindspore_lite.ModelParallelRunner.init)接口进行并发Runner的初始化，主要进行模型读取，加载`RunnerConfig`配置，创建并发，以及子图切分、算子选型调度。可将`ModelParallelRunner`理解为支持并发推理的`Model`。该阶段会耗费较多时间，所以建议`ModelParallelRunner`初始化一次，多次执行并发推理。

```python
# Build ModelParallelRunner from file
model_parallel_runner = mslite.ModelParallelRunner()
model_parallel_runner.init(model_path="./model/mobilenetv2.ms", runner_config=parallel_runner_config)
```

### 设置并发推理任务

创建多线程，绑定并发推理任务。推理任务包括向`输入Tensor`中填充数据、使用`ModelParallelRunner`的[predict](https://www.mindspore.cn/lite/api/zh-CN/r2.0.0-alpha/mindspore_lite/mindspore_lite.ModelParallelRunner.html#mindspore_lite.ModelParallelRunner.predict)接口进行并发推理和通过`输出Tensor`得到推理结果。

```python
def parallel_runner_predict(parallel_runner, parallel_id):
    """
    One Runner with 3 workers, set model input, execute inference and get output.

    Args:
        parallel_runner (mindspore_lite.ModelParallelRunner): Actuator Supporting Parallel inference.
        parallel_id (int): Simulate which client's task to process
    """

    task_index = 0
    while True:
        if task_index == TASK_NUM:
            break
        task_index += 1
        # Set model input
        inputs = parallel_runner.get_inputs()
        in_data = np.fromfile("./model/input.bin", dtype=np.float32)
        inputs[0].set_data_from_numpy(in_data)
        once_start_time = time.time()
        # Execute inference
        outputs = []
        parallel_runner.predict(inputs, outputs)
        once_end_time = time.time()
        print("parallel id: ", parallel_id, " | task index: ", task_index, " | run once time: ",
              once_end_time - once_start_time, " s")
        # Get output
        for output in outputs:
            tensor_name = output.get_tensor_name().rstrip()
            data_size = output.get_data_size()
            element_num = output.get_element_num()
            print("tensor name is:%s tensor size is:%s tensor elements num is:%s" % (tensor_name, data_size,
                                                                                     element_num))
            data = output.get_data_to_numpy()
            data = data.flatten()
            print("output data is:", end=" ")
            for j in range(5):
                print(data[j], end=" ")
            print("")


# The server creates 5 threads to store the inference tasks of 5 clients.
threads = []
total_start_time = time.time()
for i in range(PARALLEL_NUM):
    threads.append(Thread(target=parallel_runner_predict, args=(model_parallel_runner, i,)))
```

### 执行并发推理

启动多线程，执行配置好的并发推理任务。执行过程中，打印并发推理中的单次推理耗时和推理结果，结束线程后打印并发推理总耗时。

```python
# Start threads to perform parallel inference.
for th in threads:
    th.start()
for th in threads:
    th.join()
total_end_time = time.time()
print("total run time: ", total_end_time - total_start_time, " s")
```
