# Performance Profiling (Ascend)

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindinsight/docs/source_en/performance_profiling_ascend.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Overview

This article describes how to use MindSpore Profiler for performance debugging on Ascend AI processors.

## Operation Process

- Prepare a training script, add profiler APIs in the training script and run the training script.
- Start MindInsight and specify the summary-base-dir using startup parameters, note that summary-base-dir is the parent directory of the directory created by Profiler. For example, the directory created by Profiler is `/home/user/code/data/`, the summary-base-dir should be `/home/user/code`. After MindInsight is started, access the visualization page based on the IP address and port number. The default access IP address is `http://127.0.0.1:8080`.
- Find the training in the list, click the performance profiling link and view the data on the web page.

## Preparing the Training Script

To enable the performance profiling of neural networks, MindSpore Profiler APIs should be added into the script. At first, the MindSpore `Profiler` object need to be set after `set_context` is set and before the network and HCCL initialization. Then, at the end of the training, `Profiler.analyse()` should be called to finish profiling and generate the perforamnce analyse results.

> The parameters of Profiler are as follows:
>
> <https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore.profiler.html>

Profiler can control whether performance data collection is turned on or off based on step (epoch) with the start_profile parameter. For the data sinking mode of graph mode, CANN can only be told to turn on and off after each epoch, so for the data sinking mode, it needs to turn on and off based on the epoch.

The code for a normal scenario is as follows:

```python
import numpy as np
from mindspore import nn, context
from mindspore import Model
import mindspore.dataset as ds
from mindspore.profiler import Profiler


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        return self.fc(x)


def generator():
    for i in range(2):
        yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))


def train(net):
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator, ["data", "label"])
    model = Model(net, loss, optimizer)
    model.train(1, data)


if __name__ == '__main__':

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    # Init Profiler
    # Note that the Profiler should be initialized after context.set_context and before model.train
    # If you are running in parallel mode on Ascend, the Profiler should be initialized before HCCL
    # initialized.

    profiler = Profiler(output_path = './profiler_data')
    # Train Model
    net = Net()
    train(net)
    # Profiler end
    profiler.analyse()
```

Graph mode:

- When dataset_sink_mode is set to False, it needs to be enabled based on step.

    ```python
    from mindspore.train.callback import Callback
    class StopAtStep(Callback):
        def __init__(self, start_step, stop_step):
            super(StopAtStep, self).__init__()
            self.start_step = start_step
            self.stop_step = stop_step
            self.profiler = Profiler(start_profile=False)
        def step_begin(self, run_context):
            cb_params = run_context.original_args()
            step_num = cb_params.cur_step_num
            if step_num == self.start_step:
                self.profiler.start()
        def step_end(self, run_context):
            cb_params = run_context.original_args()
            step_num = cb_params.cur_step_num
            if step_num == self.stop_step:
                self.profiler.stop()
        def end(self, run_context):
            self.profiler.analyse()
    ```

- When dataset_sink_mode is set to True, It needs to be enabled based on epoch.

    ```python
    class StopAtEpoch(Callback):
        def __init__(self, start_epoch, stop_epoch):
            super(StopAtEpoch, self).__init__()
            self.start_epoch = start_epoch
            self.stop_epoch = stop_epoch
            self.profiler = Profiler(start_profile=False)
        def epoch_begin(self, run_context):
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num
            if epoch_num == self.start_epoch:
                self.profiler.start()
        def epoch_end(self, run_context):
            cb_params = run_context.original_args()
            epoch_num = cb_params.cur_epoch_num
            if epoch_num == self.stop_epoch:
                self.profiler.stop()
        def end(self, run_context):
            self.profiler.analyse()
    ```

Custom training：

```python
profiler = Profiler(start_profile=False)
data_loader = ds.create_dict_iterator()

for i, data in enumerate(data_loader):
    train()
    if i==100:
        profiler.start()
    if i==200:
        profiler.stop()

profiler.analyse()
```

## Launch MindInsight

The MindInsight launch command can refer to [MindInsight Commands](https://www.mindspore.cn/mindinsight/docs/en/r1.6/mindinsight_commands.html).

## Training Performance

Users can access the Training Performance by selecting a specific training from the training list, and click the performance profiling link.

![performance_overall.png](./images/performance_overall.png)

*Figure 1: Overall Performance*

Figure 1 displays the overall performance of the training, including the overall data of Step Trace, Operator Performance, Data Preparation Performance and Timeline. The data shown in these components include:  

- Step Trace: It will divide the training steps into several stages and collect execution time for each stage. The overall performance page will show the step trace graph.
- Operator Performance: It will collect the execution time of operators and operator types. The overall performance page will show the pie graph for different operator types.
- Data Preparation Performance: It will analyse the performance of the data input stages. The overall performance page will show the number of steps that may be the bottleneck for these stages.
- Timeline: It will collect execution time for stream tasks on the devices. The tasks will be shown on the time axis. The overall performance page will show the statistics for streams and tasks.  

Users can click the detail link to see the details of each components. Besides, MindInsight Profiler will try to analyse the performance data, the assistant on the left will show performance tuning suggestions for this training.

### Step Trace Analysis

The Step Trace Component is used to show the general performance of the stages in the training. Step Trace will divide the training into several stages:  
Step Gap (The time between the end of one step and the computation of next step), Forward/Backward Propagation, All Reduce and Parameter Update. It will show the execution time for each stage, and help to find the bottleneck stage quickly.

> Step Trace does not support heterogeneous training currently.

![step_trace.png](./images/step_trace.png)

*Figure 2: Step Trace Analysis*

Figure 2 displays the Step Trace page. The Step Trace detail will show the start/finish time for each stage. By default, it shows the average time for all the steps. Users can also choose a specific step to see its step trace statistics.

The graphs at the bottom of the page show the execution time of Step Interval, Forward/Backward Propagation and Step Tail (The time between the end of Backward Propagation and the end of Parameter Update) changes according to different steps, it will help to decide whether we can optimize the performance of some stages. Here are more details:

- **Step Interval** is the duration for reading data from data queues. If this part takes long time, it is advised to check the data preparation for further analysis.
- **Forward and Backward Propagation** is the duration for executing the forward and backward operations on the network, which handle the main calculation work of a step. If this part takes long time, it is advised to check the statistics of operators or timeline for further analysis.
- **Step Tail** is the duration for performing parameter aggregation and update operations in parallel training. If the operation takes long time, it is advised to check the statistics of communication operators and the status of parallelism.

In order to divide the stages, the Step Trace Component need to figure out the forward propagation start operator and the backward propagation end operator. MindSpore will automatically figure out the two operators to reduce the profiler configuration work. The first operator after `get_next` will be selected as the forward start operator and the operator before the last all reduce will be selected as the backward end operator.
**However, Profiler do not guarantee that the automatically selected operators will meet the user's expectation in all cases.** Users can set the two operators manually as follows:  

- Set environment variable `PROFILING_FP_START` to configure the forward start operator, for example, `export PROFILING_FP_START=fp32_vars/conv2d/BatchNorm`.
- Set environment variable `PROFILING_BP_END` to configure the backward end operator, for example, `export PROFILING_BP_END=loss_scale/gradients/AddN_70`.

### Operator Performance Analysis

The operator performance analysis component is used to display the execution time of the operators(AICORE/AICPU/HOSTCPU) during MindSpore run.

- AICORE：AI Core operator is the main component of the computing core of Ascend AI processor, which is responsible for executing vector and tensor related computation intensive operators. TBE (Tensor Boost Engine) is an extended operator development tool based on TVM (Tensor Virtual Machine) framework. Users can use TBE to register AI Core operator information.
- AICPU：AI CPU operator is a kind of CPU operator (including control operator, scalar, vector and other general-purpose calculations) that AI CPU is responsible for executing Hisilicon SOC in Ascend processor. The same operator in MindSpore may have AI Core operator and AI CPU operator at the same time. The framework will give priority to AI Core operator. If there is no AI Core operator or the selection is not satisfied, AI CPU operator will be called.
- HOSTCPU：The host side CPU is mainly responsible for distributing the graph or operator to Ascend chip, and the operator can also be developed on the host side CPU according to the actual needs. The host CPU operator refers to the operator running on the host side CPU.

![op_type_statistics.png](./images/op_type_statistics.PNG)

*Figure 3: Statistics for Operator Types*

Figure 3 displays the statistics for the operator types, including:  

- Choose pie or bar graph to show the proportion time occupied by each operator type. The time of one operator type is calculated by accumulating the execution time of operators belonging to this type.
- Display top 20 operator types with the longest execution time, show the proportion and execution time (ms) of each operator type.

![op_statistics.png](./images/op_statistics.PNG)

*Figure 4: Statistics for Operators*

Figure 4 displays the statistics table for the operators, including:  

- Choose All: Display statistics for the operators, including operator name, type, execution time, full scope time, information, etc. The table will be sorted by execution time by default.
- Choose Type: Display statistics for the operator types, including operator type name, execution time, execution frequency and proportion of total time. Users can click on each line, querying for all the operators belonging to this type.
- Search: There is a search box on the right, which can support fuzzy search for operators/operator types.

Statistics for the information related to calculation quantity of AICORE operator, including operator level and model level information.

### Calculation quantity analysis

The Calculation Quantity Analysis module shows the actual calculation quantity data, including calculation quantity data for operator granularity, scope level granularity, and model granularity. The actual calculation quantity refers to the amount of calculation that is running on the device, which is different from the theoretical calculation quantity. For example, the matrix computing unit on the Ascend910 device is dealing with a matrix of 16x16 size, so in the runtime, the original matrix will be padded to 16x16.
Only calculation quantity on AICORE devices is supported currently. The information about calculation quantity has three indicators:

- FLOPs: the number of floating point operations（the unit is million）.
- FLOPS: the number of floating point operations per second (the unit is billion).
- FLOPS utilization: obtained by dividing the FLOPS by the peak FLOPS of the AICORE device.

![flops_statistics.png](./images/flops-single-card.png)

*Figure 5: Calculation Quantity Analysis*

The red box in Figure 5 includes calculation quantity data on operator granularity, scope level granularity, and model granularity. Click the "details" to see the scope level calculation quantity data.

![flops_scope_statistics.png](./images/flops-scope.png)

*Figure 6: Scope Level FLOPs*

Figure 6 is a sankey diagram that presents data in the structure of a tree where the cursor selects a scope to see the specific FLOPs value.

### Data Preparation Performance Analysis

The Data preparation performance analysis component is used to analyse the execution of data input pipeline for the training. The data input pipeline can be divided into three stages:  
the data process pipeline, data transfer from host to device and data fetch on device. The component will analyse the performance of each stage in detail and display the results.

![minddata_profile.png](./images/minddata_profile.png)

*Figure 7: Data Preparation Performance Analysis*

Figure 7 displays the page of data preparation performance analysis component. It consists of two tabs: the step gap and the data process.

The step gap page is used to analyse whether there is performance bottleneck in the three stages. We can get our conclusion from the data queue graphs:  

- The data queue size stands for the queue length when the training fetches data from the queue on the device. If the data queue size is 0, the training will wait until there is data in the queue; If the data queue size is greater than 0, the training can get data very quickly, and it means data preparation stage is not the bottleneck for this training step.
- The host queue size can be used to infer the speed of data process and data transfer. If the host queue size is 0, it means we need to speed up the data process stage.
- If the size of the host queue is always large and the size of the data queue is continuously small, there may be a performance bottleneck in data transfer.

![data_op_profile.png](./images/data_op_profile.png)

*Figure 8: Data Process Pipeline Analysis*

Figure 8 displays the page of data process pipeline analysis. The data queues are used to exchange data between the data processing operators. The data size of the queues reflect the data consume speed of the operators, and can be used to infer the bottleneck operator. The queue usage percentage stands for the average value of data size in queue divide data queue maximum size, the higher the usage percentage, the more data that is accumulated in the queue. The graph at the bottom of the page shows the data processing pipeline operators with the data queues, the user can click one queue to see how the data size changes according to the time, and the operators connected to the queue. The data process pipeline can be analysed as follows:  

- When the input queue usage percentage of one operator is high, and the output queue usage percentage is low, the operator may be the bottleneck.
- For the leftmost operator, if the usage percentage of all the queues on the right are low, the operator may be the bottleneck.
- For the rightmost operator, if the usage percentage of all the queues on the left are high, the operator may be the bottleneck.

To optimize the performance of data processing operators, there are some suggestions:  

- If the Dataset Operator is the bottleneck, try to increase the `num_parallel_workers`.
- If a GeneratorOp type operator is the bottleneck, try to increase the `num_parallel_workers` and replace the operator to `MindRecordDataset`.
- If a MapOp type operator is the bottleneck, try to increase the `num_parallel_workers`. If it is a python operator, try to optimize the training script.
- If a BatchOp type operator is the bottleneck, try to adjust the size of `prefetch_size`.

### Timeline Analysis

The Timeline component can display:

- The operators (AICORE/AICPU/HOSTCPU operators) are executed on which device.
- The MindSpore stream split strategy for this neural network.
- The execution sequence and execution time of the operator on the device.
- The step number of training (Currently dynamic shape scene, multi-graph scene and heterogeneous training scene are not supported, steps data may be inaccurate in these scene.).
- `Scope Name` of the operator, the number of each operator's `Scope Name` could be selected and download corresponding timeline file. For example, the full name of one operator is `Default/network/lenet5/Conv2D-op11`, thus the first `Scope Name` of this operator is `Default`, the second `Scope Name` is `network`. If two `Scope Name` for each operator is selected, then the `Default` and `network` will be displayed.

Users can get the most detailed information from the Timeline:  

- From the High level, users can analyse whether the stream split strategy can be optimized and whether the step tail is too long.
- From the Low level, users can analyse the execution time for all the operators, etc.

Users can click the download button on the overall performance page to view Timeline details. The Timeline data file (json format) will be stored on local machine, and can be displayed by tools. We suggest to use `chrome://tracing` or [Perfetto](https://ui.perfetto.dev/#!viewer) to visualize the Timeline.  

- Chrome tracing: Click "load" on the upper left to load the file.
- Perfetto: Click "Open trace file" on the left to load the file.

![timeline.png](./images/timeline.png)

*Figure 9: Timeline Analysis*

The Timeline consists of the following parts:  

- Device and Stream List: It will show the stream list on each device. Each stream consists of a series of tasks. One rectangle stands for one task, and the area stands for the execution time of the task.

  Each color block represents the starting time and length of operator execution. The detailed explanation of timeline is as follows:
    - Process Device ID: contains the timeline of operators executed on AI Core.
        - Step: the number of training steps.
        - Scope Name: the Scope Name of operators.
        - Stream #ID: operators executed on the stream.
    - Process AI CPU Op: the timeline of operators executed on the AI CPU.
    - Process Communication Op: the timeline for the execution of communication operators.
    - Process Host CPU Op: contains the timeline of operators executed on the Host CPU.
    - Process Op Overlap Analyse: the timeline of all computation operators and communication operators merged, it can be used to analyse the proportion of communication time.
        - Merged Computation Op: it is the timeline after all computation operators are merged.
        - Merged Communication Op: it is the timeline after all communication operators are merged.
        - Pure Communication Op: pure communication time (the timeline of the communication operator after removing the overlap with the computation operator time).
        - Free Time: there is no communication operator and calculation operator in the execution timeline.

- The Operator Information: When we click one task, the corresponding operator of this task will be shown at the bottom.

W/A/S/D can be applied to zoom in and out of the Timeline graph.

## Resource Utilization

Resource utilization includes cpu usage analysis and memory usage analysis.

![resource_visibility.png](./images/resource_visibility.png)

*Figure 10：Overview of resource utilization*

Overview of resource utilization：Including CPU utilization analysis and memory usage analysis. You can view the details by clicking the View Details button in the upper right corner.

### CPU Utilization Analysis

CPU utilization, which is mainly used to assist performance debugging. After the performance bottleneck is determined according to the queue size, the performance can be debugged according to the CPU utilization (if the user utilization is too low, increase the number of threads; if the system utilization is too high, decrease the number of threads).
CPU utilization includes CPU utilization of the whole machine, process and Data pipeline operator.

![device_utilization.png](./images/device_cpu_utilization.png)

*Figure 11: CPU utilization of the whole machine*

CPU utilization of the whole machine: Show the overall CPU usage of the device in the training process, including user utilization, system utilization, idle utilization, IO utilization, current number of active processes, and context switching times. If the user utilization is low, you can try to increase the number of operator threads to increase the CPU utilization; if the system utilization is high, and the number of context switching and CPU waiting for processing is large, it indicates that the number of threads needs to be reduced accordingly.

![process_cpu_utilization.png](./images/process_cpu_utilizaton.png)

*Figure 12: Process utilization*

Process utilization: Show the CPU usage of a single process. The combination of whole machine utilization and process utilization can determine whether other processes affect the training process.

![data_op_utilization.png](./images/data_op_utilization.png)

*Figure 13: Operator utilization*

Operator utilization: Show the CPU utilization of Data pipeline single operator. We can adjust the number of threads of the corresponding operator according to the actual situation. If the number of threads is small and takes up a lot of CPU, you can consider whether you need to optimize the code.

Common scenarios of CPU utilization:

- According to the queue size, the network debugging personnel can judge that the performance of MindData has a bottleneck. They can adjust the number of threads by combining the utilization rate of the whole machine and the utilization rate of the operator.
- Developers can check the utilization of operators. If an operator consumes CPU utilization, they can confirm whether the code needs to be optimized.

> The default sampling interval is 1000ms. You can change the sampling interval through `mindspore.dataset.config.get_monitor_sampling_interval()`. For details：
>
> <https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore.dataset.config.html#mindspore.dataset.config.set_monitor_sampling_interval>

### Memory Analysis

This page is used to show the memory usage of the neural network model on the **device**, which is an **ideal prediction** based on the theoretical calculation results. The content of the page includes:

- An overview of the memory usage of the model, including the total available memory, peak memory and other information.
- The memory occupied varies in the execution order while the model is running.
- The memory usage of each operator is decomposed and displayed in the table of ```Operator Memory Allocation```.

> Memory Analysis does not support heterogeneous training currently.

![memory.png](./images/memory.png)

*Figure 14：Memory Analysis*

Users can obtain the summary of memory usage via the ```Memory Allocation Overview```. In addition, they can obtain more detailed information from ```Memory Usage```, including:

- **Line Chart**: Changes in model memory usage, including static memory, total occupied memory and total available memory.
- **Zooming**: There is a zoom scroll bar under the line chart. Users can zoom in or out the line chart by adjusting its size to observe more details.
- **FP/BP**: The execution positions of the start of ```Forward Propagation``` and the end of ```Backward Propagation``` of the model on the line chart.
- **Details of Nodes**: Hovering over the line chart, the information of the corresponding execution operator is shown, including the execution order of the operator, the name of the operator, the memory occupied by the operator, the total memory occupied by the model in the current position, and the relative memory change compared with the previous execution position.
- **Memory Decomposition**: Left clicking a position on the line chart, the memory breakdowns of the execution position is shown in the table below the line chart, called ```Operator Memory Allocation```. The table shows the memory decomposition of the corresponding execution position, i.e., the output tensor of which operators are allocated the occupied memory of the current execution position. The module provides users with abundant information, including tensor name, tensor size, tensor type, data type, shape, format, and the active lifetime of tensor memory.

![memory_graphics.png](./images/memory_graphics.png)

*Figure 15：Memory Statistics*

## Specifications

- To limit the data size generated by the Profiler, MindInsight suggests that for large neural network, the profiled steps should be less than 10.

  > The number of steps can be controlled by controlling the size of training data set. For example, the `num_samples` parameter in `mindspore.dataset.MindDataset` can control the size of the data set. For details, please refer to:
  > <https://www.mindspore.cn/docs/api/en/r1.6/api_python/dataset/mindspore.dataset.MindDataset.html>

- The parse of Timeline data is time consuming, and usually the data of a few steps is enough to analyze the results. In order to speed up the data parse and UI display, Profiler will show at most 20M data (Contain 10+ step information for large networks).

## Notices

- Currently running in PyNative mode is not supported.
- Currently the training and inference process does not support performance debugging, only individual training or inference is supported.
