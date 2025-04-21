# Performing Ascend Backend Multi-card/Multi-core Inference Using Python Interfaces

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/mindir/runtime_distributed_multicard_python.md)

## Overview

In single-machine multi-card and single-card multi-core scenarios, in order to fully utilize the performance of the device, it is necessary to allow the chip or different cards to perform parallel inference directly. This scenario is more common in Ascend environments, such as the Atlas 300I Duo inference card which has a single card, dual-core specification is naturally better suited for parallel processing. This tutorial describes how to perform MindSpore Lite multi-card/multi-core inference in the Ascend backend environment using the [Python interface](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite.html). The inference core process is roughly the same as the [cloud-side single-card inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_python.html) process, and the users can refer to it.

MindSpore Lite cloud-side distributed inference is only supported to run in Linux environment deployment. The device type supported in this tutorial is Atlas training series products, and takes **Atlas 300I Duo inference card + open source Stable Diffusion (SD) ONNX model** as the case. This case calls Python multiprocess library to create sub-processes, where the user interacts with the master process that interacts with the sub-processes through a pipeline.

## Preparations

1. Download the cloud-side Ascend backend multi-card/multi-core inference Python [sample code](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_parallel_python), which will later be referred to as the sample code directory.

2. Download the MindSpore Lite cloud-side inference installer [mindspore-lite-{version}-linux-{arch}.whl](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html), store it to the sample code directory, and install it via the `pip` tool.

3. Convert the ONNX model to MindSpore MINDIR format model via [converter_lite tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html), where the batch_size is set to half the size of the original model, and is used to allocate to dual-core or dual-card parallel inference, to achieve the same output results as the original model (if more cards are used in parallel, the batch size needs to divide by the number of cards used, and is set to be the result of the division when converting). For example, for the model with batch size = 2, set it to 1 when converting, which means that each sub-process will reason about a single batch when inference in parallel on the dual-core. For the SD2.1 model, the following conversion command can be used:

    ```shell
    ./converter_lite --modelFile=model.onnx --fmk=ONNX --outputFile=SD2.1_size512_bs1 --inputShape="sample:1,4,64,64;timestep:1;encoder_hidden_states:1,77,1024" --optimize=ascend_oriented
    ```

For more usage of the converter tool and configurable optimization points for model conversion, refer to the [Model Conversion Tool](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter.html).

Subsequent chapters will describe the main steps of MindSpore Lite cloud-side distributed inference with code. Refer to [sample code](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_parallel_python).

## Inference Process

The inference process here corresponds to the class `MSLitePredict` of the sample code `mslite_engine.py`.

### Creating a Contextual Configuration

The context configuration saves the basic configuration parameters needed, mainly including setting the device type to `Ascend` and specifying a device ID to assign an executing chip or card for the model. The following sample code demonstrates how to create a context through [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context):

```python
# init and set context
context = mslite.Context()
context.target = ["ascend"]
context.ascend.device_id = device_id
```

### Model Creation, Loading and Compilation

Consistent with [MindSpore Lite cloud-side single-card inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_python.html), the main entry for Ascend backend multi-card/multi-core inference is the [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model) interface, which allows for model loading compilation and execution. Create [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model) and call [Model. build_from_file](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface to achieve model loading and model compilation. Sample code is as follows:

```python
# create Model and build Model
model = mslite.Model()
model.build_from_file(mindir_file, mslite.ModelType.MINDIR, context)
```

### Model Input Data Padding

First, use the [Model.get_inputs](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.get_inputs) method to get all inputs [Tensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Tensor.html#mindspore_lite.Tensor), and fill in the Host data using the relevant interface. Sample code is as follows:

```python
# set model input
inputs = model.get_inputs()
for i, _input in enumerate(inputs):
    _input.set_data_from_numpy(input_data[i])
```

### Inference Execution

Call [Model.predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface to perform inference, and sample code is shown below:

```python
# execute inference
outputs = model.predict(inputs)
```

## Multi-Card/Multi-Core Parallel

The sample code `parallel_predict_utils.py` uses multiprocessing Process and Pipe methods to implement parallel startup of sub-processes as well as synchronized waiting operations. The function `predict_process_func` defines the actions to be performed in the child process, which initializes the `MSLitePredict` class introduced in the previous section and calls its inference methods to complete the model inference. The class in the main process is `ParallelPredictUtils`, and the specific parallel process is as follows.

1. Call multiprocessing.Process in the main process to generate two child processes, the main process passes the model path and `device_id` to the child process, the child process initializes and compiles the model, and sends a compilation success (`BUILD_FINISH`)/failure (`ERROR`) message to the main process through the pipeline when it is done;

2. Upon receiving the build success message from both child processes, the master process pipes in the message to start inference (`PREDICT`) to the child process with the inference input data;

3. The child process performs parallel inference and puts the inference completion message (`REPLY`) and the inference result into the pipeline;

4. The master process receives the inference results from the two child processes in the pipeline;

5. Repeat steps 2-4 until the master process sends an exit message (`EXIT`), which the child process receives and interrupts pipeline listening to exit normally.

We can define the above messages used for pipeline communication as global variables:

```python
# define message id
ERROR = 0
PREDICT = 1
REPLY = 2
BUILD_FINISH = 3
EXIT = 4
```

The main process initialization code is as follows:

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

The sample code for creating and scheduling child processes by the master process is as follows:

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

The master process ends the child process and waits for synchronization, and the sample code is as follows:

```python
# finalize subprocess
for i in range(self.device_num):
    self.pipe_parent_end[i].send((EXIT, ""))
for i in range(self.device_num):
    self.process[i].join()
```

The subprocess initialization loading model sample code is as follows:

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

The subprocess inference example code is as follows:

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

## Performing Dual-card/Dual-core Parallel Samples

Configure the model path `mindir_path` in `main.py` in the sample code directory as the mindir path obtained from the model conversion (refer to the section on Preparation), and execute `main.py` to realize the single-card dual-core/dual-card parallel inference. The sample code slices the input data according to the batch dimension, i.e., the data with batch_size=2 in the input is sliced into two single-batch data as shown below:

```python
# data slice
input_data_0 = [sample[0], timestep, encoder_hidden_states[0]]
input_data_1 = [sample[1], timestep, encoder_hidden_states[1]]
input_data_all = [input_data_0, input_data_1]
```

When inference is complete, the total inference time `Total predict time = XXX` in seconds is printed. If the inference is successful, the entire execution prints the following message. The inference result is saved in the variable `predict_result`, which contains the inference results of two single-batch models, spliced together to match the results of a model with batch_size = 2.

```shell
Success to build model 0
Success to build model 1
Total predict time =  XXX
Receive exit message  and start to exit
Receive exit message  and start to exit
=========== success ===========
```

