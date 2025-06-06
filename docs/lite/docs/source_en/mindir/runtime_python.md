# Using Python Interface to Perform Cloud-side Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/mindir/runtime_python.md)

## Overview

This tutorial provides a sample program for MindSpore Lite to perform cloud-side inference, demonstrating the [Python interface](https://mindspore.cn/lite/api/en/master/mindspore_lite.html) to perform the basic process of cloud-side inference through file input, inference execution, and inference result printing, and enables users to quickly understand the use of MindSpore Lite APIs related to cloud-side inference execution. The related files are put in the directory [mindspore/lite/examples/cloud_infer/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_python).

MindSpore Lite cloud-side inference is supported to run in Linux environment deployment only. Atlas 200/300/500 inference product, Atlas inference series, Atlas training series, Nvidia GPU and CPU hardware backends are supported.

The following is an example of how to use the Python Cloud-side Inference Demo on a Linux X86 operating system and a CPU hardware platform, using Ubuntu 18.04 as an example:

- One-click installation of inference-related model files, MindSpore Lite and its required dependencies. See the [One-click installation](#one-click-installation) section for details.

- Execute the Python Cloud-side Inference Demo. See the [Execute Demo](#executing-demo) section for details.

- For a description of the Python Cloud-side Inference Demo content, see the [Demo Content Description](#demo-content-description) section for details.

## One-click Installation

This session introduces the installation of MindSpore Lite for Python version 3.7 via pip on a Linux-x86_64 system with a CPU environment, taking the new Ubuntu 18.04 as an example.

Go to the [mindspore/lite/examples/cloud_infer/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_python) directory, and execute the `lite-cpu-pip.sh` script for a one-click installation, taking installation of MindSpore Lite version 2.0.0 as an example. Script installation needs to download the model required for inference and input data files, the dependencies required for MindSpore_Lite installation, and download and install MindSpore Lite.

Note: This command sets the installed version of MindSpore Lite. Since the cloud-side inference Python interface is supported from MindSpore Lite version 2.0.0, the version cannot be set lower than 2.0.0. See the version provided in [Download MindSpore Lite](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) for details on the versions that can be set.

```bash
MINDSPORE_LITE_VERSION=2.0.0 bash ./lite-cpu-pip.sh
```

> If the MobileNetV2 model download fails, please manually download the relevant model file [mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/mobilenetv2.mindir) and copy it to the `mindspore/lite/examples/cloud_infer/quick_start_python/model` directory.
>
> If the input.bin input data file download fails, please manually download the relevant input data file [input.bin](https://download.mindspore.cn/model_zoo/official/lite/quick_start/input.bin) and copy it to the ` mindspore/lite/examples/cloud_infer/quick_start_python/model` directory.
>
> If MindSpore Lite inference framework by using the script download fails, please manually download [MindSpore Lite model cloud-side inference framework](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) corresponding to the hardware platform of CPU and operating system of Linux-x86_64 or Linux-aarch64. Users can use the `uname -m` command to query the operating system in the terminal, and copy it to the `mindspore/lite/examples/cloud_infer/quick_start_python` directory.
>
> If you need to use MindSpore Lite corresponding to Python 3.7 or above, please [compile](https://mindspore.cn/lite/docs/en/master/mindir/build.html) locally. Note that the Python API module compilation depends on Python >= 3.7.0, NumPy >= 1.17.0, wheel >= 0.32.0. After successful compilation, copy the Whl installation package generated in the `output/` directory to the `mindspore/lite/examples/cloud_infer/quick_start_python` directory.
>
> If the MindSpore Lite installation package does not exist in the `mindspore/lite/examples/cloud_infer/quick_start_python` directory, the one-click installation script will uninstall the currently installed MindSpore Lite and then download and install MindSpore Lite from the Huawei image. Otherwise, if the MindSpore Lite installation package exists in the directory, it will be installed first.
>
> After manually downloading and placing the files in the specified location, you need to execute the lite-cpu-pip.sh script again to complete the one-click installation.

A successful execution will show the following results. The model files and input data files can be found in the `mindspore/lite/examples/cloud_infer/quick_start_python/model` directory.

```text
Successfully installed mindspore-lite-2.0.0
```

## Executing Demo

After one-click installation, go to the [mindspore/lite/examples/cloud_infer/quick_start_python](https://gitee.com/mindspore/mindspore/tree/master/mindspore/lite/examples/cloud_infer/quick_start_python) directory and execute the following command to experience MindSpore Lite inference MobileNetV2 models.

```bash
python quick_start_cloud_infer_python.py
```

When the execution is completed, the following results will be obtained, printing the name of the output Tensor, the data size of the output Tensor, the number of elements of the output Tensor and the first 50 pieces of data.

```text
tensor's name is:shape1 data size is:4000 tensor elements num is:1000
output data is: 5.3937547e-05 0.00037763786 0.00034193686 0.00037316754 0.00022436169 9.953917e-05 0.00025308868 0.00032044895 0.00025788433 0.00018915901 0.00079509866 0.003382262 0.0016214572 0.0010760546 0.0023826156 0.0011769629 0.00088481285 0.000534926 0.0006929171 0.0010826243 0.0005747609 0.0014443205 0.0010454883 0.0016276307 0.00034437355 0.0001039985 0.00022641376 0.00035307938 0.00014567627 0.00051178376 0.00016933997 0.00075814105 9.704676e-05 0.00066705025 0.00087511574 0.00034623547 0.00026317223 0.000319407 0.0015627446 0.0004044049 0.0008798965 0.0005202293 0.00044808138 0.0006453716 0.00044969268 0.0003431648 0.0009871059 0.00020436312 7.405098e-05 8.805057e-05
```

## Demo Content Description

Running MindSpore Lite inference framework mainly consists of the following steps:

1. Model reading: Export MindIR model via MindSpore or get MindIR model by [model conversion tool](https://www.mindspore.cn/lite/docs/en/master/mindir/converter_tool.html).
2. Create configuration context: Create a configuration context [Context](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context) and save some basic configuration parameters used to guide model compilation and model execution.
3. Model creation and compilation: Before executing inference, you need to call [build_from_file](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface of [Model](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model) for model loading and model compilation. The model loading phase parses the file cache into a runtime model. The model compilation phase can take more time, so it is recommended that the model be created once, compiled once and performed inference about multiple times.
4. Input data: The input data needs to be padded before the model execution.
5. Execute inference: Use [Predict](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) of [Model](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model) method for model inference.

For more advanced usage and examples of Python interfaces, please refer to the [Python API](https://www.mindspore.cn/lite/api/en/master/mindspore_lite.html).

![img](../images/lite_runtime.png)

### Creating Configuration Context

Create the configuration context `Context`. Since this tutorial demonstrates a scenario where inference is performed on a CPU device, they need to set Context's target to cpu.

```python
import numpy as np
import mindspore_lite as mslite

# init context, and set target is cpu
context = mslite.Context()
context.target = ["cpu"]
context.cpu.thread_num = 1
context.cpu.thread_affinity_mode=2
```

If the user needs to run inference on Ascend device, they need to set Context's target to ascend.

```python
import numpy as np
import mindspore_lite as mslite

# init context, and set target is ascend.
context = mslite.Context()
context.target = ["ascend"]
context.ascend.device_id = 0
```

If the backend is Ascend deployed on the Elastic Cloud Server, set the `provider` to `ge`.

```python
context.ascend.provider = "ge"
```

If the user needs to run inference on a GPU device, they need to set Context's target to gpu.

```python
import numpy as np
import mindspore_lite as mslite

# init context, and set target is gpu.
context = mslite.Context()
context.target = ["gpu"]
context.gpu.device_id = 0
```

### Model Loading and Compilation

Model loading and compilation can be done by calling [build_from_file](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface of `Model` to load and compile the runtime model directly from the file cache.

```python
# build model from file
MODEL_PATH = "./model/mobilenetv2.mindir"
IN_DATA_PATH = "./model/input.bin"
model = mslite.Model()
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context)
```

### Inputting the Data

The way that this tutorial sets the input data is importing from a file. For other ways to set the input data, please refer to [predict](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface of `Model`.

```python
# set model input
inputs = model.get_inputs()
in_data = np.fromfile(IN_DATA_PATH, dtype=np.float32)
inputs[0].set_data_from_numpy(in_data)
```

### Executing Inference

Call [predict](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface of `Model` to perform inference, and the inference result is output to `output`.

```python
# execute inference
outputs = model.predict(inputs)
```

### Obtaining the Output

Print the output results after performing inference. Iterate through the `outputs` list and print the name, data size, number of elements, and the first 50 data for each output Tensor.

```python
# get output
for output in outputs:
  name = output.name.rstrip()
  data_size = output.data_size
  element_num = output.element_num
  print("tensor's name is:%s data size is:%s tensor elements num is:%s" % (name, data_size, element_num))
  data = output.get_data_to_numpy()
  data = data.flatten()
  print("output data is:", end=" ")
  for i in range(50):
    print(data[i], end=" ")
  print("")
```

## Dynamic Weight Update

MindSpore Lite inference supports dynamic weight updates on the Ascend backend. The usage steps are as follows:

### Creating Config File

Write all tensor names corresponding to the Matmul operators that need to be updated into a text file, with each tensor name occupying one line. Build a model to load the configuration file, set the configuration file, and the content of the configuration file `config.ini` is as follows:

```text
[ascend_context]
variable_weights_file="update_weight_name_list.txt"
```

### Model Loading and Compilation

```python
import numpy as np
import mindspore_lite as mslite

# init context, and set target is gpu.
context = mslite.Context()
context.target = ["ascend"]
context.gpu.device_id = 0

# build model from file
MODEL_PATH = "./SD1.5/unet.mindir"
model = mslite.Model()
model.build_from_file(MODEL_PATH, mslite.ModelType.MINDIR, context,  "config.ini")
```

### Building A New Weight Tensor

Convert the SaveTensor data structure exported from third-party framework training to Tensor format that MindSpore Lite can support.

### Update Weights

Call the `update_weights` interface provided by MindSpore Lite to update weights, as shown below:

```python
new_weight = mslite.Tensor(data)
new_weights = [new_weight]
model.update_weights([new_weights])
```

## Weight Sharing between Processes

In scenarios where different processes load the same model file, the use of cross-process sharing of model weight parameters avoids the need for each process to store redundant copies independently, thus significantly reducing the memory usage in multitasking scenarios. MindSpore Lite provides the ability to obtain the PID of the current process, obtain the shared weight memory and set the weight memory, and the transfer of PID and shared memory between multiple processes needs to be implemented by the user, and the following steps should be performed when using this feature:

### Step 1: Get the Process ID

Use the following method to obtain the PID of the slave process:

```python
current_pid = mindspore_lite. Model().get_model_info("current_pid") # Do not use system interfaces such as os.getpid().
```

Note: This method must be used to obtain the PID, otherwise there may be unforeseen problems!

### Step 2: The Main Process is Collected and the Model is Initialized

The user needs to pass the PID from the process to the master process through inter-process communication, and the master process will create the model as shown in the following code after receiving the PID:

```python
master_model = mindspore_lite. Model()
master_model.build_from_file(MODEL_PATH, mslite. ModelType.MINDIR, context, config_dict={"pids": collected_pids})
```

Note: The collected_pids is a string and its format is 'pid1, pid2, pid3'. collected_pids is a process whitelist, and only the PIDs declared in the list can use the shared memory.

### Step 3: Get and Pass the Memory Handle

Get the shared memory of the current model as follows:

```python
shared_mem_handle = master_model.get_model_info("shareable_weight_mem_handle") # uint64 type
```

Note: Shared video memory is a uint64 value, which is passed to the slave process by means of inter-process communication.

### Step 4: Bound Shared Video Memory from the Process

After the slave process obtains the shared memory delivered to the main process, it needs to use the shared memory in the following ways:

```python
slave_model = mindspore_lite. Model()
slave_model.build_from_file(model_path, mslite. ModelType.MINDIR, context, config_dict={"shared_mem_handle": shared_mem_handle})
```

Note: The model that uses the shared memory in the slave process should be consistent with the main process, and the device in it also needs to be consistent, that is, the model is initialized by the same model file.

Here's a Python example of sharing weights between two processes, including simple multi-process communication and how to use mindspore_lite interfaces:

```python
import mindspore_lite as mslite
import numpy as np
import time
from multiprocessing import Process,Value,Array
import ctypes
PROCESS_NUM = 3
shared_arr = Array('i', [0]*PROCESS_NUM)
share_pid = Array('i', [0]*PROCESS_NUM)
share_handle = Value(ctypes.c_int64,0)

class MsliteModel:
    def __init__(self,model_name='unet'):
        self.model_name = model_name
        self.model = mslite.Model()
        self.pid = self.model.get_model_info("current_pid")

    def __call__(self, inputs):
        self.ms_inputs = self.model.get_inputs()
        shapes = [list(input.shape) for input in inputs]
        self.model.resize(self.ms_inputs, shapes)
        for i in range(len(inputs)):
            self.ms_inputs[i].set_data_from_numpy(inputs[i])
        outputs = self.model.predict(self.ms_inputs)
        outputs = [output.get_data_to_numpy() for output in outputs]
        return outputs

def init_context(device_id, device_type='ascend'):
        context = mslite.Context()
        context.target = [device_type]
        context.ascend.device_id = device_id
        return context

def ChaekPidsComplete():
    filled_count = 0
    for pid in share_pid:
        if pid != 0:
            filled_count += 1
    return filled_count==PROCESS_NUM-1

def thread_infer(index):
    print("index:", index)
    model_path1 = "path_to_your_mindir"
    inputs1 = [np.random.randn(1,8,168,128).astype(np.float32), np.random.randn(1).astype(np.int32)]

    context_1 = init_context(2)
    print("begin model group")
    print("*" * 10, "begin build model", "*" * 10)
    m1 = MsliteModel()
    if index == 0:
        while True:
            if ChaekPidsComplete():
                break
        pids = []
        for pid in share_pid:
            pids.append(int(pid))
        print("str of pids:", str(pids)[1:-1])
        config_info = {"ascend_context":{"shareable_weight_pid_list":str(pids)[1:-1]}}
        m1.model.build_from_file(model_path1,mslite.ModelType.MINDIR, context_1,config_dict=config_info)
        shareable_handle = m1.model.get_model_info("shareable_weight_mem_handle")
        print("shareable handle:", shareable_handle)
        share_handle.value = int(shareable_handle)
        print("int shareable_handle:", int(shareable_handle))
        print("share handle value:", share_handle.value)
    else:
        pid = m1.model.get_model_info("current_pid")
        pid = int(pid)
        print("pid:", pid)
        share_pid[index] = pid
        while True:
            if share_handle.value != 0:
                break
        print("sub process sharehandle:", share_handle.value)
        config_info = {"ascend_context":{"shareable_weight_mem_handle":str(share_handle.value)}}
        m1.model.build_from_file(model_path1, mslite.ModelType.MINDIR, context_1, config_dict=config_info)
    shared_arr[index] = 1
    while True:
        sum = 0
        for num in shared_arr:
            sum += num
        if (sum == PROCESS_NUM):
            break
    tic = time.time()
    res = m1(inputs1)
    toc = time.time()
    print("share ", index, " ", res)
    print("share ", index, "spend time:", toc-tic, "s")

if __name__=="__main__":
    processes = [Process(target=thread_infer, args=(i,)) for i in range(PROCESS_NUM)]
    for process in processes:
        process.start()

    for process in processes:
        process.join()
```
