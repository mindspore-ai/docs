# Performing Cloud-side Distributed Inference Using Python Interface

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/mindir/runtime_distributed_python.md)

## Overview

For scenarios where large-scale neural network models have many parameters and cannot be fully loaded into a single device for inference, distributed inference can be performed using multiple devices. This tutorial describes how to perform MindSpore Lite cloud-side distributed inference using the [Python interface](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite.html). Cloud-side distributed inference is roughly the same process as [Cloud-side single-card inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_python.html) and can be cross-referenced. MindSpore Lite cloud-side distributed inference has more optimization for performance aspects.

MindSpore Lite cloud-side distributed inference is only supported to run in Linux environment deployments with Atlas training series and Nvidia GPU as the supported device types. As shown in the figure below, the distributed inference is currently initiated by a multi-process approach, where each process corresponds to a `Rank` in the communication set, loading, compiling and executing the respective sliced model, with the same input data for each process.

![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/images/lite_runtime_distributed.png)

Each process consists of the following main steps:

1. Model reading: Slice and export the distributed MindIR model via MindSpore. The number of MindIR models is the same as the number of devices for loading to each device for inference.
2. Context creation and configuration: Create and configure the [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context), and hold the distributed inference parameters to guide distributed model compilation and model execution.
3. Model loading and compilation: Use the [Model.build_from_file](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface for model loading and model compilation. The model loading phase parses the file cache into a runtime model. The model compilation phase optimizes the front-end computational graph into a high-performance back-end computational graph. The process is time-consuming and it is recommended to compile once and inference multiple times.
4. Model input data padding.
5. Distributed inference execution: use the [Model.predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.predict) interface for model distributed inference.
6. Model output data obtaining.
7. Multi-process execution of distributed inference programs.

## Preparation

1. To download the cloud-side distributed inference python sample code, please select the device type: [Ascend](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_ge_distributed_cpp) or [GPU](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/gpu_trt_distributed_cpp). The directory will be referred to later as the example code directory.

2. Slice and export the distributed MindIR model via MindSpore and store it to the sample code directory. For a quick experience, you can download the two sliced Matmul model files [Matmul0.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/Matmul0.mindir), [Matmul1.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/Matmul1.mindir).

3. For Ascend device type, generate the networking information file through hccl_tools.py as needed, store it in the sample code directory, and fill the path of the file into the configuration file `config_file.ini` in the sample code directory.

4. Download the MindSpore Lite cloud-side inference installation package [mindspore-lite-{version}-linux-{arch}.whl](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html), store it to the sample code directory, and install it via `pip`.

The main steps of MindSpore Lite cloud-side distributed inference will be described in the subsequent sections in conjunction with the code, and please refer to `main.py` in the sample code directory for the complete code.

## Creating Contextual Configuration

The contextual configuration holds the required basic configuration parameters and distributed inference parameters to guide model compilation and model distributed execution. The following sample code demonstrates how to create a context through [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context).

```python
# init context
context = mslite.Context()
```

Ascend, Nvidia GPU are supported in distributed inference scenarios, and can be specified by [Context.target](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context.target) to specify the device to run.

### Configuring Ascend Device Context

When the device type is Ascend (Atlas training series is currently supported by distributed inference), set [Context.target](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context.target) to `Ascend` and set `DeviceID`, `RankID` by the following way. Since Ascend provides multiple inference engine backends, currently only the `ge` backend supports distributed inference, and the Ascend inference engine backend is specified as `ge` by via `ascend.provider`.The sample code is as follows.

```python
# set Ascend target and distributed info
context.target = ["Ascend"]
context.ascend.device_id = args.device_id
context.ascend.rank_id = args.rank_id
context.ascend.provider = "ge"
```

### Configuring GPU Device Context

When the device type is GPU, set [Context.target](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Context.html#mindspore_lite.Context.target) as `gpu`. The distributed inference multi-process application for GPU devices is pulled up by mpi, which automatically sets the `RankID` of each process, and the user only needs to specify `CUDA_VISIBLE_DEVICES` in the environment variable, without specifying the group information file. Therefore, the `RankID` of each process can be used as `DeviceID`. In addition, GPU also provides multiple inference engine backends. Currently only `tensorrt` backend supports distributed inference, and the GPU inference engine backend is specified as `tensorrt` by `gpu.provider`. The sample code is as follows.

```python
# set GPU target and distributed info
context.target = ["gpu"]
context.gpu.device_id = context.gpu.rank_id
context.gpu.provider = "tensorrt"
```

## Model Creation, Loading and Compilation

Consistent with [MindSpore Lite Cloud-side Single Card Inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_cpp.html), the main entry point for distributed inference is the [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface for model loading, compilation and execution. Create [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model) and call the [Model.build_from_file](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.build_from_file) interface to implement the model Loading and model compilation, the sample code is as follows.

```python
# create Model and build Model
model = mslite.Model()
model.build_from_file(model_path, mslite.ModelType.MINDIR, context, args.config_file)
```

## Model Input Data Padding

First, use the [Model.get_inputs](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Model.html#mindspore_lite.Model.get_inputs) method to get all the input [Tensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Tensor.html#mindspore_lite.Tensor), and fill in the Host data through the related interface. The sample code is as follows.

```python
# set model input as ones
inputs = model.get_inputs()
for input_i in inputs:
    input_i.set_data_from_numpy(np.ones(input_i.shape, dtype=np.float32))
```

MindSpore Lite input can also be constructed in the following way.

```python
# np_inputs is a list or tuple of numpy array
inputs = [mslite.Tensor(np_input) for np_input in np_inputs]
```

## Distributed Inference Execution

Call the [Model.predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface to perform distributed inference, with the following sample code.

```python
# execute inference
outputs = model.predict(inputs)
```

## Model Output Data Obtaining

The model output data is stored in the output [Tensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Tensor.html#mindspore_lite.Tensor) defined in the previous step, and the output data can be accessed through the relevant interface. The following example code shows how to access the output data and print it.

```python
# get output and print
for output in outputs:
    name = output.name.rstrip()
    data_size = output.data_size
    element_num = output.element_num
    print("tensor's name is:%s data size is:%s tensor elements num is:%s" %
          (name, data_size, element_num))
    data = output.get_data_to_numpy()
    data = data.flatten()
    print("output data is:", end=" ")
    for i in range(10):
        print(data[i], end=" ")
    print("")
```

## Performing Distributed Inference Example

Start the distributed inference in the following multi-process manner. Please refer to `run.sh` in the sample code directory for the complete run command. When run successfully, the name, data size, number of elements and the first 10 elements of each output `Tensor` will be printed.

```bash
# for Ascend, run the executable file for each rank using shell commands
python3 ./ascend_ge_distributed.py --model_path=/your/path/to/Matmul0.mindir --device_id=0 --rank_id=0 --config_file=./config_file.ini &
python3 ./ascend_ge_distributed.py --model_path=/your/path/to/Matmul1.mindir --device_id=1 --rank_id=1 --config_file=./config_file.ini

# for GPU, run the executable file for each rank using mpi
RANK_SIZE=2
mpirun -n $RANK_SIZE python3 ./main.py --model_path=/your/path/to/Matmul.mindir
```

## Multiple Models Sharing Weights

In the Ascend device and graph compilation grade O2 scenario, a single card can deploy multiple models, and models deployed to the same card can share weights. For details, please refer to [Advanced Usage - Multiple Model Sharing Weights](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_cpp.html#multiple-models-sharing-weights).
