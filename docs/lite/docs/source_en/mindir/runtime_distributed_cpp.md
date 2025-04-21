# Performing Cloud-side Distributed Inference Using C++ Interface

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/mindir/runtime_distributed_cpp.md)

## Overview

For scenarios where large-scale neural network models have many parameters and cannot be fully loaded into a single device for inference, distributed inference can be performed using multiple devices. This tutorial describes how to perform MindSpore Lite cloud-side distributed inference using the [C++ interface](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/index.html). Cloud-side distributed inference is roughly the same process as [Cloud-side single-card inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_cpp.html) and can be cross-referenced. MindSpore Lite cloud-side distributed inference has more optimization for performance aspects.

MindSpore Lite cloud-side distributed inference is only supported to run in Linux environment deployments with Atlas training series and Nvidia GPU as the supported device types. As shown in the figure below, the distributed inference is currently initiated by a multi-process approach, where each process corresponds to a `Rank` in the communication set, loading, compiling and executing the respective sliced model, with the same input data for each process.

![img](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/images/lite_runtime_distributed.png)

Each process consists of the following main steps:

1. Model reading: Slice and export the distributed MindIR model via MindSpore. The number of MindIR models is the same as the number of devices for loading to each device for inference.
2. Context creation and configuration: Create and configure the [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Context.html), and hold the distributed inference parameters to guide distributed model compilation and model execution.
3. Model loading and compilation: Use the [Model::Build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface for model loading and model compilation. The model loading phase parses the file cache into a runtime model. The model compilation phase optimizes the front-end computational graph into a high-performance back-end computational graph. The process is time-consuming and it is recommended to compile once and inference multiple times.
4. Model input data padding.
5. Distributed inference execution: use the [Model::Predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface for model distributed inference.
6. Model output data obtaining.
7. Compilation and multi-process execution of distributed inference programs.

## Preparation

1. To download the cloud-side distributed inference C++ sample code, please select the device type: [Ascend](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_ge_distributed_cpp) or [GPU](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/gpu_trt_distributed_cpp). The directory will be referred to later as the example code directory.

2. Slice and export the distributed MindIR model via MindSpore and store it to the sample code directory. For a quick experience, you can download the two sliced Matmul model files [Matmul0.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/Matmul0.mindir), [Matmul1.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/Matmul1.mindir).

3. For Ascend device type, generate the networking information file through hccl_tools.py as needed, store it in the sample code directory, and fill the path of the file into the configuration file `config_file.ini` in the sample code directory.

4. Download the MindSpore Lite cloud-side inference installation package [mindspore-lite-{version}-linux-{arch}.tar.gz](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) and store it to the sample code directory. Unzip this installation package and refer to the Environment Variables in the Quick Start to set environment variables.

The main steps of MindSpore Lite cloud-side distributed inference will be described in the subsequent sections in conjunction with the code, and please refer to `main.cc` in the sample code directory for the complete code.

## Creating Contextual Configuration

The contextual configuration holds the required basic configuration parameters and distributed inference parameters to guide model compilation and model distributed execution. The following sample code demonstrates how to create a context through [Context](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Context.html) and specify a running device through [Context:. MutableDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Context.html) to specify the running device.

```c++
// Create and init context, add Ascend device info
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return nullptr;
}
auto &device_list = context->MutableDeviceInfo();
```

Distributed inference scenarios support [AscendDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_AscendDeviceInfo.html), [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_GPUDeviceInfo.html), which are used to set Ascend and Nvidia GPU context information respectively.

### Configuring Ascend Device Context

When the device type is Ascend (Atlas training series is currently supported by distributed inference), a new [AscendDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_AscendDeviceInfo.html) is created, and set `DeviceID`, `RankID` respectively by [AscendDeviceInfo::SetDeviceID](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_AscendDeviceInfo.html), [AscendDeviceInfo::SetRankID](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_AscendDeviceInfo.html). Since Ascend provides multiple inference engine backends, currently only the `ge` backend supports distributed inference, and the Ascend inference engine backend is specified as `ge` by [DeviceInfoContext::SetProvider](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_DeviceInfoContext.html). The sample code is as follows.

```c++
// for Atlas training series
auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
if (device_info == nullptr) {
  std::cerr << "New AscendDeviceInfo failed." << std::endl;
  return nullptr;
}
// Set Atlas training series device id, rank id and provider.
device_info->SetDeviceID(device_id);
device_info->SetRankID(rank_id);
device_info->SetProvider("ge");
// Device context needs to be push_back into device_list to work.
device_list.push_back(device_info);
```

### Configuring GPU Device Context

When the device type is GPU, new [GPUDeviceInfo](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_GPUDeviceInfo.html) is created. The distributed inference multi-process application for GPU devices is pulled up by mpi, which automatically sets the `RankID` of each process, and the user only needs to specify `CUDA_VISIBLE_DEVICES` in the environment variable, without specifying the group information file. Therefore, the `RankID` of each process can be used as `DeviceID`. In addition, GPU also provides multiple inference engine backends. Currently only `tensorrt` backend supports distributed inference, and the GPU inference engine backend is specified as `tensorrt` by [DeviceInfoContext::SetProvider](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_DeviceInfoContext.html). The sample code is as follows.

```c++
// for GPU
auto device_info = std::make_shared<mindspore::GPUDeviceInfo>();
if (device_info == nullptr) {
  std::cerr << "New GPUDeviceInfo failed." << std::endl;
  return -1;
}

// set distributed info
auto rank_id = device_info->GetRankID();  // rank id is returned from mpi
device_info->SetDeviceID(rank_id);  // as we set visible device id in env, we use rank id as device id
device_info->SetProvider("tensorrt");
device_list.push_back(device_info);
```

## Model Creation, Loading and Compilation

Consistent with [MindSpore Lite Cloud-side Single Card Inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_cpp.html), the main entry point for distributed inference is the [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface for model loading, compilation and execution. For Ascend devices, use the [Model::LoadConfig](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface to load the configuration file [config_file.ini](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_ge_distributed_cpp/config_file.ini), which is not required for GPU devices. Finally, call the [Model::Build](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface to implement model loading and model compilation, and the sample code is as follows.

```c++
mindspore::Model model;
// Load config file for Atlas training series
if (!config_path.empty()) {
  if (model.LoadConfig(config_path) != mindspore::kSuccess) {
    std::cerr << "Failed to load config file " << config_path << std::endl;
    return -1;
  }
}

// Build model
auto build_ret = model.Build(model_path, mindspore::kMindIR, context);
if (build_ret != mindspore::kSuccess) {
  std::cerr << "Build model error " << build_ret << std::endl;
  return -1;
}
```

## Model Input Data Padding

First, use the [Model::GetInputs](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) method to get all the input `Tensor`, and fill in the Host data through the [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_MSTensor.html)-related interface. The sample code is as follows.

```c++
// helper function
template <typename T, typename Distribution>
void GenerateRandomData(int size, void *data, Distribution distribution) {
  std::mt19937 random_engine;
  int elements_num = size / sizeof(T);
  (void)std::generate_n(static_cast<T *>(data), elements_num,
                        [&distribution, &random_engine]() { return static_cast<T>(distribution(random_engine)); });
}

// Get input tensor pointer and write random data
int GenerateInputDataWithRandom(std::vector<mindspore::MSTensor> inputs) {
  for (auto tensor : inputs) {
    auto input_data = tensor.MutableData();
    if (input_data == nullptr) {
      std::cerr << "MallocData for inTensor failed." << std::endl;
      return -1;
    }
    GenerateRandomData<float>(tensor.DataSize(), input_data, std::uniform_real_distribution<float>(1.0f, 1.0f));
  }
  return 0;
}

// Get Input
auto inputs = model.GetInputs();
// Generate random data as input data.
if (GenerateInputDataWithRandom(inputs) != 0) {
  std::cerr << "Generate Random Input Data failed." << std::endl;
  return -1;
}
```

## Distributed Inference Execution

Create a model output `Tensor` of type [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_MSTensor.html). Call the [Model::Predict](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_Model.html) interface to perform distributed inference, with the following sample code.

```c++
// Model Predict
std::vector<mindspore::MSTensor> outputs;
auto predict_ret = model.Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Predict error " << predict_ret << std::endl;
  return -1;
}
```

## Model Output Data Obtaining

The model output data is stored in the output `Tensor` defined in the previous step, and the output data is accessible through the [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/generate/classmindspore_MSTensor.html)-related interface. The following example code shows how to access the output data and print it.

```c++
// Print Output Tensor Data.
constexpr int kNumPrintOfOutData = 10;
for (auto &tensor : outputs) {
  std::cout << "tensor name is:" << tensor.Name() << " tensor size is:" << tensor.DataSize()
            << " tensor elements num is:" << tensor.ElementNum() << std::endl;
  auto out_data = reinterpret_cast<const float *>(tensor.Data().get());
  std::cout << "output data is:";
  for (int i = 0; i < tensor.ElementNum() && i <= kNumPrintOfOutData; i++) {
    std::cout << out_data[i] << " ";
  }
  std::cout << std::endl;
}
```

## Compiling and Performing Distributed Inference Example

In the sample code directory, compile the sample in the following way. Please refer to `build.sh` in the sample code directory for the complete command.

```bash
mkdir -p build
cd build || exit
cmake ..
make
```

After successful compilation, the `{device_type}_{backend}_distributed_cpp` executable programs is obtained in the `build` directory, and the distributed inference is started in the following multi-process manner. Please refer to `run.sh` in the sample code directory for the complete run command. When run successfully, the name, data size, number of elements and the first 10 elements of each output `Tensor` will be printed.

```bash
# for Ascend, run the executable file for each rank using shell commands
./build/ascend_ge_distributed /your/path/to/Matmul0.mindir 0 0 ./config_file.ini &
./build/ascend_ge_distributed /your/path/to/Matmul1.mindir 1 1 ./config_file.ini

# for GPU, run the executable file for each rank using mpi
RANK_SIZE=2
mpirun -n $RANK_SIZE ./build/gpu_trt_distributed /your/path/to/Matmul.mindir
```

## Multiple Models Sharing Weights

In the Ascend device and graph compilation grade O2 scenario, a single card can deploy multiple models, and models deployed to the same card can share weights. For details, please refer to [Advanced Usage - Multiple Model Sharing Weights](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/runtime_cpp.html#multiple-models-sharing-weights).
