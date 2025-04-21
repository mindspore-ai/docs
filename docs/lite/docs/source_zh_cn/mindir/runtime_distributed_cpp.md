# 使用C++接口执行云侧分布式推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/mindir/runtime_distributed_cpp.md)

## 概述

针对大规模神经网络模型参数多、无法完全加载至单设备推理的场景，可利用多设备进行分布式推理。本教程介绍如何使用[C++接口](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/index.html)执行MindSpore Lite云侧分布式推理。云侧分布式推理与[云侧单卡推理](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_cpp.html)流程大致相同，可以相互参考。MindSpore Lite云侧分布式推理针对性能方面具有更多的优化。

MindSpore Lite云侧分布式推理仅支持在Linux环境部署运行，支持的设备类型为Atlas训练系列产品和Nvidia GPU。如下图所示，当前通过多进程方式启动分布式推理，每个进程对应通信集合中的一个`Rank`，对各自已切分的模型进行加载、编译与执行，每个进程输入数据相同。

![img](./images/lite_runtime_distributed.png)

每个进程主要包括以下步骤：

1. 模型读取：通过MindSpore切分，并导出分布式MindIR模型，MindIR模型数量与设备数相同，用于加载到各个设备进行推理。
2. 上下文创建与配置：创建并配置上下文[Context](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#context)，保存分布式推理参数，用于指导分布式模型编译和模型执行。
3. 模型加载与编译：使用[Model::Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build-2)接口进行模型加载和模型编译。模型加载阶段将文件缓存解析成运行时的模型。模型编译阶段将前端计算图优化为高性能后端计算图，该过程耗时较长，建议一次编译，多次推理。
4. 模型输入数据填充。
5. 分布式推理执行：使用[Model::Predict](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#predict)接口进行模型分布式推理。
6. 模型输出数据获取。
7. 编译和多进程执行分布式推理程序。

## 准备工作

1. 下载云侧分布式推理C++示例代码，请选择设备类型：[Ascend](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_ge_distributed_cpp)或[GPU](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/gpu_trt_distributed_cpp)。后文将该目录称为示例代码目录。

2. 通过MindSpore切分，并导出分布式MindIR模型，将其存放至示例代码目录。如需快速体验，可下载已切分的两个Matmul模型文件[Matmul0.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/Matmul0.mindir)、[Matmul1.mindir](https://download.mindspore.cn/model_zoo/official/lite/quick_start/Matmul1.mindir)。

3. 对于Ascend设备类型，通过hccl_tools.py按照需要生成组网信息文件，存放至示例代码目录，并将该文件路径填入示例代码目录下配置文件 `config_file.ini` 中。

4. 下载MindSpore Lite云侧推理安装包[mindspore-lite-{version}-linux-{arch}.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/use/downloads.html)，存放至示例代码目录。解压该安装包，参考快速入门的环境变量设置环境变量，

后续章节将结合代码讲述MindSpore Lite云侧分布式推理主要步骤，完整代码请参考示例代码目录下`main.cc`。

## 创建上下文配置

上下文配置保存了所需基本配置参数与分布式推理参数，用于指导模型编译和模型分布式执行。如下示例代码演示如何通过[Context](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#context)创建上下文，通过[Context::MutableDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mutabledeviceinfo)指定运行设备。

```c++
// Create and init context, add Ascend device info
auto context = std::make_shared<mindspore::Context>();
if (context == nullptr) {
    std::cerr << "New context failed." << std::endl;
    return nullptr;
}
auto &device_list = context->MutableDeviceInfo();
```

分布式推理场景下支持[AscendDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#ascenddeviceinfo)、[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#gpudeviceinfo)，分别用于设置Ascend与Nvidia GPU上下文信息。

### 配置Ascend设备上下文

当设备类型为Ascend时(目前分布式推理支持Atlas训练系列产品)，新建[AscendDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#ascenddeviceinfo)，并通过[AscendDeviceInfo::SetDeviceID](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#ascenddeviceinfo)、[AscendDeviceInfo::SetRankID](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#ascenddeviceinfo)分别设置`DeviceID`、`RankID`。由于Ascend提供多个推理引擎后端，当前仅`ge`后端支持分布式推理，通过[DeviceInfoContext::SetProvider](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#deviceinfocontext)指定Ascend推理引擎后端为`ge`。示例代码如下：

```c++
// for Atlas training series
auto device_info = std::make_shared<mindspore::AscendDeviceInfo>();
if (device_info == nullptr) {
  std::cerr << "New AscendDeviceInfo failed." << std::endl;
  return nullptr;
}
// Set Atlas training series device id， rank id and provider.
device_info->SetDeviceID(device_id);
device_info->SetRankID(rank_id);
device_info->SetProvider("ge");
// Device context needs to be push_back into device_list to work.
device_list.push_back(device_info);
```

### 配置使用GPU设备上下文

当设备类型为GPU时，新建[GPUDeviceInfo](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#gpudeviceinfo)。GPU设备的分布式推理多进程应用由mpi拉起，mpi会自动设置每个进程的`RankID`，用户只需在环境变量中指定`CUDA_VISIBLE_DEVICES`，无需指定组网信息文件。因此，每个进程的`RankID`可以当作`DeviceID`使用。另外，GPU也提供多个推理引擎后端，当前仅`tensorrt`后端支持分布式推理，通过[DeviceInfoContext::SetProvider](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#deviceinfocontext)指定GPU推理引擎后端为`tensorrt`。示例代码如下：

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

## 模型创建、加载与编译

与[MindSpore Lite云侧单卡推理](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_cpp.html)一致，分布式推理的主入口是[Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#model)接口，可进行模型加载、编译和执行。对于Ascend设备，使用[Model::LoadConfig](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#loadconfig)接口载入配置文件[config_file.ini](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/examples/cloud_infer/ascend_ge_distributed_cpp/config_file.ini)，GPU设备则不需要。最后，调用[Model::Build](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build-2)接口来实现模型加载与模型编译，示例代码如下：

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

## 模型输入数据填充

首先，使用[Model::GetInputs](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#getinputs)方法获取所有输入`Tensor`，通过[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)相关接口将Host数据填入。示例代码如下：

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

## 分布式推理执行

创建模型输出`Tensor`，类型为[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)。调用[Model::Predict](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#predict)接口执行分布式推理，示例代码如下：

```c++
// Model Predict
std::vector<mindspore::MSTensor> outputs;
auto predict_ret = model.Predict(inputs, &outputs);
if (predict_ret != mindspore::kSuccess) {
  std::cerr << "Predict error " << predict_ret << std::endl;
  return -1;
}
```

## 模型输出数据获取

模型输出数据保存在上一步定义的输出`Tensor`中，通过[MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#mstensor)相关接口可访问输出数据。如下示例代码展示了如何访问输出数据并打印。

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

## 编译和执行分布式推理样例

在样例代码目录下，按照如下方式编译样例。完整命令请参考示例代码目录下`build.sh`。

```bash
mkdir -p build
cd build || exit
cmake ..
make
```

在编译成功后，在`build`目录下得到`{device_type}_{backend}_distributed_cpp`可执行程序，按照如下多进程方式启动分布式推理。完整运行命令请参考示例代码目录下`run.sh`。运行成功后，将打印每个输出`Tensor`的名称、数据大小、元素个数与前10个元素值。

```bash
# for Ascend, run the executable file for each rank using shell commands
./build/ascend_ge_distributed /your/path/to/Matmul0.mindir 0 0 ./config_file.ini &
./build/ascend_ge_distributed /your/path/to/Matmul1.mindir 1 1 ./config_file.ini

# for GPU, run the executable file for each rank using mpi
RANK_SIZE=2
mpirun -n $RANK_SIZE ./build/gpu_trt_distributed /your/path/to/Matmul.mindir
```

## 多模型共享权重

Ascend设备图编译等级为O2的场景下，单个卡可以部署多个模型，部署到同一张卡的模型可以共享权重，详情可参考[高级用法-多模型共享权重](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/mindir/runtime_cpp.html#%E5%A4%9A%E6%A8%A1%E5%9E%8B%E5%85%B1%E4%BA%AB%E6%9D%83%E9%87%8D)。
