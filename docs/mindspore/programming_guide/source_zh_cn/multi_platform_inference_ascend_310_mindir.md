# Ascend 310 AI处理器上使用MindIR模型进行推理

`Ascend` `推理应用`

<!-- TOC -->

- [Ascend 310 AI处理器上使用MindIR模型进行推理](#ascend-310-ai处理器上使用mindir模型进行推理)
    - [概述](#概述)
    - [开发环境准备](#开发环境准备)
    - [导出MindIR模型文件](#导出mindir模型文件)
    - [推理目录结构介绍](#推理目录结构介绍)
    - [推理代码介绍](#推理代码介绍)
        - [需要手动定义预处理的模型推理方式：main.cc](#需要手动定义预处理的模型推理方式maincc)
            - [使用CPU算子数据预处理](#使用cpu算子数据预处理)
            - [使用Ascend 310算子数据预处理](#使用ascend-310算子数据预处理)
        - [免手动定义预处理的模型推理方式：main_hide_preprocess.cc](#免手动定义预处理的模型推理方式main_hide_preprocesscc)
    - [构建脚本介绍](#构建脚本介绍)
    - [编译推理代码](#编译推理代码)
    - [执行推理并查看结果](#执行推理并查看结果)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/multi_platform_inference_ascend_310_mindir.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

Ascend 310是面向边缘场景的高能效高集成度AI处理器，本教程介绍如何在Ascend 310上使用MindSpore基于MindIR模型文件执行推理，主要包括以下流程：

1. 导出MindIR模型文件，这里以ResNet-50模型为例。

2. 编译推理代码，生成可执行`main`文件。

3. 加载保存的MindIR模型，执行推理并查看结果。

> 你可以在这里找到完整可运行的样例代码：<https://gitee.com/mindspore/docs/tree/master/docs/sample_code/ascend310_resnet50_preprocess_sample> 。

## 开发环境准备

参考[安装指导](https://www.mindspore.cn/install)准备Ascend环境与MindSpore。

## 导出MindIR模型文件

在CPU/GPU/Ascend 910的机器上训练好目标网络，并保存为CheckPoint文件，通过网络和CheckPoint文件导出对应的MindIR格式模型文件，导出流程参见[导出MindIR格式文件](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html#mindir)。

> 这里提供使用BatchSize为1的ResNet-50模型导出的示例MindIR文件[resnet50_imagenet.mindir](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir)。

## 推理目录结构介绍

创建目录放置推理代码工程，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample`，可以从官网示例下载[样例代码](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/ascend310_resnet50_preprocess_sample)，`model`目录用于存放上述导出的`MindIR`模型文件，`test_data`目录用于存放待分类的图片，推理代码工程目录结构如下:

```text
└─ascend310_resnet50_preprocess_sample
    ├── CMakeLists.txt                    // 构建脚本
    ├── README.md                         // 使用说明
    ├── main.cc                           // 主函数
    ├── main_hide_preprocess.cc           // 主函数2，免预处理代码的推理方式（已嵌入到MindIR中）
    ├── model
    │   └── resnet50_imagenet.mindir      // MindIR模型文件
    └── test_data
        ├── ILSVRC2012_val_00002138.JPEG  // 输入样本图片1
        ├── ILSVRC2012_val_00003014.JPEG  // 输入样本图片2
        ├── ...                           // 输入样本图片n
```

## 推理代码介绍

### 需要手动定义预处理的模型推理方式：main.cc

#### 使用CPU算子数据预处理

推理代码样例：<https://gitee.com/mindspore/docs/blob/master/docs/sample_code/ascend310_resnet50_preprocess_sample/main.cc> 。

引用`mindspore`和`mindspore::dataset`的名字空间。

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

环境初始化，指定硬件为Ascend 310，DeviceID为0：

```c++
auto context = std::make_shared<ms::Context>();
auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
ascend310_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend310_info);
```

加载模型文件:

```c++
// Load MindIR model
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// Build model with graph object
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
```

获取模型所需输入信息：

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

加载图片文件:

```c++
// Readfile is a function to read images
ms::MSTensor ReadFile(const std::string &file);
auto image = ReadFile(image_file);
```

图片预处理（使用CPU算子）:

```c++
// Create the CPU operator provided by MindData to get the function object

// Decode the input to RGB format
std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
// Resize the image to the given size
std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({256}));
// Normalize the input
std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize(
    {0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
// Crop the input image at the center
std::shared_ptr<ds::TensorTransform> center_crop(new ds::vision::CenterCrop({224, 224}));
// shape (H, W, C) to shape (C, H, W)
std::shared_ptr<ds::TensorTransform> hwc2chw(new ds::vision::HWC2CHW());

// // Define a MindData preprocessor
ds::Execute preprocessor({decode, resize, normalize, center_crop, hwc2chw});

// Call the function object to get the processed image
ret = preprocessor(image, &image);
```

执行推理:

```c++
// Create outputs vector
std::vector<ms::MSTensor> outputs;
// Create inputs vector
std::vector<ms::MSTensor> inputs;
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                    image.Data().get(), image.DataSize());
// Call the Predict function of Model for inference
ret = resnet50.Predict(inputs, &outputs);
```

获取推理结果:

```c++
// Output the maximum probability to the screen
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

#### 使用Ascend 310算子数据预处理

Dvpp模块为Ascend 310芯片内置硬件解码器，相较于CPU拥有对图形处理更强劲的性能。支持JPEG图片的解码缩放等基础操作。

引用`mindspore`和`mindspore::dataset`的名字空间。

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

环境初始化，指定硬件为Ascend 310，DeviceID为0：

```c++
auto context = std::make_shared<ms::Context>();
auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
ascend310_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend310_info);
```

加载图片文件:

```c++
// Readfile is a function to read images
ms::MSTensor ReadFile(const std::string &file);
auto image = ReadFile(image_file);
```

图片预处理（使用Ascend 310算子）:

```c++
// Create the Dvpp operator provided by MindData to get the function object

// Decode the input to YUV420 format
std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
// Resize the image to the given size
std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({256}));
// Normalize the input
std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize(
    {0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
// Crop the input image at the center
std::shared_ptr<ds::TensorTransform> center_crop(new ds::vision::CenterCrop({224, 224}));
```

图片预处理（使用Ascend 310算子， 性能为CPU算子的2.3倍），需显式指定计算硬件为Ascend 310。

```c++
// Define a MindData preprocessor, set deviceType = kAscend310， device id = 0
ds::Execute preprocessor({decode, resize, center_crop, normalize}, MapTargetDevice::kAscend310, 0);

// Call the function object to get the processed image
ret = preprocessor(image, &image);
```

加载模型文件: 若使用Ascend 310算子，则需要为模型插入Aipp算子。

```c++
// Load MindIR model
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// Build model with graph object
ascend310_info->SetInsertOpConfigPath(preprocessor.AippCfgGenerator());
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
```

获取模型所需输入信息：

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

执行推理:

```c++
// Create outputs vector
std::vector<ms::MSTensor> outputs;
// Create inputs vector
std::vector<ms::MSTensor> inputs;
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                    image.Data().get(), image.DataSize());
// Call the Predict function of Model for inference
ret = resnet50.Predict(inputs, &outputs);
```

获取推理结果:

```c++
// Output the maximum probability to the screen
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

### 免手动定义预处理的模型推理方式：main_hide_preprocess.cc

> 注意：目前只支持CV类的模型

推理代码样例：<https://gitee.com/mindspore/docs/blob/master/docs/sample_code/ascend310_resnet50_preprocess_sample/main_hide_preprocess.cc> 。

引用`mindspore`和`mindspore::dataset`的名字空间。

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

环境初始化，指定硬件为Ascend 310，DeviceID为0：

```c++
auto context = std::make_shared<ms::Context>();
auto ascend310_info = std::make_shared<ms::Ascend310DeviceInfo>();
ascend310_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend310_info);
```

加载模型文件:

```c++
// Load MindIR model
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// Build model with graph object
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
```

获取模型所需输入信息：

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

提供图片文件，一键执行预处理与模型推理：

```c++
std::vector<MSTensor> inputs = {ReadFile(image_path)};
std::vector<MSTensor> outputs;
ret = resnet50.PredictWithPreprocess(inputs, &outputs);
```

获取推理结果：

```c++
// 获取推理结果的最大概率
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

## 构建脚本介绍

构建脚本用于构建用户程序，样例来自于：<https://gitee.com/mindspore/docs/blob/master/docs/sample_code/ascend310_resnet50_preprocess_sample/CMakeLists.txt> 。

为编译器添加头文件搜索路径：

```cmake
option(MINDSPORE_PATH "mindspore install path" "")
include_directories(${MINDSPORE_PATH})
include_directories(${MINDSPORE_PATH}/include)
```

在MindSpore中查找所需动态库：

```cmake
find_library(MS_LIB libmindspore.so ${MINDSPORE_PATH}/lib)
file(GLOB_RECURSE MD_LIB ${MINDSPORE_PATH}/_c_dataengine*)
```

使用指定的源文件生成目标可执行文件，并为目标文件链接MindSpore库：

```cmake
add_executable(resnet50_sample main.cc)
target_link_libraries(resnet50_sample ${MS_LIB} ${MD_LIB})

add_executable(resnet50_hide_preprocess main_hide_preprocess.cc)
target_link_libraries(resnet50_hide_preprocess ${MS_LIB} ${MD_LIB})
```

## 编译推理代码

进入工程目录`ascend310_resnet50_preprocess_sample`，设置如下环境变量：

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}
# if MindSpore is installed by binary, run "export LD_LIBRARY_PATH=path-to-your-custom-dir:${LD_LIBRARY_PATH}"

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

执行`cmake`命令，其中`pip3`需要按照实际情况修改：

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
# if MindSpore is installed by binary, run "cmake . -DMINDSPORE_PATH=path-to-your-custom-dir"
```

再执行`make`命令编译即可。

```bash
make
```

编译完成后，在`ascend310_resnet50_preprocess_sample`下会生成可执行`main`文件。

## 执行推理并查看结果

登录Ascend 310环境，创建`model`目录放置MindIR文件`resnet50_imagenet.mindir`，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample/model`。
创建`test_data`目录放置图片，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample/test_data`。
就可以开始执行推理了:

如果使用的MindIR，在导出时是不带数据预处理的，可以执行该主函数：

```bash
./resnet50_sample
```

执行后，会对`test_data`目录下放置的所有图片进行推理，比如放置了9张[ImageNet2012](http://image-net.org/download-images)验证集中label为0的图片，可以看到推理结果如下。

```text
Image: ./test_data/ILSVRC2012_val_00002138.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00003014.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00006697.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00007197.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009111.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009191.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009346.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009379.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00009396.JPEG infer result: 0
```

如果使用的MindIR，在导出时是带数据预处理的，可以执行该主函数：

```bash
./resnet50_hide_preprocess
```

执行后，会对`test_data`目录下放置的ILSVRC2012_val_00002138.JPEG图片（在main_hide_preprocess.cc中可配置）进行推理，可以看到推理结果如下。

```text
Image: ./test_data/ILSVRC2012_val_00002138.JPEG infer result: 0
```
