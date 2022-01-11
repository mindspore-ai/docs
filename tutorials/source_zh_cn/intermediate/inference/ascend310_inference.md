# Ascend310处理器上推理MindIR模型

`Ascend` `进阶` `推理应用`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/tutorials/source_zh_cn/intermediate/inference/ascend310_inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

本文介绍如何在Ascend310处理器中推理MindIR模型。Ascend环境配置可参考[Ascend安装指南](https://www.mindspore.cn/install/)，完整推理代码可参考[ascend310_resnet50_preprocess_sample](https://gitee.com/mindspore/docs/tree/r1.6/docs/sample_code/ascend310_resnet50_preprocess_sample)。

## 推理代码介绍

推理部分使用了CPU算子来进行数据的预处理，然后完成推理。整体代码存放在`main.cc`文件中，现在对其中的功能实现进行说明。

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

加载模型文件：

```c++
// 加载MindIR模型
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// 图编译
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
```

获取模型所需输入信息：

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

加载图片文件：

```c++
ms::MSTensor ReadFile(const std::string &file);
auto image = ReadFile(image_file);
```

图片预处理（使用CPU算子）：

```c++
// 对图片进行解码，变为RGB格式，并重设尺寸
std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({256}));
// 输入归一化
std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize(
    {0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
// 剪裁图片
std::shared_ptr<ds::TensorTransform> center_crop(new ds::vision::CenterCrop({224, 224}));
// shape (H, W, C) 变为 shape (C, H, W)
std::shared_ptr<ds::TensorTransform> hwc2chw(new ds::vision::HWC2CHW());

// 定义preprocessor
ds::Execute preprocessor({decode, resize, normalize, center_crop, hwc2chw});

// 调用函数，获取处理后的图像
ret = preprocessor(image, &image);
```

执行推理：

```c++
// 创建输入输出向量
std::vector<ms::MSTensor> outputs;
std::vector<ms::MSTensor> inputs;
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                    image.Data().get(), image.DataSize());
// 执行推理
ret = resnet50.Predict(inputs, &outputs);
```

获取推理结果：

```c++
// 获取推理结果的最大概率
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

## 构建脚本介绍

构建脚本用于构建用户程序，完整代码位于`CMakeLists.txt` ，下面进行解释说明。

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
```

## 编译并执行推理代码

若运行完整推理代码[ascend310_resnet50_preprocess_sample](https://gitee.com/mindspore/docs/tree/r1.6/docs/sample_code/ascend310_resnet50_preprocess_sample)，可将实验的脚本下载至Ascend310环境中编译并执行。

创建并进入工程目录`ascend310_resnet50_preprocess_sample`，执行`cmake`命令，其中`pip3`需要按照实际情况修改：

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
```

再执行`make`命令编译即可。

```bash
make
```

编译成功后，会获得`resnet50_sample`可执行文件。在工程目录`ascend310_resnet50_preprocess_sample`下创建`model`目录放置MindIR文件[resnet50_imagenet.mindir](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir)。此外，创建`test_data`目录用于存放待分类的图片，图片可来自ImageNet等各类开源数据集，输入执行命令即可获取推理结果：

```bash
./resnet50_sample
```
