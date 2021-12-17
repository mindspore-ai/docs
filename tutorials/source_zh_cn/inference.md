# 推理

`Ascend` `端侧` `入门` `推理应用`

<a href="https://gitee.com/mindspore/docs/blob/r1.5/tutorials/source_zh_cn/inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

本节是初级教程的最后一节，介绍使用昇腾AI处理器推理。

昇腾（Ascend）AI处理器是面向边缘场景的高能效高集成度AI处理器。可以实现图像、视频等多种数据分析与推理计算，可广泛用于智能监控、机器人、无人机、视频服务器等场景。本节我们来学习如何在昇腾AI处理器上使用MindSpore执行推理。

## 推理代码介绍

首先创建目录放置推理代码工程，例如`/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample`，可以从官网示例下载[样例代码](https://gitee.com/mindspore/docs/tree/r1.5/docs/sample_code/ascend910_resnet50_preprocess_sample)，`model`目录用于存放上述导出的[MindIR模型文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir)，`test_data`目录用于存放待分类的图片，待分类图片可以从[ImageNet2012](http://image-net.org/download-images)验证集中选取，推理代码工程目录结构如下：

```text
└─ascend910_resnet50_preprocess_sample
    ├── CMakeLists.txt                    // 构建脚本
    ├── README.md                         // 使用说明
    ├── main.cc                           // 主函数
    ├── model
    │   └── resnet50_imagenet.mindir      // MindIR模型文件
    └── test_data
        ├── ILSVRC2012_val_00002138.JPEG  // 输入样本图片1
        ├── ILSVRC2012_val_00003014.JPEG  // 输入样本图片2
        ├── ...                           // 输入样本图片n
```

其中`main.cc`中代码执行过程如下：

引用`mindspore`和`mindspore::dataset`的名字空间。

```c++
namespace ms = mindspore;
namespace ds = mindspore::dataset;
```

初始化环境，指定推理使用的硬件平台，设置DeviceID。

这里设置硬件为Ascend 910，DeviceID为0，示例代码如下：

```c++
auto context = std::make_shared<ms::Context>();
auto ascend910_info = std::make_shared<ms::Ascend910DeviceInfo>();
ascend910_info->SetDeviceID(0);
context->MutableDeviceInfo().push_back(ascend910_info);
```

加载模型文件：

```c++
// 加载MindIR模型
ms::Graph graph;
ms::Status ret = ms::Serialization::Load(resnet_file, ms::ModelType::kMindIR, &graph);
// 用图构建模型
ms::Model resnet50;
ret = resnet50.Build(ms::GraphCell(graph), context);
```

获取模型所需的输入信息：

```c++
std::vector<ms::MSTensor> model_inputs = resnet50.GetInputs();
```

加载图片文件：

```c++
// ReadFile是读取图像的函数
ms::MSTensor ReadFile(const std::string &file);
auto image = ReadFile(image_file);
```

图片预处理：

```c++
// 使用MindData提供的CPU算子进行图片预处理

// 创建算子，该算子将输入编码成RGB格式
std::shared_ptr<ds::TensorTransform> decode(new ds::vision::Decode());
// 创建算子，该算子把图片缩放到指定大小
std::shared_ptr<ds::TensorTransform> resize(new ds::vision::Resize({256}));
// 创建算子，该算子归一化输入
std::shared_ptr<ds::TensorTransform> normalize(new ds::vision::Normalize(
    {0.485 * 255, 0.456 * 255, 0.406 * 255}, {0.229 * 255, 0.224 * 255, 0.225 * 255}));
// 创建算子，该算子执行中心抠图
std::shared_ptr<ds::TensorTransform> center_crop(new ds::vision::CenterCrop({224, 224}));
// 创建算子，该算子将shape (H, W, C)变换成shape (C, H, W)
std::shared_ptr<ds::TensorTransform> hwc2chw(new ds::vision::HWC2CHW());

// 定义一个MindData数据预处理函数，按顺序包含上述算子
ds::Execute preprocessor({decode, resize, normalize, center_crop, hwc2chw});

// 调用数据预处理函数获取处理后的图像
ret = preprocessor(image, &image);
```

执行推理：

```c++
// 创建输出vector
std::vector<ms::MSTensor> outputs;
// 创建输入vector
std::vector<ms::MSTensor> inputs;
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                    image.Data().get(), image.DataSize());
// 调用Model的Predict函数进行推理
ret = resnet50.Predict(inputs, &outputs);
```

获取推理结果：

```c++
// 输出概率最大值
std::cout << "Image: " << image_file << " infer result: " << GetMax(outputs[0]) << std::endl;
```

## 构建脚本

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

>详细样例请参考：
><https://gitee.com/mindspore/docs/blob/r1.5/docs/sample_code/ascend910_resnet50_preprocess_sample/CMakeLists.txt>

## 编译推理代码

接下来编译推理的代码，首先要进入工程目录`ascend910_resnet50_preprocess_sample`，设置如下环境变量：

> 如果是Ascend 310设备，则进入工程目录`ascend310_resnet50_preprocess_sample`，以下代码均用Ascend 910为例。

```bash
# 控制log的打印级别. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 默认是WARNING级别.
export GLOG_v=2

# 选择Conda环境
LOCAL_ASCEND=/usr/local/Ascend # 运行包的根目录

# 运行包依赖的lib库
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64/common:${LOCAL_ASCEND}/driver/lib64/driver:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# MindSpore依赖的lib库
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# 配置必要的环境变量
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE算子的路径
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP路径
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE算子编译工具的路径
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # TBE依赖的Python库
```

执行`cmake`命令，其中`pip3`需要按照实际情况修改：

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
```

再执行`make`命令编译即可。

```bash
make
```

编译完成后，在`ascend910_resnet50_preprocess_sample`下会生成可执行文件。

## 执行推理并查看结果

以上操作完成之后，我们可以开始学习如何执行推理。

首先，登录Ascend 910环境，创建`model`目录放置MindIR文件`resnet50_imagenet.mindir`，例如`/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample/model`。

创建`test_data`目录放置图片，例如`/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample/test_data`。
就可以开始执行推理了：

```bash
./resnet50_sample
```

执行后，会对`test_data`目录下放置的所有图片进行推理，比如放置了2张[ImageNet2012](http://image-net.org/download-images)验证集中label为0的图片，可以看到推理结果如下。

```text
Image: ./test_data/ILSVRC2012_val_00002138.JPEG infer result: 0
Image: ./test_data/ILSVRC2012_val_00003014.JPEG infer result: 0
```
