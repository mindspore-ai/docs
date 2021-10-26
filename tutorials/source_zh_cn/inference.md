# 推理

`Ascend` `端侧` `入门` `推理应用`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本节是初级教程的最后一节，为了更好地适配不同推理设备，因此推理分为 1）昇腾AI处理器推理和 2）移动设备推理。

## 昇腾AI处理器推理

昇腾（Ascend）AI处理器是面向边缘场景的高能效高集成度AI处理器。可以实现图像、视频等多种数据分析与推理计算，可广泛用于智能监控、机器人、无人机、视频服务器等场景。本节我们来学习如何在昇腾AI处理器上使用MindSpore执行推理。

### 推理代码介绍

首先创建目录放置推理代码工程，例如`/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample`，可以从官网示例下载[样例代码](https://gitee.com/mindspore/docs/tree/master/docs/sample_code/ascend910_resnet50_preprocess_sample)，`model`目录用于存放上述导出的`MindIR`模型文件，`test_data`目录用于存放待分类的图片，推理代码工程目录结构如下：

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

### 构建脚本

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
><https://gitee.com/mindspore/docs/blob/master/docs/sample_code/ascend910_resnet50_preprocess_sample/CMakeLists.txt>

### 编译推理代码

接下来编译推理的代码，首先要进入工程目录`ascend910_resnet50_preprocess_sample`，设置如下环境变量：

> 如果是Ascend 310设备，则进入工程目录`ascend310_resnet50_preprocess_sample`，以下代码均用Ascend 910为例。

```bash
# 控制log的打印级别. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, 默认是WARNING级别.
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

编译完成后，在`ascend910_resnet50_preprocess_sample`下会生成可执行`main`文件。

### 执行推理并查看结果

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

## 移动设备推理

MindSpore Lite是端边云全场景AI框架MindSpore的端侧部分，可以在手机等移动设备上实现智能应用。MindSpore Lite提供高性能推理引擎和超轻量的解决方案，支持iOS、Android等手机操作系统以及LiteOS嵌入式操作系统，支持手机、大屏、平板、IoT等各种智能设备，支持MindSpore/TensorFlow Lite/Caffe/Onnx模型的应用。

在本环节中，提供了运行在Windows和Linux操作系统下，基于C++ API编写的Demo，帮助用户熟悉端侧推理流程。Demo通过随机生成的数据作为输入数据，执行MobileNetV2模型的推理，直接在电脑中打印获得输出数据。

> 运行在手机中的完整实例可以参考官网示例：[基于JNI接口的Android应用开发]( https://www.mindspore.cn/lite/docs/zh-CN/master/quick_start/quick_start.html)。

### 模型转换

模型在用于端侧推理之前需要先进行格式的转换。当前，MindSpore Lite支持MindSpore、TensorFlow Lite、Caffe和ONNX 4类AI框架。

下面以MindSpore训练得到的[mobilenetv2.mindir](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.mindir)模型为例，说明Demo中所使用的`mobilenetv2.ms`模型是如何生成的。

> 本小节展开说明了转换的操作过程，仅实现Demo运行可跳过本小节。
>
> 本小节仅针对Demo所用模型，详细的转换工具使用说明请参考官网[推理模型转换](https://www.mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html#)章节。

- 转换工具下载

  根据所使用的操作系统，下载转换工具的[压缩包](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)并解压至本地目录，获得`converter`工具，并配置环境变量。

- 转换工具使用

    - Linux使用说明

        进入`converter_lite`可执行文件所在的目录，将下载的`mobilenetv2.mindir`模型放入同一路径下，在电脑终端中输入命令完成转换：

        ```cpp
        ./converter_lite --fmk=MINDIR --modelFile=mobilenetv2.mindir --outputFile=mobilenetv2
        ```

    - Windows使用说明

        进入`converter_lite`可执行文件所在的目录，将下载的`mobilenetv2.mindir`模型放入同一路径下，在电脑终端中输入命令完成转换：

        ```cpp
        call converter_lite --fmk=MINDIR --modelFile=mobilenetv2.mindir --outputFile=mobilenetv2
        ```

    - 参数说明

        在执行命令的过程中设置了三个参数，`--fmk`代表输入模型的原始格式，这里设置为`MINDIR`，即MindSpore框架训练模型的导出格式；`--modelFile`指输入模型的路径；`--outputFile`设定了模型的输出路径，这里自动将转换后的模型添加了`.ms`后缀。

### 构建环境与运行

#### Linux系统构建与运行

- 编译构建

  在`mindspore/lite/examples/quick_start_cpp`目录下执行build脚本，将能够自动下载相关文件并编译Demo。

  ```bash
  bash build.sh
  ```

- 执行推理

  编译构建后，进入`mindspore/lite/examples/quick_start_cpp/build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  ./mindspore_quick_start_cpp ../model/mobilenetv2.ms
  ```

  执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```text
  tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
  output data is:5.26823e-05 0.00049752 0.000296722 0.000377607 0.000177048 .......
  ```

#### Windows系统构建与运行

- 编译构建

    - 库下载：请手动下载硬件平台为CPU、操作系统为Windows-x64的MindSpore Lite模型推理框架[mindspore-lite-{version}-win-x64.zip](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)，将解压后`inference/lib`目录下的`libmindspore-lite.a`拷贝到`mindspore/lite/examples/quick_start_cpp/lib`目录、`inference/include`目录拷贝到`mindspore/lite/examples/quick_start_cpp/include`目录。

    - 模型下载：请手动下载相关模型文件[mobilenetv2.ms](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_imagenet/mobilenetv2.ms)，并将其拷贝到`mindspore/lite/examples/quick_start_cpp/model`目录。

        > 可选择使用模型转换小节所获得的mobilenetv2.ms模型文件。

    - 编译：在`mindspore/lite/examples/quick_start_cpp`目录下执行build脚本，将能够自动下载相关文件并编译Demo。

        ```bash
        call build.bat
        ```

- 执行推理

  编译构建后，进入`mindspore/lite/examples/quick_start_cpp/build`目录，并执行以下命令，体验MindSpore Lite推理MobileNetV2模型。

  ```bash
  call ./mindspore_quick_start_cpp.exe ../model/mobilenetv2.ms
  ```

  执行完成后将能得到如下结果，打印输出Tensor的名称、输出Tensor的大小，输出Tensor的数量以及前50个数据：

  ```text
  tensor name is:Default/head-MobileNetV2Head/Softmax-op204 tensor size is:4000 tensor elements num is:1000
  output data is:5.26823e-05 0.00049752 0.000296722 0.000377607 0.000177048 .......
  ```

### 推理代码解析

下面分析Demo源代码中的推理流程，显示C++ API的具体使用方法。

#### 模型加载

首先从文件系统中读取MindSpore Lite模型，并通过`mindspore::lite::Model::Import`函数导入模型进行解析。

```c++
// 读模型文件
size_t size = 0;
char *model_buf = ReadFile(model_path, &size);
if (model_buf == nullptr) {
  std::cerr << "Read model file failed." << std::endl;
  return RET_ERROR;
}
// 加载模型
auto model = mindspore::lite::Model::Import(model_buf, size);
delete[](model_buf);
if (model == nullptr) {
  std::cerr << "Import model file failed." << std::endl;
  return RET_ERROR;
}
```

#### 模型编译

模型编译主要包括创建配置上下文、创建会话、图编译等步骤。

```c++
mindspore::session::LiteSession *Compile(mindspore::lite::Model *model) {
  // 初始化上下文
  auto context = std::make_shared<mindspore::lite::Context>();
  if (context == nullptr) {
    std::cerr << "New context failed while." << std::endl;
    return nullptr;
  }

  // 创建session
  mindspore::session::LiteSession *session = mindspore::session::LiteSession::CreateSession(context.get());
  if (session == nullptr) {
    std::cerr << "CreateSession failed while running." << std::endl;
    return nullptr;
  }

  // 图编译
  auto ret = session->CompileGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    delete session;
    std::cerr << "Compile failed while running." << std::endl;
    return nullptr;
  }

  // 注意:如果使用 model->Free(),模型将不能再次被编译
  if (model != nullptr) {
    model->Free();
  }
  return session;
}
```

#### 模型推理

模型推理主要包括输入数据、执行推理、获得输出等步骤，其中本示例中的输入数据是通过随机数据构造生成，最后将执行推理后的输出结果打印出来。

```c++
int Run(mindspore::session::LiteSession *session) {
  // 获取输入数据
  auto inputs = session->GetInputs();
  auto ret = GenerateInputDataWithRandom(inputs);
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Generate Random Input Data failed." << std::endl;
    return ret;
  }

  // 运行
  ret = session->RunGraph();
  if (ret != mindspore::lite::RET_OK) {
    std::cerr << "Inference error " << ret << std::endl;
    return ret;
  }

  // 获取输出数据
  auto out_tensors = session->GetOutputs();
  for (auto tensor : out_tensors) {
    std::cout << "tensor name is:" << tensor.first << " tensor size is:" << tensor.second->Size()
              << " tensor elements num is:" << tensor.second->ElementsNum() << std::endl;
    auto out_data = reinterpret_cast<float *>(tensor.second->MutableData());
    std::cout << "output data is:";
    for (int i = 0; i < tensor.second->ElementsNum() && i <= 50; i++) {
      std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
  }
  return mindspore::lite::RET_OK;
}
```

#### 内存释放

无需使用MindSpore Lite推理框架时，需要释放已经创建的`LiteSession`和`Model`。

```c++
// 删除模型缓存
delete model;
// 删除session缓存
delete session;
```
