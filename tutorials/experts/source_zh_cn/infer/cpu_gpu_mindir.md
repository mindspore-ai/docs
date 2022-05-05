# GPU/CPU推理

<a href="https://gitee.com/mindspore/docs/blob/r1.7/tutorials/experts/source_zh_cn/infer/cpu_gpu_mindir.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## 使用C++接口推理MindIR格式文件

用户可以创建C++应用程序，调用MindSpore的C++接口推理MindIR模型。

### 推理目录结构介绍

首先创建目录放置推理代码工程，例如`/home/mindspore_sample/gpu_resnet50_inference_sample`，可以从官网示例下载[样例代码](https://gitee.com/mindspore/docs/tree/r1.7/docs/sample_code/gpu_resnet50_inference_sample)，`model`目录用于存放`MindIR`[模型文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/sample_resources/ascend310_resnet50_preprocess_sample/resnet50_imagenet.mindir)，推理代码工程目录结构如下：

```text
└─gpu_resnet50_inference_sample
    ├── build.sh                          // 构建脚本
    ├── CMakeLists.txt                    // CMake构建脚本
    ├── README.md                         // 使用说明
    ├── src
    │   └── main.cc                       // 主函数
    │── model
        └── resnet50_imagenet.mindir      // 模型文件
```

### 推理代码介绍

推理代码样例：<https://gitee.com/mindspore/docs/blob/r1.7/docs/sample_code/gpu_resnet50_inference_sample/src/main.cc> 。

引用`mindspore`名字空间：

```c++
using mindspore::Context;
using mindspore::Serialization;
using mindspore::Model;
using mindspore::Status;
using mindspore::ModelType;
using mindspore::GraphCell;
using mindspore::kSuccess;
using mindspore::MSTensor;
```

初始化环境，指定推理使用的硬件平台，设置DeviceID和精度。

这里设置硬件为 GPU，DeviceID为0，推理精度为FP16，示例代码如下：

```c++
auto gpu_device_info = std::make_shared<mindspore::GPUDeviceInfo>();
gpu_device_info->SetDeviceID(device_id);
gpu_device_info->SetPrecisionMode("fp16");
context->MutableDeviceInfo().push_back(gpu_device_info);
```

加载模型文件：

```c++
// 加载MindIR模型
mindspore::Graph graph;
Serialization::Load(mindir_path, ModelType::kMindIR, &graph);
// 用图构建模型
ms::Model model;
model.Build(ms::GraphCell(graph), context);
```

获取模型所需的输入信息：

```c++
std::vector<ms::MSTensor> model_inputs = model->GetInputs();
```

构造网络输入：

```c++
std::vector<MSTensor> inputs;
float *dummy_data = new float[1*3*224*224];
inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                      dummy_data, 1*3*224*224*sizeof(float));
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
ret = model.Predict(inputs, &outputs);
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
```

使用指定的源文件生成目标可执行文件，并为目标文件链接MindSpore库：

```cmake
add_executable(main src/main.cc)
target_link_libraries(main ${MS_LIB})
```

>详细样例请参考：
><https://gitee.com/mindspore/docs/blob/r1.7/docs/sample_code/gpu_resnet50_inference_sample/CMakeLists.txt>

### 编译推理代码

接下来编译推理的代码，首先要进入工程目录`gpu_resnet50_inference_sample`：

可以根据实际情况对build.sh中的`pip3`修改，修改完成后`bash build.sh`命令编译即可。

```bash
bash build.sh
```

编译完成后，在`gpu_resnet50_inference_sample/out`下会生成可执行`main`文件。

### 执行推理并查看结果

以上操作完成之后，我们可以开始学习如何执行推理。

首先，登录GPU环境，创建`model`目录放置MindIR文件`resnet50_imagenet.mindir`，例如`/home/mindspore_sample/gpu_resnet50_inference_sample/model`。

在执行推理之前，首先需要设置环境变量，环境变量需要根据实际情况修改。其中TensorRT库为可选配置项，推荐将TensorRT库路径添加到`LD_LIBRARY_PATH`环境变量中，有助提升模型推理性能。

```bash
export LD_PRELOAD=/home/miniconda3/lib/libpython3.7m.so
export LD_LIBRARY_PATH=/usr/local/TensorRT-7.2.2.3/lib/:$LD_LIBRARY_PATH
```

就可以开始执行推理了：

```bash
cd out/
./main ../model/resnet50_imagenet.mindir 1000 10
```

在当前测试脚本中，我们打印了每一个step的推理时延和平均时延：

```text
Start to load model..
Load model successuflly
Start to warmup..
Warmup finished
Start to infer..
step 0 cost 1.54004ms
step 1 cost 1.5271ms
... ...
step 998 cost 1.30688ms
step 999 cost 1.30493ms
infer finished.
=================Average inference time: 1.35195 ms
```

### 备注

- 一些网络在训练过程时，人为将部分算子精度设置为FP16。例如ModelZoo中的[Bert网络](https://gitee.com/mindspore/models/blob/master/official/nlp/bert/src/bert_model.py)，将Dense和LayerNorm设置为FP16进行训练。

```python
class BertOutput(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 compute_type=mstype.float32):
        super(BertOutput, self).__init__()
        # Set the nn.Dense to fp16.
        self.dense = nn.Dense(in_channels, out_channels,
                              weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.dropout_prob = dropout_prob
        self.add = P.Add()
        # Set the nn.LayerNorm to fp16.
        self.layernorm = nn.LayerNorm((out_channels,)).to_float(compute_type)
        self.cast = P.Cast()
        ... ...
```

建议部署推理任务时，将其修改为FP32后再单精度MindIR模型；如果希望进一步提升推理性能，可以通过`mindspore::GPUDeviceInfo::SetPrecisionMode("fp16")`将推理精度设置为FP16，框架会自动选择性能较优的算子推理。

- 部分推理脚本中可能引入了一些训练过程中特有的网络结构，例如模型要求传入图片或者语料的Label，并直接将Label传递到网络输出，建议删除这部分算子之后，再导出MindIR模型，以提高推理性能。

## 使用ONNX格式文件推理

1. 在训练平台上生成ONNX格式模型，具体步骤请参考[导出ONNX格式文件](https://www.mindspore.cn/tutorials/zh-CN/r1.7/advanced/train/save.html#导出onnx格式文件)。

2. 在GPU上进行推理，具体可以参考推理使用runtime/SDK的文档。如在Nvidia GPU上进行推理，使用常用的TensorRT，可参考[TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt)。
