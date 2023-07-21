# Ascend 910 AI处理器上推理

`Linux` `Ascend` `推理应用` `初级` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/inference/source_zh_cn/multi_platform_inference_ascend_910.md)

## 使用checkpoint格式文件单卡推理

1. 使用`model.eval`接口来进行模型验证。

   1.1 模型已保存在本地

   首先构建模型，然后使用`mindspore.train.serialization`模块的`load_checkpoint`和`load_param_into_net`从本地加载模型与参数，传入验证数据集后即可进行模型推理，验证数据集的处理方式与训练数据集相同。

    ```python
    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    dataset = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size,
                             1)
    acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    ```

    其中，  
    `model.eval`为模型验证接口，对应接口说明：<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.Model.eval>。
    > 推理样例代码：<https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/lenet/eval.py>。

   1.2 使用MindSpore Hub从华为云加载模型

   首先构建模型，然后使用`mindspore_hub.load`从云端加载模型参数，传入验证数据集后即可进行推理，验证数据集的处理方式与训练数据集相同。

    ```python
    model_uid = "mindspore/ascend/0.7/googlenet_v1_cifar10"  # using GoogleNet as an example.
    network = mindspore_hub.load(model_uid, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    dataset = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size,
                             1)
    acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    ```

    其中，  
    `mindspore_hub.load`为加载模型参数接口，对应接口说明：<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore_hub/mindspore_hub.html#module-mindspore_hub>。

2. 使用`model.predict`接口来进行推理操作。

   ```python
   model.predict(input_data)
   ```

   其中，  
   `model.predict`为推理接口，对应接口说明：<https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.Model.predict>。

## 分布式推理

分布式推理是指推理阶段采用多卡进行推理。如果训练时采用数据并行或者模型参数是合并保存，那么推理方式与上述一致，只需要注意每卡加载同样的checkpoint文件进行推理。

本篇教程主要介绍在多卡训练过程中，每张卡上保存模型的切片，在推理阶段采用多卡形式，按照推理策略重新加载模型进行推理的过程。针对超大规模神经网络模型的参数个数过多，模型无法完全加载至单卡中进行推理的问题，可利用多卡进行分布式推理。

> 分布式推理样例代码：
>
> <https://gitee.com/mindspore/docs/tree/r1.1/tutorials/tutorial_code/distributed_inference>

分布式推理流程如下：

1. 执行训练，生成checkpoint文件和模型参数切分策略文件。

    > - 分布式训练教程和样例代码可参考链接：<https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/distributed_training_ascend.html>.
    > - 在分布式推理场景中，训练阶段的`CheckpointConfig`接口的`integrated_save`参数需设定为`False`，表示每卡仅保存模型切片而不是全量模型。
    > - `set_auto_parallel_context`接口的`parallel_mode`参数需设定为`auto_parallel`或者`semi_auto_parallel`，并行模式为自动并行或者半自动并行。
    > - 此外还需指定`strategy_ckpt_save_file`参数，即生成的策略文件的地址。

2. 设置context，根据推理数据推导出推理策略。

    ```python
    context.set_auto_parallel_context(full_batch=True, parallel_mode='semi_auto_parallel', strategy_ckpt_load_file='./train_strategy.ckpt')
    network = Net()
    model = Model(network)
    predict_data = create_predict_data()
    predict_strategy = model.infer_predict_layout(predict_data)
    ```

    其中，

    - `full_batch`：是否全量导入数据集，为`True`时表明全量导入，每卡的数据相同，该场景中必须设置为`True`。
    - `parallel_mode`：并行模式，该场景中必须设置为自动并行或者半自动并行模式。
    - `strategy_ckpt_load_file`：训练阶段生成的策略文件的文件地址，分布式推理场景中该参数必须设置。
    - `create_predict_data`：用户需自定义的接口，返回推理数据。与训练阶段不同的是，分布式推理场景中返回类型必须为`Tensor`。
    - `infer_predict_layout`：根据推理数据生成推理策略。

3. 导入checkpoint文件，根据推理策略加载相应的模型切片至每张卡中。

    ```python
    ckpt_file_list = create_ckpt_file_list()
    load_distributed_checkpoint(network, ckpt_file_list, predict_strategy)
    ```

    其中，

    - `create_ckpt_file_list`：用户需自定义的接口，返回按rank id排序的CheckPoint文件名列表。
    - `load_distributed_checkpoint`：对模型切片进行合并，再根据推理策略进行切分，加载至网络中。

    > `load_distributed_checkpoint`接口支持predict_strategy为`None`，此时为单卡推理，其过程与分布式推理有所不同，详细用法请参考链接：
    > <https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.load_distributed_checkpoint>.

4. 进行推理，得到推理结果。

    ```python
    model.predict(predict_data)
    ```

## 使用C++接口推理MindIR格式文件

用户可以创建C++应用程序，调用MindSpore的C++接口推理MindIR模型。

### 推理目录结构介绍

创建目录放置推理代码工程，例如`/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample`，可以从官网示例下载[样例代码](https://gitee.com/mindspore/docs/tree/r1.1/tutorials/tutorial_code/ascend910_resnet50_preprocess_sample)，`model`目录用于存放上述导出的`MindIR`模型文件，`test_data`目录用于存放待分类的图片，推理代码工程目录结构如下:

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

### 推理代码介绍

推理代码样例：<https://gitee.com/mindspore/docs/blob/r1.1/tutorials/tutorial_code/ascend910_resnet50_preprocess_sample/main.cc> 。

环境初始化，指定硬件为Ascend 910，DeviceID为0：

```c++
ms::GlobalContext::SetGlobalDeviceTarget(ms::kDeviceTypeAscend910);
ms::GlobalContext::SetGlobalDeviceID(0);
```

加载模型文件:

```c++
// Load MindIR model
auto graph =ms::Serialization::LoadModel(resnet_file, ms::ModelType::kMindIR);
// Build model with graph object
ms::Model resnet50((ms::GraphCell(graph)));
ms::Status ret = resnet50.Build({});
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

图片预处理:

```c++
// Create the CPU operator provided by MindData to get the function object
ms::dataset::Execute preprocessor({ms::dataset::vision::Decode(),  // Decode the input to RGB format
                                   ms::dataset::vision::Resize({256}),  // Resize the image to the given size
                                   ms::dataset::vision::Normalize({0.485 * 255, 0.456 * 255, 0.406 * 255},
                                                                  {0.229 * 255, 0.224 * 255, 0.225 * 255}),  // Normalize the input
                                   ms::dataset::vision::CenterCrop({224, 224}),  // Crop the input image at the center
                                   ms::dataset::vision::HWC2CHW(),  // shape (H, W, C) to shape(C, H, W)
                                  });
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

### 构建脚本介绍

构建脚本用于构建用户程序，样例来自于：<https://gitee.com/mindspore/docs/blob/r1.1/tutorials/tutorial_code/ascend910_resnet50_preprocess_sample/CMakeLists.txt> 。

由于MindSpore使用[旧版的C++ ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html)，因此用户程序需与MindSpore一致，否则编译链接会失败。

```cmake
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_STANDARD 17)
```

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

## 编译推理代码

进入工程目录`ascend910_resnet50_preprocess_sample`，设置如下环境变量：

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64/common:${LOCAL_ASCEND}/driver/lib64/driver:${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
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

## 执行推理并查看结果

登录Ascend 910环境，创建`model`目录放置MindIR文件`resnet50_imagenet.mindir`，例如`/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample/model`。
创建`test_data`目录放置图片，例如`/home/HwHiAiUser/mindspore_sample/ascend910_resnet50_preprocess_sample/test_data`。
就可以开始执行推理了:

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
