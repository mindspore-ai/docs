# 加载和发布模型（Hub）

`Linux` `Ascend` `GPU` `模型发布` `模型加载` `初级` `中级` `高级`

<!-- TOC -->

- [加载和发布模型（Hub）](#加载和发布模型hub)
    - [概述](#概述)
    - [模型加载](#模型加载)
    - [模型发布](#模型发布)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_zh_cn/use/load_and_publish_model.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

MindSpore Hub是MindSpore生态的预训练模型应用工具，作为模型开发者和应用开发者的管道，它不仅向模型开发者提供了方便快捷的模型发布方式，而且向应用开发者提供了简单易用的模型加载和微调API。

本教程以GoogleNet为例，对想要将模型发布到MindSpore Hub的模型开发者介绍了模型上传的详细步骤，也对想要使用MindSpore Hub模型进行推理的应用开发者提供了具体操作流程。

## 模型加载 

`mindspore_hub.load` API用于加载预训练模型，可以实现一行代码完成模型的加载。主要的模型加载流程如下：

1. 在[MindSpore Hub官网](https://www.mindspore.cn/resources/hub)上搜索感兴趣的模型。

    例如，想使用GoogleNet对CIFAR-10数据集进行分类，可以在MindSpore Hub官网上使用关键词`GoogleNet`进行搜索。页面将会返回与GoogleNet相关的所有模型。进入相关模型页面之后，获得详情页`url`。

2. 使用`url`完成模型的加载，示例代码如下：

   ```python
   import mindspore_hub as mshub
   import mindspore
   from mindspore import context, Tensor, nn
   from mindspore.train.model import Model
   from mindspore.common import dtype as mstype
   import mindspore.dataset.vision.py_transforms as py_transforms

   context.set_context(mode=context.GRAPH_MODE,
                        device_target="Ascend",
                        device_id=0)

   model = "mindspore/ascend/0.7/googlenet_v1_cifar10"

   # Initialize the number of classes based on the pre-trained model.
   network = mshub.load(model, num_classes=10)
   network.set_train(False)

   # ...

   ```

3. 完成模型加载后，可以使用MindSpore进行推理，参考[这里](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.0/multi_platform_inference.html)。

## 模型发布

用户可通过向[hub](https://gitee.com/mindspore/hub)仓提交PR的方式向MindSpore Hub发布模型。这里我们以GoogleNet为例，列出模型提交到MindSpore Hub的步骤。

1. 将你的预训练模型托管在可以访问的存储位置。

2. 参照[模板](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/googlenet/mindspore_hub_conf.py)，在你自己的代码仓中添加模型生成文件`mindspore_hub_conf.py`，文件放置的位置如下： 

   ```shell
   googlenet
   ├── src
   │   ├── googlenet.py
   ├── script
   │   ├── run_train.sh
   ├── train.py
   ├── test.py
   ├── mindspore_hub_conf.py
   ```

3. 参照[模板](https://gitee.com/mindspore/hub/blob/r1.0/mshub_res/assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md)，在`hub/mshub_res/assets/mindspore/ascend/0.7`文件夹下创建`{model_name}_{model_version}_{dataset}.md`文件，其中`ascend`为模型运行的硬件平台，`0.7`为MindSpore的版本号，`hub/mshub_res`的目录结构为：

   ```shell
   hub
   ├── mshub_res
   │   ├── assets
   │       ├── mindspore
   |           ├── gpu
   |               ├── 0.7
   |           ├── ascend
   |               ├── 0.7 
   |                   ├── googlenet_v1_cifar10.md
   │   ├── tools
   |       ├── md_validator.py
   |       └── md_validator.py 
   ```
   注意，`{model_name}_{model_version}_{dataset}.md`文件中需要补充如下所示的`file-format`、`asset-link` 和 `asset-sha256`信息，它们分别表示模型文件格式、模型存储位置（步骤1所得）和模型哈希值。

   ```shell
   file-format: ckpt  
   asset-link: https://download.mindspore.cn/model_zoo/official/cv/googlenet/goolenet_ascend_0.2.0_cifar10_official_classification_20200713/googlenet.ckpt  
   asset-sha256: 114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7
   ```   

   其中，MindSpore Hub支持的模型文件格式有：
   - [MindSpore CKPT](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/use/save_and_load_model.html#id3)
   - [AIR](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/use/save_and_load_model.html#id7)
   - [MindIR](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/use/save_and_load_model.html#id9)
   - [ONNX](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/use/save_and_load_model.html#id8)
   - [MSLite](https://www.mindspore.cn/tutorial/lite/zh-CN/r1.0/use/converter_tool.html)

   对于每个预训练模型，执行以下命令，用来获得`.md`文件`asset-sha256`处所需的哈希值，其中`googlenet.ckpt`是从步骤1的存储位置处下载并保存到`tools`文件夹的预训练模型，运行后输出的哈希值为`114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7`。

   ```python
   cd /hub/mshub_res/tools
   python get_sha256.py ../googlenet.ckpt
   ```

4. 使用`hub/mshub_res/tools/md_validator.py`在本地核对`.md`文件的格式，执行以下命令，输出结果为`All Passed`，表示`.md`文件的格式和内容均符合要求。

   ```python
   python md_validator.py ../assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md
   ```

5. 在`mindspore/hub`仓创建PR，详细创建方式可以参考[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md)。

一旦你的PR合入到`mindspore/hub`的master分支，你的模型将于24小时内在[MindSpore Hub 网站](https://www.mindspore.cn/resources/hub)上显示。有关模型上传的更多详细信息，请参考[README](https://gitee.com/mindspore/hub/blob/master/mshub_res/README.md)。
