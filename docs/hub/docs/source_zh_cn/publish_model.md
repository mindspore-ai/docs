# 发布模型

<a href="https://gitee.com/mindspore/docs/blob/master/docs/hub/docs/source_zh_cn/publish_model.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 概述

[MindSpore Hub](https://www.mindspore.cn/resources/hub/)是存放MindSpore官方或者第三方开发者提供的预训练模型的平台。它向应用开发者提供了简单易用的模型加载和微调APIs，使得用户可以基于预训练模型进行推理或者微调，并部署到自己的应用中。用户也可以将自己训练好的模型按照指定的步骤发布到MindSpore Hub中，以供其他用户进行下载和使用。

本教程以GoogleNet为例，对想要将模型发布到MindSpore Hub的模型开发者介绍了模型上传步骤。

## 发布模型到MindSpore Hub

用户可通过向[hub](https://gitee.com/mindspore/hub)仓提交PR的方式向MindSpore Hub发布模型。这里我们以GoogleNet为例，列出模型提交到MindSpore Hub的步骤。

1. 将你的预训练模型托管在可以访问的存储位置。

2. 参照[模板](https://gitee.com/mindspore/models/blob/master/official/cv/googlenet/mindspore_hub_conf.py)，在你自己的代码仓中添加模型生成文件`mindspore_hub_conf.py`，文件放置的位置如下：

   ```text
   googlenet
   ├── src
   │   ├── googlenet.py
   ├── script
   │   ├── run_train.sh
   ├── train.py
   ├── test.py
   ├── mindspore_hub_conf.py
   ```

3. 参照[模板](https://gitee.com/mindspore/hub/blob/master/mshub_res/assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md#)，在`hub/mshub_res/assets/mindspore/ascend/0.7`文件夹下创建`{model_name}_{model_version}_{dataset}.md`文件，其中`ascend`为模型运行的硬件平台，`0.7`为MindSpore的版本号，`hub/mshub_res`的目录结构为：

   ```text
   hub
   ├── mshub_res
   │   ├── assets
   │       ├── mindspore
   │           ├── gpu
   │               ├── 0.7
   │           ├── ascend
   │               ├── 0.7
   │                   ├── googlenet_v1_cifar10.md
   │   ├── tools
   │       ├── get_sha256.py
   │       ├── load_markdown.py
   │       └── md_validator.py
   ```

   注意，`{model_name}_{model_version}_{dataset}.md`文件中需要补充如下所示的`file-format`、`asset-link` 和 `asset-sha256`信息，它们分别表示模型文件格式、模型存储位置（步骤1所得）和模型哈希值。

   ```text
   file-format: ckpt
   asset-link: https://download.mindspore.cn/model_zoo/official/cv/googlenet/goolenet_ascend_0.2.0_cifar10_official_classification_20200713/googlenet.ckpt
   asset-sha256: 114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7
   ```

   其中，MindSpore Hub支持的模型文件格式有：
   - [MindSpore CKPT](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html#checkpoint)
   - [MINDIR](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html#mindir)
   - [AIR](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html#air)
   - [ONNX](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/save_model.html#onnx)

   对于每个预训练模型，执行以下命令，用来获得`.md`文件`asset-sha256`处所需的哈希值，其中`googlenet.ckpt`是从步骤1的存储位置处下载并保存到`tools`文件夹的预训练模型，运行后输出的哈希值为`114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7`。

   ```bash
   cd /hub/mshub_res/tools
   python get_sha256.py --file ../googlenet.ckpt
   ```

4. 使用`hub/mshub_res/tools/md_validator.py`在本地核对`.md`文件的格式，执行以下命令，输出结果为`All Passed`，表示`.md`文件的格式和内容均符合要求。

   ```bash
   python md_validator.py --check_path ../assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md
   ```

5. 在`mindspore/hub`仓创建PR，详细创建方式可以参考[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md#)。

一旦你的PR合入到`mindspore/hub`的master分支，你的模型将于24小时内在[MindSpore Hub 网站](https://www.mindspore.cn/resources/hub)上显示。有关模型上传的更多详细信息，请参考[README](https://gitee.com/mindspore/hub/blob/master/mshub_res/README.md#)。
