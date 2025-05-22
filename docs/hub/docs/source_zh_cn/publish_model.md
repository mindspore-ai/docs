# 发布模型

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/hub/docs/source_zh_cn/publish_model.md)

## 概述

[MindSpore Hub](https://www.mindspore.cn/hub)是存放MindSpore官方或者第三方开发者提供的预训练模型的平台。它向应用开发者提供了简单易用的模型加载和微调APIs，使得用户可以基于预训练模型进行推理或者微调，并部署到自己的应用中。用户也可以将自己训练好的模型按照指定的步骤发布到MindSpore Hub中，以供其他用户进行下载和使用。

本教程以GoogleNet为例，对想要将模型发布到MindSpore Hub的模型开发者介绍了模型上传步骤。

## 发布模型到MindSpore Hub

用户可通过向[hub](https://gitee.com/mindspore/hub)仓提交PR的方式向MindSpore Hub发布模型。这里我们以GoogleNet为例，列出模型提交到MindSpore Hub的步骤。

1. 将你的预训练模型托管在可以访问的存储位置。

2. 参照[模板](https://gitee.com/mindspore/models/blob/master/research/cv/SE_ResNeXt50/mindspore_hub_conf.py)，在你自己的代码仓中添加模型生成文件`mindspore_hub_conf.py`，文件放置的位置如下：

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

3. 参照[模板](https://gitee.com/mindspore/hub/blob/master/mshub_res/assets/mindspore/1.6/googlenet_cifar10.md#)，在`hub/mshub_res/assets/mindspore/1.6`文件夹下创建`{model_name}_{dataset}.md`文件，其中`1.6`为MindSpore的版本号，`hub/mshub_res`的目录结构为：

   ```text
   hub
   ├── mshub_res
   │   ├── assets
   │       ├── mindspore
   │           ├── 1.6
   │               ├── googlenet_cifar10.md
   │   ├── tools
   │       ├── get_sha256.py
   │       ├── load_markdown.py
   │       └── md_validator.py
   ```

   注意，`{model_name}_{dataset}.md`文件中需要补充如下所示的`file-format`、`asset-link` 和 `asset-sha256`信息，它们分别表示模型文件格式、模型存储位置（步骤1所得）和模型哈希值。

   ```text
   file-format: ckpt
   asset-link: https://download.mindspore.cn/models/r1.6/googlenet_ascend_v160_cifar10_official_cv_acc92.53.ckpt
   asset-sha256: b2f7fe14782a3ab88ad3534ed5f419b4bbc3b477706258bd6ed8f90f529775e7
   ```

   其中，MindSpore Hub支持的模型文件格式有：
   - [MindSpore CKPT](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html#保存与加载)
   - [MINDIR](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html#保存和加载mindir)
   - [AIR](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.export.html#mindspore.export)
   - [ONNX](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.export.html#mindspore.export)

   对于每个预训练模型，执行以下命令，用来获得`.md`文件`asset-sha256`处所需的哈希值，其中`googlenet.ckpt`是从步骤1的存储位置处下载并保存到`tools`文件夹的预训练模型，运行后输出的哈希值为`b2f7fe14782a3ab88ad3534ed5f419b4bbc3b477706258bd6ed8f90f529775e7`。

   ```bash
   cd /hub/mshub_res/tools
   python get_sha256.py --file ../googlenet.ckpt
   ```

4. 使用`hub/mshub_res/tools/md_validator.py`在本地核对`.md`文件的格式，执行以下命令，输出结果为`All Passed`，表示`.md`文件的格式和内容均符合要求。

   ```bash
   python md_validator.py --check_path ../assets/mindspore/1.6/googlenet_cifar10.md
   ```

5. 在`mindspore/hub`仓创建PR，详细创建方式可以参考[贡献者Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md#)。

一旦你的PR合入到`mindspore/hub`的master分支，你的模型将于24小时内在[MindSpore Hub 网站](https://www.mindspore.cn/hub)上显示。有关模型上传的更多详细信息，请参考[README](https://gitee.com/mindspore/hub/blob/master/mshub_res/README.md#)。
