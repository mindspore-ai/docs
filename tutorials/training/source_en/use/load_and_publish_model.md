# Loading and Publishing Models using MindSpore Hub

`Linux` `Ascend` `GPU` `Model Publishing` `Model Loading` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Loading and Publishing Models using MindSpore Hub](#loading-and-publishing-models-using-mindspore-hub)
    - [Overview](#overview)
    - [How to load models](#how-to-load-models)
    - [How to publish models](#how-to-publish-models)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/advanced_use/load_and_publish_model.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

MindSpore Hub is a pre-trained model application tool of the MindSpore ecosystem, which serves as a channel for model developers and application developers. It not only provides model developers with a convenient and fast channel for model submission, but also provides application developers with simple model loading and fine-tuning APIs.

For model developers who are interested in publishing models into MindSpore Hub, this tutorial introduces the specific steps to submit models using GoogleNet as an example. It also describes how to load MindSpore Hub models for application developers who aim to do inference learning on new dataset.

## How to load models

`mindspore_hub.load` API is used to load the pre-trained model in a single line of code. The main process of model loading is as follows:

1. Search the model of interest on [MindSpore Hub Website](https://www.mindspore.cn/resources/hub).

   For example, if you aim to perform image classification on CIFAR-10 dataset using GoogleNet, please search on [MindSpore Hub Website](https://www.mindspore.cn/resources/hub) with the keyword `GoogleNet`. Then all related models will be returned.  Once you enter into the related model page, you can get the website `url`.

2. Complete the task of loading model using `url` , as shown in the example below:

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

3. After loading the model, you can use MindSpore to do inference. You can refer to [here](https://www.mindspore.cn/tutorial/inference/en/r1.0/multi_platform_inference.html).

## How to publish models

You can publish models to MindSpore Hub via PR in [hub](https://gitee.com/mindspore/hub) repo. Here we use GoogleNet as an example to list the steps of model submission to MindSpore Hub. 

1. Host your pre-trained model in a storage location where we are able to access. 

2. Add a model generation python file called `mindspore_hub_conf.py` in your own repo using this [template](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/googlenet/mindspore_hub_conf.py). The location of the `mindspore_hub_conf.py` file is shown below:

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

3. Create a `{model_name}_{model_version}_{dataset}.md` file in `hub/mshub_res/assets/mindspore/ascend/0.7` using this [template](https://gitee.com/mindspore/hub/blob/master/mshub_res/assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md). Here `ascend` refers to the hardware platform for the pre-trained model, and `0.7` indicates the MindSpore version. The structure of the `hub/mshub_res` folder is as follows：

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

   Note that it is required to fill in the `{model_name}_{model_version}_{dataset}.md` template by providing `file-format`、`asset-link` and `asset-sha256` below, which refers to the model file format, model storage location from step 1 and model hash value, respectively.

   ```shell
   file-format: ckpt  
   asset-link: https://download.mindspore.cn/model_zoo/official/cv/googlenet/goolenet_ascend_0.2.0_cifar10_official_classification_20200713/googlenet.ckpt  
   asset-sha256: 114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7
   ```   

   The MindSpore Hub supports multiple model file formats including:
   - [MindSpore CKPT](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#checkpoint-configuration-policies)
   - [AIR](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#export-air-model)
   - [MindIR](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#export-mindir-model)
   - [ONNX](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#export-onnx-model)
   - [MSLite](https://www.mindspore.cn/doc/lite/en/r1.0/use/converter_tool.html)

   For each pre-trained model, please run the following command to obtain a hash value required at `asset-sha256` of this `.md` file. Here the pre-trained model `googlenet.ckpt` is accessed from the storage location in step 1 and then saved in `tools` folder. The output hash value is: `114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7`.

   ```python
   cd ../tools
   python get_sha256.py ../googlenet.ckpt
   ```

4. Check the format of the markdown file locally using `hub/mshub_res/tools/md_validator.py` by running the following command. The output is `All Passed`，which indicates that the format and content of the `.md` file meets the requirements.

   ```python
   python md_validator.py ../assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md
   ```

5. Create a PR in `mindspore/hub` repo. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md) for more information about creating a PR. 

Once your PR is merged into master branch here, your model will show up in [MindSpore Hub Website](https://www.mindspore.cn/resources/hub) within 24 hours. Please refer to [README](https://gitee.com/mindspore/hub/blob/master/mshub_res/README.md) for more information about model submission. 
