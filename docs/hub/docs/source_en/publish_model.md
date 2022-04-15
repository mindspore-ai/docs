# Publishing Models using MindSpore Hub

<a href="https://gitee.com/mindspore/docs/blob/master/docs/hub/docs/source_en/publish_model.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

[MindSpore Hub](https://www.mindspore.cn/resources/hub/) is a platform for storing pre-trained models provided by MindSpore or third-party developers. It provides application developers with simple model loading and fine-tuning APIs, which enables the users to perform inference or fine-tuning based on the pre-trained models and thus deploy to their own applications. Users can also submit their pre-trained models into MindSpore Hub following the specific steps. Thus other users can download and use the published models.

This tutorial uses GoogleNet as an example to describe how to submit models for model developers who are interested in publishing models into MindSpore Hub.

## How to publish models

You can publish models to MindSpore Hub via PR in [hub](https://gitee.com/mindspore/hub) repo. Here we use GoogleNet as an example to list the steps of model submission to MindSpore Hub.

1. Host your pre-trained model in a storage location where we are able to access.

2. Add a model generation python file called `mindspore_hub_conf.py` in your own repo using this [template](https://gitee.com/mindspore/models/blob/master/official/cv/googlenet/mindspore_hub_conf.py). The location of the `mindspore_hub_conf.py` file is shown below:

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

3. Create a `{model_name}_{model_version}_{dataset}.md` file in `hub/mshub_res/assets/mindspore/ascend/0.7` using this [template](https://gitee.com/mindspore/hub/blob/master/mshub_res/assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md#). Here `ascend` refers to the hardware platform for the pre-trained model, and `0.7` indicates the MindSpore version. The structure of the `hub/mshub_res` folder is as follows:

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

   Note that it is required to fill in the `{model_name}_{model_version}_{dataset}.md` template by providing `file-format`、`asset-link` and `asset-sha256` below, which refers to the model file format, model storage location from step 1 and model hash value, respectively.

   ```text
   file-format: ckpt
   asset-link: https://download.mindspore.cn/model_zoo/official/cv/googlenet/goolenet_ascend_0.2.0_cifar10_official_classification_20200713/googlenet.ckpt
   asset-sha256: 114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7
   ```

   The MindSpore Hub supports multiple model file formats including:
   - [MindSpore CKPT](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/train/save.html#saving-the-models)
   - [MindIR](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/train/save.html#export-mindir-model)
   - [AIR](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/train/save.html#export-air-model)
   - [ONNX](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/train/save.html#export-onnx-model)

   For each pre-trained model, please run the following command to obtain a hash value required at `asset-sha256` of this `.md` file. Here the pre-trained model `googlenet.ckpt` is accessed from the storage location in step 1 and then saved in `tools` folder. The output hash value is: `114e5acc31dad444fa8ed2aafa02ca34734419f602b9299f3b53013dfc71b0f7`.

   ```bash
   cd ../tools
   python get_sha256.py --file ../googlenet.ckpt
   ```

4. Check the format of the markdown file locally using `hub/mshub_res/tools/md_validator.py` by running the following command. The output is `All Passed`，which indicates that the format and content of the `.md` file meets the requirements.

   ```bash
   python md_validator.py --check_path ../assets/mindspore/ascend/0.7/googlenet_v1_cifar10.md
   ```

5. Create a PR in `mindspore/hub` repo. See our [Contributor Wiki](https://gitee.com/mindspore/mindspore/blob/master/CONTRIBUTING.md#) for more information about creating a PR.

Once your PR is merged into master branch here, your model will show up in [MindSpore Hub Website](https://www.mindspore.cn/resources/hub) within 24 hours. Please refer to [README](https://gitee.com/mindspore/hub/blob/master/mshub_res/README.md#) for more information about model submission.
