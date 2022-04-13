# Preparation

Translator: [Misaka19998](https://gitee.com/Misaka19998/docs/tree/master)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/preparation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Before developing or migrating networks, you need to install MindSpore and learn machine learning knowledge. Users have a choice to buy *Introduction to Deep learning with MindSpore* to learn related knowledge and visit [MindSpore Official website](https://www.mindspore.cn/en) to know how to use MindSpore.

## Installing MindSpore

Refer to the following figure, to determine the release version and the architecture(x86 or Arm) of the system, and the Python version.

| System | Query Content          | Query Command       |
| ------ | ---------------------- | ------------------- |
| Linux  | System Release Version | `cat /proc/version` |
| Linux  | System Architecture    | `uname -m`           |
| Linux  | Python Version         | `python3`           |

Choose a corresponding MindSpore version based on users own operating system. MindSpore is installed in the manner of Pip, Conda, Docker or source code compilation. It is recommended to visit the [MindSpore installation page](https://www.mindspore.cn), and complete the installation by referring to this website for instructions.

### Verifying MindSpore

After the MindSpore is installed, the following commands can be run (taking the MindSpore r1.6 as an example), to test whether the installation of the MindSpore has been completed.

```python
import mindspore
mindspore.run_check()
```

Output the result:

```text
MindSpore version: 1.6.0
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

## Knowledge Preparation

### MindSpore Tutorials

Users can read [MindSpore Tutorials](https://www.mindspore.cn/tutorials/experts/en/master/index.html) to learn how to train, debug, optimize and infer by MindSpore. Users can also see detailed MindSpore interfaces by referring to [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html).

### ModelZoo and Hub

[ModelZoo](https://gitee.com/mindspore/models/tree/master) is a model market of MindSpore and community, which provides deeply-optimized models to developers. In order that the users of MindSpore will have individual development conveniently based on models in ModelZoo. Currently, there are major models in several fields, like computer vision, natural language processing, audio and recommender systems.

[mindspore Hub](https://www.mindspore.cn/resources/hub/en) is a platform to save pretrained model of official MindSpore or third party developers. It provides some simple and useful APIs for developers to load and finetune models, so that users can infer or tune models based on pretrained models and deploy models to their applications. Users is able to follow some steps to [publish model](https://www.mindspore.cn/hub/docs/en/master/publish_model.html) to MindSpore Hub,for other developers to download and use.

### Training on the Cloud

ModelArts is a one-stop development platform for AI developers provided by HUAWEI Cloud, which contains Ascend resource pool. Users can experience MindSpore in this platform and read related document [AI Platform ModelArts](https://support.huaweicloud.com/intl/en-us/wtsnew-modelarts/index.html).
