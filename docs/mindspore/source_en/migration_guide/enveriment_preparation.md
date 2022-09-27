# Environment Preparation and Information Acquisition

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/migration_guide/enveriment_preparation.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

Network migration starts with the configuration of the MindSpore development environment. This chapter describes in detail the installation process and related materials. The materials include a basic introduction to the MindSpore components models and Hub, including their uses, scenarios and usage. There are also tutorials on training on the cloud: using ModelArts to adapt scripts, uploading datasets in OBS, training online, etc.

## Installing MindSpore

[MindSpore](https://www.mindspore.cn/tutorials/en/master/beginner/introduction.html) is a full-scene deep learning framework, which currently supports running on [Ascend](https://e.huawei.com/cn/products/servers/ascend), GPU, CPU and other kinds of devices. Before installing MindSpore on Ascend and GPU, you need to configure the corresponding runtime environment.

> MindSpore Ascend supports AI training (910), inference cards (310 and 310P) and training servers on various Atlas series chips, etc. Note that MindSpore version needs to be used with Ascend AI processor package. For example, MindSpore 1.8.1 must be used with Ascend package commercial version 22.0.RC2 or CANN Community version 5.1.RC2.alpha008. There may be problems with other versions. Please refer to the "Installing Ascend AI Processor Package" section in the MindSpore Ascend Version Installation Guide for more details.
>
> The MindSpore GPU supports CUDA 10.1 and CUDA 11.1 on Linux environments. NVIDIA provides a variety of installation methods and installation instructions, which can be found on the [CUDA download page](https://developer.nvidia.com/cuda-toolkit-archive) and and [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
>
> MindSpore CPU currently supports Linux, Windows and Mac.

After the package is installed, follow the MindSpore installation guide and install the corresponding version of MindSpore to experience.

Refer to the following table to determine the release version, architecture (X86 or Arm), and Python version.

|System|Query Content|Query commands|
|:----|:----|:----|
|Linux|Release Version| `cat /proc/version`|
|Linux|Architecture| `uname -m`|
|Linux|Python Version| `python3`|

According to the operating system and computing hardware you are using, choose the corresponding MindSpore version and install MindSpore by Pip, Conda, Docker or source code compilation, etc. We recommend visiting [MindSpore Installation](https://www.mindspore.cn/install/en) and referring to the website for instructions to complete the installation and verification.

## models and hub

[MindSpore models](https://gitee.com/mindspore/models) is a deep-optimized model bazaar jointly provided by MindSpore and the community, which provides developers with deep-optimized models. The ecological partners can easily personalize their development based on the models in ModelZoo. Currently it has covered mainstream models in many fields such as machine vision, natural language processing, speech, recommendation systems.

At present, there are 300+ model implementations, among which the network under the official directory is the official network, with some optimization for the model implementation. Most of the models under the research directory are the models of Zongzhi, with certain guarantee of accuracy and performance. The community directory is the code contributed by developers, which has not been maintained yet and is for reference only.

[mindspore Hub](https://www.mindspore.cn/resources/hub/en) is a platform for storing pre-trained models provided by MindSpore official or third-party developers. It provides easy-to-use model loading and fine-tuning APIs to application developers, enabling users to make inference or fine-tuning based on pre-trained models and deploy them to their own applications. Users can also follow the specified steps on [Publish Model](https://www.mindspore.cn/hub/docs/en/master/publish_model.html), publishing their trained models into MindSpore Hub for other users to download and use.

The [Download Center](https://download.mindspore.cn/model_zoo/) saves the parameter files of the trained models in the models bin, where users can download the corresponding parameter files for development.

## ModelArts

### Development Environment Introduction

ModelArts is a one-stop development platform for AI developers provided by HUAWEI Cloud, which contains Ascend resource pool. Users can experience MindSpore in this platform. For related document, refer to [AI Platform ModelArts](https://support.huaweicloud.com/wtsnew-modelarts/index.html).

### Development Environment and Training Environment

**Development environment** mainly refers to Notebook's development environment, which is mainly used for code writing and debugging and has almost the same development experience as offline, but the number of machines and cores is relatively small and the usage time is limited.

![notebook](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/migration_guide/images/modelart_notebook.png "Development environment")

**Training environment** is the primary production environment on the cloud with more machines and cores for large clusters and batch tasks.

### Introduction to Development Environment Storage Method

The storage methods supported by the development environment are OBS and EFS, as shown in the following figure.

![ModelArts](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/migration_guide/images/modelart.png "Development Environment Storage Method")

OBS: also called S3 bucket. The code, data, and pre-trained models stored on the OBS need to be transferred to the corresponding physical machine first in the development environment and training environment before the job is executed. [Upload local files to OBS](https://bbs.huaweicloud.com/blogs/212453).

[MoXing](https://bbs.huaweicloud.com/blogs/101129): MoXing is a network model development API provided by Huawei Cloud Deep Learning Service, the use of which needs to focus on the data copy interface.

```python
import moxing as mox
mox.file.copy_parallel(src_path, target_path)  # Copy the data from the OBS bucket to the physical machine where it is actually executed or vice versa
```

EFS: It can be understood as a mountable cloud disk, which can be directly mounted to the corresponding physical machine in the development environment and training environment to facilitate the execution of jobs.

Welcome to click the video link below.

<div style="position: relative; padding: 30% 45%;">
<iframe style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;" src="https://player.bilibili.com/player.html?aid=814612708&bvid=BV16G4y1a7A8&cid=805013543&page=1&high_quality=1&&danmaku=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"></iframe>
</div>