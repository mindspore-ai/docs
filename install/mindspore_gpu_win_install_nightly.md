# pip方式安装MindSpore GPU Nightly版本-Windows

<!-- TOC -->

- [pip方式安装MindSpore GPU Nightly版本-Windows](#pip方式安装mindspore-gpu-nightly版本-windows)
    - [确认系统环境信息](#确认系统环境信息)
    - [下载安装MindSpore](#下载安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/install/mindspore_gpu_win_install_nightly.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

MindSpore Nightly是包含当前最新功能与bugfix的预览版本，但是可能未经完整的测试与验证，希望体验最新功能或者问题修复的用户可以使用该版本。

本文档介绍如何在GPU环境的Windows系统上，使用pip方式快速安装MindSpore Nightly。

## 确认系统环境信息

- 当前支持Windows 10/11，x86架构64位的操作系统。
- 确认安装Python（>=3.7.5）。可以从[Python官网](https://www.python.org/downloads/windows/)或者[华为云](https://repo.huaweicloud.com/python/)选择合适的版本进行安装。
- 确认安装[CUDA 11.1/11.6](https://developer.nvidia.com/cuda-11.1.1-download-archive)以及[cuDNN](https://developer.nvidia.com/cudnn)，cuDNN的安装可以参考Nvidia[官方文档](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)，同时需要将cuDNN的安装目录加入到`CUDNN_HOME`环境变量。

## 下载安装MindSpore

执行如下命令安装MindSpore：

```bash
pip install mindspore-cuda11-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- MindSpore GPU Nightly for Windows当前仅提供CUDA11.6版本。
- 在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r2.0.0-alpha/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/requirements.txt)。
- pip会自动安装当前最新版本的Nightly版本MindSpore，如果需要安装指定版本，请参照下方升级MindSpore版本相关指导，在下载时手动指定版本。

## 验证是否成功安装

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore-cuda11-dev=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如2.0.0rc1.dev20211125；如果希望自动升级到最新版本，`=={version}`字段可以缺省。

注意：当前MindSpore GPU Nightly for Windows仅提供CUDA11.6版本，若仍希望使用CUDA10.1或者CUDA11.1版本，请参考源码编译指导在对应环境上自行编译。
