# Installing MindSpore in GPU by pip-Windows

<!-- TOC -->

- [Installing MindSpore in GPU by pip-Windows](#installing-mindspore-in-cpu-by-pip-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/install/mindspore_gpu_win_install_pip_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by pip in a Windows system with a GPU environment.

## System Environment Information Confirmation

- Caurrently supports Windows 10/11 with the x86 architecture 64-bit operating system.
- Ensure that you have Python(>=3.7.5) installed. If not installed, follow the links to [Python official website](https://www.python.org/downloads/windows/) or [Huawei Cloud](https://repo.huaweicloud.com/python/) to download and install Python.
- Ensure that [CUDA 11.1/11.6](https://developer.nvidia.com/cuda-11.1.1-download-archive) and [cuDNN](https://developer.nvidia.com/cudnn) is installed. Nvidia [cuDNN official installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) is a good place to follow, and make sure put cuDNN install path into `CUDNN_HOME` environment variable.

## Installing MindSpore

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 1.9.0 as an example, execute the following commands.

```bash
set MS_VERSION=1.9.0
```

Then run the following commands to install MindSpore according to Python version.

```bash
# Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/gpu/x86_64/mindspore_gpu-%MS_VERSION:-=%-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/gpu/x86_64/mindspore_gpu-%MS_VERSION:-=%-cp38-cp38-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/gpu/x86_64/mindspore_gpu-%MS_VERSION:-=%-cp39-cp39-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r2.0/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r2.0.0-alpha/requirements.txt).

## Installation Verification

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Use the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore=={version}
```

Of which,

- When updating to a release candidate (rc) version, `{version}` should be specified manually, e.g. 2.0.0rc1; When updating to a standard release, `=={version}` could be removed.
