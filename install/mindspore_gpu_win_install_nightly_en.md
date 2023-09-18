# Installing MindSpore GPU Nightly by pip-Windows

<!-- TOC -->

- [Installing MindSpore GPU Nightly by pip-Windows](#installing-mindspore-gpu-nightly-by-pip-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/install/mindspore_gpu_win_install_nightly_en.md)

MindSpore Nightly is a preview version which includes latest features and bugfixes, not fully supported and tested. Install MindSpore Nightly version if you wish to try out the latest changes on MindSpore.

This document describes how to install MindSpore Nightly by pip on Linux in a GPU environment.

## System Environment Information Confirmation

- Caurrently supports Windows 10/11 with the x86 architecture 64-bit operating system.
- Ensure that you have Python(>=3.7.5) installed. If not installed, follow the links to [Python official website](https://www.python.org/downloads/windows/) or [Huawei Cloud](https://repo.huaweicloud.com/python/) to download and install Python.
- Ensure that [CUDA 11.1/11.6](https://developer.nvidia.com/cuda-11.1.1-download-archive) and [cuDNN](https://developer.nvidia.com/cudnn) is installed. Nvidia [cuDNN official installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) is a good place to follow, and make sure put cuDNN install path into `CUDNN_HOME` environment variable.

## Installing MindSpore

Execute the following command to install MindSpore:

```bash
pip install mindspore-cuda11-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- Currently, MindSpore GPU Nightly for Windows only provides CUDA11.6 version.
- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py).) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).
- pip will be installing the latest version of MindSpore Nightly automatically. If you wish to specify the version to be installed, please refer to the instruction below regarding to version update, and specify version manually.

## Installation Verification

```bash
python -c "import mindspore;mindspore.set_context(device_target='GPU');mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [GPU] successfully!
```

It means MindSpore has been installed successfully.

## Version Update

Use the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore-cuda11-dev=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 2.0.0.rc1. When updating to a stable release, you can remove `=={version}`.

Note: Currently MindSpore GPU Nightly for Windows only provides CUDA11.6 version. If you wish to run on CUDA10.1 or CUDA11.1 environments, please refer to the source code compilation guide and compile MindSpore on corresponding environments.
