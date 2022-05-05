# Installing MindSpore in CPU by pip-Windows

<!-- TOC -->

- [Installing MindSpore in CPU by pip-Windows](#installing-mindspore-in-cpu-by-pip-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_cpu_win_install_pip_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

This document describes how to quickly install MindSpore by pip in a Windows system with a CPU environment.

For details about how to install third-party dependency software when confirming the system environment information, see the third-party dependency software installation section in the [Installing MindSpore Using Source Code Build on Windows (CPU)](https://www.mindspore.cn/news/newschildren?id=364) provided by the community. Thanks to the community member [lvmingfu](https://gitee.com/lvmingfu) for sharing.

## System Environment Information Confirmation

- Ensure that Windows 10 is installed with the x86 architecture 64-bit operating system.
- Ensure that you have Python versions between 3.7 to 3.9 installed. Consider the following examples:
    - Python 3.7.5 (64-bit): [Python official website](https://www.python.org/ftp/python/3.7.5/python-3.7.5-amd64.exe) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.7.5/python-3.7.5-amd64.exe).
    - Python 3.8.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.8.0/python-3.8.0-amd64.exe) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.8.0/python-3.8.0-amd64.exe).
    - Python 3.9.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.9.0/python-3.9.0-amd64.exe) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.9.0/python-3.9.0-amd64.exe).
- After installing Python, add Python and pip to the environment variable.
    - Add Python: Control Panel -> System -> Advanced System Settings -> Environment Variables. Double click the Path in the environment variable and add the path of `python.exe`.
    - Add pip: The `Scripts` folder in the same directory `python.exe` is the pip file that comes with Python, and add it to the system environment variable.

## Installing MindSpore

First, refer to [Version List](https://www.mindspore.cn/versions) to select the version of MindSpore you want to install, and perform SHA-256 integrity check. Taking version 1.7.0 as an example, execute the following commands.

```bash
set MS_VERSION=1.7.0
```

Then run the following commands to install MindSpore according to Python version.

```bash
# Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp38-cp38-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/%MS_VERSION%/MindSpore/cpu/x86_64/mindspore-%MS_VERSION:-=%-cp39-cp39-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt).

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

- When updating to a release candidate (rc) version, `{version}` should be specified manually, e.g. 1.5.0rc1; When updating to a standard release, `=={version}` could be removed.
