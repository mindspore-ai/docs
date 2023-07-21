# Installing MindSpore in CPU by pip (macOS)

<!-- TOC -->

- [Installing MindSpore in CPU by pip (macOS)](#installing-mindspore-in-cpu-by-pip-macos)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_cpu_mac_install_pip_en.md)

This document describes how to quickly install MindSpore on macOS by pip.

## System Environment Information Confirmation

- According to your Macbook configuration(click "About This Mac" to get chip/arch info),choose the right Python version based on following table:

    |Chip|Architecture|macOS Version|Supported Python Version|
    |-|-|-|-|
    |M1|ARM|11.3|Python 3.9.1+(3.7.x is not supported with M1, 3.9.1 is the least supported version)|
    |Intel|x86_64|10.15/11.3|Python 3.7.5/Python 3.9.0|

- Ensure that right Python version is installed. If not installed, download and install Python from:

    - Python 3.7.5 (64-bit)：[Python official website](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg) or [HUAWEI CLOUD](https://repo.huaweicloud.com/python/3.7.5/python-3.7.5-macosx10.9.pkg).
    - Python 3.9.0 (64-bit)：[Python official website](https://www.python.org/ftp/python/3.9.0/python-3.9.0-macosx10.9.pkg) or [HUAWEI CLOUD](https://repo.huaweicloud.com/python/3.9.0/python-3.9.0-macosx10.9.pkg).
    - Python 3.9.1 (64-bit)：[Python official website](https://www.python.org/ftp/python/3.9.1/python-3.9.1-macos11.0.pkg) or [HUAWEI CLOUD](https://www.python.org/ftp/python/3.9.1/python-3.9.1-macos11.0.pkg).

## Installing MindSpore

It is recommended to refer to [Version List](https://www.mindspore.cn/versions/en) to perform SHA-256 integrity verification, and then execute the following command to install MindSpore after the verification is consistent.

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/{arch}/mindspore-{version}-{python_version}-macosx_{platform_version}_{platform_arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/r1.6/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).
- `{version}` specifies the MindSpore version number. For example, when installing MindSpore 1.5.0, set `{version}` to 1.5.0, when installing MindSpore 1.5.0-rc1, the first `{version}` which represents download path should be written as 1.5.0-rc1, and the second `{version}` which represents file name should be 1.5.0rc1.
- `{arch}` denotes the architecture. For example, the macOS you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If you use M1 chip, then it should be `aarch64`.
- `{python_version}` spcecifies the python version for which MindSpore is built. If you wish to use Python3.7.5,`{python_version}` should be `cp37-cp37m`. If Python3.9.0 is used, it should be `cp39-cp39`.
- `{platform_version}` specifies the macOS version number. If you use M1 chip, set `{platform_version}` to `11_0`, otherwise set `{platform_version}` to `10_15`.
- `{platform_arch}` denotes the system architecture. For example, the macOS you are using is x86 architecture 64-bit, `{platform_arch}` should be `x86_64`. If you use M1 chip, then it should be `arm64`.

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

Using the following command if you need to update the MindSpore version:

```bash
pip install --upgrade mindspore=={version}
```

Of which,

- When updating to a release candidate (rc) version, `{version}` should be specified, e.g. 1.5.0rc1; When updating to a standard release, `=={version}` could be removed.
