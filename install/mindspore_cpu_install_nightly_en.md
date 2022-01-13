# Installing MindSpore CPU Nightly by pip

<!-- TOC -->

- [Installing MindSpore CPU Nightly by pip](#installing-mindspore-cpu-nightly-by-pip)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading and Installing MindSpore](#downloading-and-installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.6/install/mindspore_cpu_install_nightly_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

MindSpore Nightly is a preview version which includes latest features and bugfixes, not fully supported and tested. Install MindSpore Nightly version if you wish to try out the latest changes on MindSpore.

This document describes how to quickly install MindSpore Nightly by pip in a Linux system with a CPU environment.

## System Environment Information Confirmation

- Ensure that the 64-bit operating system is installed and the [glibc](https://www.gnu.org/software/libc/)>=2.17, where Ubuntu 18.04 are verified.
- Ensure that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Ensure that [gmp 6.1.2](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz) is installed.
- Ensure that Python 3.7.5 or 3.9.0 is installed. If not installed, download and install Python from:
    - Python 3.7.5 (64-bit): [Python official website](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz).
    - Python 3.9.0 (64-bit): [Python official website](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz).

- If you are using a ARM architecture system, please ensure that pip installed for current Python has a version >= 19.3.

## Downloading and Installing MindSpore

Execute the following command to install MindSpore:

```bash
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).
- pip will be installing the latest version of MindSpore Nightly automatically. If you wish to specify the version to be installed, please refer to the instruction below regarding to version update, and specify version manually.

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
pip install --upgrade mindspore-dev=={version}
```

Of which,

- When updating to a release candidate (rc) version, `{version}` should be specified, e.g. 1.6.0rc1.dev20211125; When updating to a standard release, `=={version}` could be removed.
