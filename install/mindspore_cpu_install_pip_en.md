# Installing MindSpore in CPU by pip

<!-- TOC -->

- [Installing MindSpore in CPU by pip](#installing-mindspore-in-cpu-by-pip)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading and Installing MindSpore](#downloading-and-installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)
    - [Installing MindArmour](#installing-mindarmour)
    - [Installing MindSpore Hub](#installing-mindspore-hub)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_cpu_install_pip_en.md)

This document describes how to quickly install MindSpore by pip in a Linux system with a CPU environment.

## System Environment Information Confirmation

- Confirm that Ubuntu 18.04 is installed with 64-bit operating system.
- Confirm that [GCC 7.3.0](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) is installed.
- Confirm that Python 3.7.5 is installed.
    - If you didn't install Python or you have installed other versions, please download the Python 3.7.5 64-bit from [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [Huaweicloud](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz) to install.

## Downloading and Installing MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/{system}/mindspore-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt)). In other cases, you need to manually install dependency items.  
- `{version}` denotes the version of MindSpore. For example, when you are downloading MindSpore 1.0.1, `{version}` should be 1.0.1.  
- `{arch}` denotes the system architecture. For example, the Linux system you are using is x86 architecture 64-bit, `{arch}` should be `x86_64`. If the system is ARM architecture 64-bit, then it should be `aarch64`.  
- `{system}` denotes the system version. For example, if you are using Ubuntu x86 architecture, `{system}` should be `ubuntu_x86`. Currently, the following systems are supported by CPU: `ubuntu_aarch64`/`ubuntu_x86`.

## Installation Verification

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

If the MindSpore version number is output, it means that MindSpore is installed successfully, and if the output is `No module named 'mindspore'`, it means that the installation was not successful.

## Version Update

Using the following command if you need update MindSpore version:

```bash
pip install --upgrade mindspore
```

## Installing MindArmour

If you need to conduct AI model security research or enhance the security of the model in you applications, you can install MindArmour.

For more details, please refer to [MindArmour](https://gitee.com/mindspore/mindarmour/blob/master/README.md).

## Installing MindSpore Hub

If you need to access and experience MindSpore pre-trained models quickly, you can install MindSpore Hub.

For more details, please refer to [MindSpore Hub](https://gitee.com/mindspore/hub/blob/master/README.md).
