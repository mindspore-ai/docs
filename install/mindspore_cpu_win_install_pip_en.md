# Installing MindSpore in CPU by pip (Windows)

<!-- TOC -->

- [Installing MindSpore in CPU by pip (Windows)](#installing-mindspore-in-cpu-by-pip-windows)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installation-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_cpu_win_install_pip_en.md)

This document describes how to quickly install MindSpore by pip in a Windows system with a CPU environment.

## System Environment Information Confirmation

- Confirm that Windows 10 is installed with x86 architecture 64-bit operating system.
- Confirm that Python 3.7.5 is installed.  
    - If you didn't install Python or you have installed other versions, please download the Python 3.7.5 64-bit from [Huaweicloud](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz) to install.
- After installing Python, add Python and pip to the environment variable.
    - Add Python: Control Panel -> System -> Advanced System Settings -> Environment Variables. Double click the Path in the environment variable and add the path of `python.exe`.
    - Add pip: The `Scripts` folder in the same directory of `python.exe` is the pip file that comes with Python, add it to the system environment variable.

## Installing MindSpore

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/cpu/windows_x64/mindspore-{version}-cp37-cp37m-win_amd64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt)). In other cases, you need to manually install dependency items.  
- `{version}` denotes the version of MindSpore. For example, when you are downloading MindSpore 1.0.1, `{version}` should be 1.0.1.

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
