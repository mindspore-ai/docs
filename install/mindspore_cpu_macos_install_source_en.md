# Installing MindSpore in CPU by Source Code (macOS)

<!-- TOC -->

- [Installing MindSpore in CPU by Source Code (macOS)](#installing-mindspore-in-cpu-by-source-code-macOS)
    - [System Environment Information Confirmation](#system-environment-information-confirmation)
    - [Downloading Source Code from Code Repository](#downloading-source-code-from-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Installation Verification](#installing-verification)
    - [Version Update](#version-update)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.1/install/mindspore_cpu_macos_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.1/resource/_static/logo_source.png"></a>

This document describes how to quickly install MindSpore by source code in a macOS system with a CPU environment.

## System Environment Information Confirmation

- Confirm that macOS Catalina is installed with the x86 architecture 64-bit operating system.
- Confirm that the Xcode and Clang 11.0.0 is installed.
- Confirm that [CMake 3.18.3](https://github.com/Kitware/Cmake/releases/tag/v3.18.3) is installed.
    - After installing, add the path of `cmake` to the environment variable PATH.
- Confirm that [Python 3.7.5](https://www.python.org/ftp/python/3.7.5/python-3.7.5-macosx10.9.pkg) is installed.
    - After installing, add the path of `python` to the environment variable PATH.
- Confirm that [OpenSSL 1.1.1 or later](https://github.com/openssl/openssl.git) is installed.
    - After installing, add the path of `Openssl` to the environment variable PATH.
- Confirm that [wheel 0.32.0 or later](https://pypi.org/project/wheel/) is installed.
- Confirm that the git tool is installed.  
     If not, use the following command to install it:

    ```bash
    brew install git
    ```

## Downloading Source Code from Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.1
```

## Compiling MindSpore

Run the following command in the root directory of the source code to compile MindSpore:

```bash
bash build.sh -e cpu
```

## Installing MindSpore

```bash
pip install build/package/mindspore-{version}-py37-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependency items are automatically downloaded during .whl package installation. (For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.1/requirements.txt)). In other cases, you need to manually install dependency items.
- `{version}` denotes the version of MindSpore. For example, when you are downloading MindSpore 1.0.1, `{version}` should be 1.0.1.

## Installation Verification

```bash
python -c "import mindspore;print(mindspore.__version__)"
```

If the MindSpore version number is displayed, it means that MindSpore is installed successfully, and if the output is `No module named 'mindspore'`, it means that the installation was not successful.

## Version Update

Using the following command if you need to update MindSpore version:

- Update online

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.bat` in the root path of the source code, find the whl package in path `build/package`, use the following command to update your version.

```bash
pip install --upgrade mindspore-{version}-cp37-cp37m-win_amd64.whl
```
