# MindSpore Installation Guide

This document describes how to quickly install MindSpore in a Ubuntu system with a CPU environment.

<!-- TOC -->

- [MindSpore Installation Guide](#mindspore-installation-guide)
    - [Environment Requirements](#environment-requirements)
        - [System Requirements and Software Dependencies](#system-requirements-and-software-dependencies)
        - [(Optional) Installing Conda](#optional-installing-conda)
    - [Installation Guide](#installation-guide)
        - [Installing Using Executable Files](#installing-using-executable-files)
        - [Installing Using the Source Code](#installing-using-the-source-code)
        - [Version Update](#version-update)
- [Installing MindArmour](#installing-mindarmour)
- [Installing MindSpore Hub](#installing-mindspore-hub)

<!-- /TOC -->

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_cpu_install_en.md)

This document describes how to quickly install MindSpore in a Ubuntu system with a CPU environment.

## Environment Requirements

### System Requirements and Software Dependencies

| Version | Operating System | Executable File Installation Dependencies | Source Code Compilation and Installation Dependencies |
| ---- | :--- | :--- | :--- |
| MindSpore 1.0.1 | - Ubuntu 18.04 x86_64 <br> - Ubuntu 18.04 aarch64 | - [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5 <br> - For details about other dependency items, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt). | **Compilation dependencies:**<br> - [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5 <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) 7.3.0 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [patch](http://ftp.gnu.org/gnu/patch/) >= 2.5 <br> same as the executable file installation dependencies. |

- GCC 7.3.0 can be installed by using apt command.
- If Python has already installed, using `python --version` to check whether the version match, make sure your Python was added in environment variable `PATH`.
- Add pip to the environment variable to ensure that Python related toolkits can be installed directly through pip. You can get pip installer here `https://pypi.org/project/pip/` if pip is not install.
- When the network is connected, dependency items in the `requirements.txt` file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

### (Optional) Installing Conda

1. Download the Conda installation package from the following path:

   - [X86 Anaconda](https://www.anaconda.com/distribution/) or [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. Create and activate the Python environment.

    ```bash
    conda create -n {your_env_name} python=3.7.5
    conda activate {your_env_name}
    ```

> Conda is a powerful Python environment management tool. Beginners are adviced to check related information on the Internet.

## Installation Guide

### Installing Using Executable Files

1. Download the .whl package from the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first and run the following command to install MindSpore:

    ```bash
    pip install mindspore-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
    ```

2. Run the following command. If no loading error message such as `No module named 'mindspore'` is displayed, the installation is successful.

    ```bash
    python -c 'import mindspore'
    ```

### Installing Using the Source Code

1. Download the source code from the code repository.

    ```bash
    git clone https://gitee.com/mindspore/mindspore.git -b r1.0
    ```

2. Run the following command in the root directory of the source code to compile MindSpore:

    ```bash
    bash build.sh -e cpu -j4
    ```
    >
    > - Before running the preceding command, ensure that the paths where the executable files cmake and patch store have been added to the environment variable PATH.
    > - Before running the preceding command, ensure that [OpenSSL](https://github.com/openssl/openssl) is installed and set system variable `export OPENSSL_ROOT_DIR="path/to/openssl/install/directory"`.
    > - In the `build.sh` script, the `git clone` command will be executed to obtain the code in the third-party dependency database. Ensure that the network settings of Git are correct.
    > - If the compiler performance is strong, you can add -j{Number of threads} in to script to increase the number of threads. For example, `bash build.sh -e cpu -j12`.

3. Run the following command to install MindSpore:

    ```bash
    chmod +x build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    pip install build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
    ```

4. Run the following command. If no loading error message such as `No module named 'mindspore'` is displayed, the installation is successful.

    ```bash
    python -c 'import mindspore'
    ```

# Installing MindArmour

If you need to conduct AI model security research or enhance the security of the model in you applications, you can install MindArmour.

## Environment Requirements

### System Requirements and Software Dependencies

| Version | Operating System | Executable File Installation Dependencies | Source Code Compilation and Installation Dependencies |
| ---- | :--- | :--- | :--- |
| MindArmour 1.0.1 | - Ubuntu 18.04 x86_64 <br> - Ubuntu 18.04 aarch64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore 1.0.1 <br> - For details about other dependency items, see [setup.py](https://gitee.com/mindspore/mindarmour/blob/r1.0/setup.py). | Same as the executable file installation dependencies. |

- When the network is connected, dependency items in the `setup.py` file are automatically downloaded during .whl package installation. In other cases, you need to manually install dependency items.

## Installation Guide

### Installing Using Executable Files

1. Download the .whl package from the [MindSpore website](https://www.mindspore.cn/versions/en). It is recommended to perform SHA-256 integrity verification first and run the following command to install MindArmour:

   ```bash
   pip install mindarmour-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
   ```

2. Run the following command. If no loading error message such as `No module named 'mindarmour'` is displayed, the installation is successful.

   ```bash
   python -c 'import mindarmour'
   ```

### Installing Using the Source Code

1. Download the source code from the code repository.

   ```bash
   git clone https://gitee.com/mindspore/mindarmour.git -b r1.0
   ```

2. Run the following command in the root directory of the source code to compile and install MindArmour:

   ```bash
   cd mindarmour
   python setup.py install
   ```

3. Run the following command. If no loading error message such as `No module named 'mindarmour'` is displayed, the installation is successful.

   ```bash
   python -c 'import mindarmour'
   ```

### Version Update

Using the following command if you need update MindSpore version.

- Update Online

    ```bash
    pip install --upgrade mindspore
    ```

- Update after source code compilation

    After successfully executing the compile script `build.sh` in the root path of the source code, find the whl package in path `build/package`, use the following command to update your version.

    ```bash
    pip install --upgrade mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    ```

# Installing MindSpore Hub

If you need to access and experience MindSpore pre-trained models quickly, you can install MindSpore Hub.

For details about install steps, see [MindSpore Hub](https://gitee.com/mindspore/hub/blob/master/README.md).
