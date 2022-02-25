# pip方式安装MindSpore CPU版本

<!-- TOC -->

- [pip方式安装MindSpore CPU版本](#pip方式安装mindspore-cpu版本)
    - [环境准备](#环境准备)
    - [下载安装MindSpore](#下载安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Linux系统上，使用pip方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore安装步骤。

- 如果您想在一个全新的Ubuntu 18.04上通过pip安装MindSpore，可以使用[自动安装脚本](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-pip.sh)进行一键式安装。自动安装脚本会安装MindSpore及其所需的依赖。

    自动安装脚本需要更改软件源配置以及通过APT安装依赖，所以需要root权限执行。使用以下命令获取自动安装脚本并执行。

    ```bash
    wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/ubuntu-cpu-pip.sh
    # 默认安装Python 3.7和MindSpore 1.6.0
    sudo bash ./ubuntu-cpu-pip.sh
    # 如需指定Python和MindSpore版本，以Python 3.9和MindSpore 1.5.0为例，使用以下方式
    # sudo PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.5.0 bash ./ubuntu-cpu-pip.sh
    ```

    该脚本会执行以下操作：

    - 更改软件源配置为华为云源。
    - 安装MindSpore所需的依赖，如GCC，gmp。
    - 通过APT安装Python3和pip3，并设为默认。
    - 通过pip安装MindSpore CPU版本。

    更多的用法请参看脚本头部的说明。

- 如果您的系统已经安装了部分依赖，如Python，GCC等，则推荐参照下面的安装步骤手动安装。

## 环境准备

下表列出了安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|运行MindSpore的操作系统|
|[Python](#安装python)|3.7.5或3.9.0|MindSpore的使用依赖Python环境|
|[GCC](#安装gccgmp)|7.3.0|用于编译MindSpore的C++编译器|
|[gmp](#安装gccgmp)|6.1.2|MindSpore使用的多精度算术库|

下面给出第三方依赖的安装方法。

### 安装Python

[Python](https://www.python.org/)可通过多种方式进行安装。

- 通过Conda安装Python

    安装Miniconda：

    ```bash
    cd /tmp
    curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
    bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
    cd -
    . ~/miniconda3/etc/profile.d/conda.sh
    conda init bash
    ```

    安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

    创建Python 3.7.5环境：

    ```bash
    conda create -n mindspore_py37 python=3.7.5 -y
    conda activate mindspore_py37
    ```

    或者创建Python 3.9.0环境：

    ```bash
    conda create -n mindspore_py39 python=3.9.0 -y
    conda activate mindspore_py39
    ```

- 通过APT安装Python，命令如下。

    ```bash
    sudo apt-get update
    sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get install python3.7 python3.7-dev python3.7-distutils python3-pip -y
    # 将新安装的Python设为默认
    sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 100
    # 安装pip
    python -m pip install pip -i https://repo.huaweicloud.com/repository/pypi/simple
    sudo update-alternatives --install /usr/bin/pip pip ~/.local/bin/pip3.7 100
    pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple
    ```

    若要安装Python 3.9版本，只需将命令中的`3.7`替换为`3.9`。

可以通过以下命令查看Python版本。

```bash
python --version
```

### 安装GCC/gmp

可以通过以下命令安装GCC和gmp。

```bash
sudo apt-get install gcc-7 libgmp-dev -y
```

## 下载安装MindSpore

参考[版本列表](https://www.mindspore.cn/versions)先进行SHA-256完整性校验，校验一致后根据系统架构及Python版本执行如下命令安装MindSpore 1.6.0。

如需安装其他版本的MindSpore，请参考[版本列表](https://www.mindspore.cn/versions)，获取对应版本的WHL包地址进行安装。

```bash
# x86_64/Python3.7
# SHA-256: b4fe66629150c47397722057c32c806cd3eece5e158a93c62cac0bc03b464e3f
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/x86_64/mindspore-1.6.0-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64/Python3.9
# SHA-256: 87151059854e7388c6b5d6d56a7b75087f171efdcd84896c61d0e9088fcebcc0
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/x86_64/mindspore-1.6.0-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64/Python3.7
# SHA-256: 5dc6b9abe668d53960773d6e8ac4dc6f7016c15cce5c9480045c74a3e456e40f
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/aarch64/mindspore-1.6.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64/Python3.9
# SHA-256: 4a8f49587b30b6a0413edba85ebbae07600fc84f8a1fa48c42a2359402bfc852
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.6.0/MindSpore/cpu/aarch64/mindspore-1.6.0-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。

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
pip install --upgrade mindspore=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.5.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
