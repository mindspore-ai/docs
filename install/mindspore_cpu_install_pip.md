# pip方式安装MindSpore CPU版本

<!-- TOC -->

- [pip方式安装MindSpore CPU版本](#pip方式安装mindspore-cpu版本)
    - [自动安装](#自动安装)
    - [手动安装](#手动安装)
        - [安装Python](#安装python)
        - [安装GCC和gmp](#安装gcc和gmp)
        - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_cpu_install_pip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

本文档介绍如何在CPU环境的Linux系统上，使用pip方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore安装步骤。

- 如果您想在一个全新的Ubuntu 18.04上通过pip安装MindSpore，可以使用[自动安装脚本](https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-cpu-pip.sh)进行一键式安装，参见[自动安装](#自动安装)小节。自动安装脚本会安装MindSpore及其所需的依赖。

- 如果您的系统已经安装了部分依赖，如Python，GCC等，则推荐参照[手动安装](#手动安装)小节的安装步骤手动安装。

## 自动安装

自动安装脚本需要更改软件源配置以及通过APT安装依赖，所以需要申请root权限。使用以下命令获取自动安装脚本并执行。自动安装脚本仅支持安装MindSpore>=1.6.0。

```bash
wget https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/ubuntu-cpu-pip.sh
# 安装MindSpore 1.7.0和Python 3.7
MINDSPORE_VERSION=1.7.0 bash ./ubuntu-cpu-pip.sh
# 如需指定Python和MindSpore版本，以Python 3.9和MindSpore 1.6.0为例，使用以下方式
# PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash ./ubuntu-cpu-pip.sh
```

该脚本会执行以下操作：

- 更改软件源配置为华为云源。
- 安装MindSpore所需的依赖，如GCC，gmp。
- 通过APT安装Python3和pip3，并设为默认。
- 通过pip安装MindSpore CPU版本。

自动安装脚本执行完成后，需要重新打开终端窗口以使环境变量生效。

更多的用法请参看脚本头部的说明。

## 手动安装

下表列出了安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|运行MindSpore的操作系统|
|[Python](#安装python)|3.7-3.9|MindSpore的使用依赖Python环境|
|[GCC](#安装gcc和gmp)|7.3.0到9.4.0之间|用于编译MindSpore的C++编译器|
|[gmp](#安装gcc和gmp)|6.1.2|MindSpore使用的多精度算术库|

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

    创建虚拟环境，以Python 3.7.5为例：

    ```bash
    conda create -n mindspore_py37 python=3.7.5 -y
    conda activate mindspore_py37
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

    若要安装其他Python版本，只需更改命令中的`3.7`。

可以通过以下命令查看Python版本。

```bash
python --version
```

### 安装GCC和gmp

可以通过以下命令安装GCC和gmp。

```bash
sudo apt-get install gcc-7 libgmp-dev -y
```

如果要安装更高版本的GCC，使用以下命令安装GCC 8。

```bash
sudo apt-get install gcc-8 -y
```

或者安装GCC 9。

```bash
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y
```

### 安装MindSpore

首先参考[版本列表](https://www.mindspore.cn/versions)选择想要安装的MindSpore版本，并进行SHA-256完整性校验。以1.7.0版本为例，执行以下命令。

```bash
export MS_VERSION=1.7.0
```

然后根据系统架构及Python版本执行如下命令安装MindSpore。

```bash
# x86_64 + Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp38-cp38-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/x86_64/mindspore-${MS_VERSION/-/}-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.7
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.8
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp38-cp38-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/cpu/aarch64/mindspore-${MS_VERSION/-/}-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/r1.7/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/r1.7/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.7/requirements.txt)。

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

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.6.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
