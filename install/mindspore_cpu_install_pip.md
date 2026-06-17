# pip方式安装MindSpore CPU版本

<!-- TOC -->

- [pip方式安装MindSpore CPU版本](#pip方式安装mindspore-cpu版本)
    - [安装MindSpore与依赖软件](#安装mindspore与依赖软件)
        - [安装Python](#安装python)
        - [安装GCC](#安装gcc)
        - [安装MindSpore](#安装mindspore)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/install/mindspore_cpu_install_pip.md)

本文档介绍如何在CPU环境的Linux系统上，使用pip方式快速安装MindSpore。下面以Ubuntu 18.04为例说明MindSpore安装步骤。

## 安装MindSpore与依赖软件

下表列出了安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu|18.04|运行MindSpore的操作系统|
|[Python](#安装python)|3.9-3.12|MindSpore的使用依赖Python环境|
|[GCC](#安装gcc)|9.5.0-11.3.0 （优选9.5.0）|用于编译MindSpore的C++编译器|

下面给出第三方依赖的安装方法。

### 安装Python

[Python](https://www.python.org/)可通过Conda进行安装。

安装Miniconda：

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py310_26.1.1-1-Linux-$(arch).sh
bash Miniconda3-py310_26.1.1-1-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

创建虚拟环境，以Python 3.10.20为例：

```bash
conda create -n mindspore_py310 python=3.10.20 -y
conda activate mindspore_py310
```

可以通过以下命令查看Python版本。

```bash
python --version
```

### 安装GCC

下面以GCC 9为例，展示GCC在常用操作系统上的安装方式：

- Ubuntu可以使用以下命令安装。

    ```bash
    sudo apt-get install gcc-9 g++-9 -y
    ```

- CentOS可以使用以下命令安装。

    ```bash
    sudo yum install centos-release-scl
    sudo yum install -y devtoolset-9-gcc devtoolset-9-gcc-c++ devtoolset-9-binutils
    ```

    安装完成后，需要使用以下命令切换到GCC 9。

    ```bash
    scl enable devtoolset-9 bash
    ```

- EulerOS和openEuler可以使用以下命令安装。

    ```bash
    sudo yum install gcc g++ -y
    ```

### 安装MindSpore

首先参考[版本列表](https://www.mindspore.cn/versions)，选择想要安装的MindSpore版本，并进行SHA-256完整性校验。以2.9.0版本为例，执行以下命令。

```bash
export MS_VERSION=2.9.0
```

然后执行以下命令安装MindSpore。

```bash
pip install mindspore==${MS_VERSION} -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple/
```

在联网状态下，安装whl包时会自动下载mindspore安装包的依赖项（依赖项详情参见[setup.py](https://atomgit.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装依赖。

## 验证是否成功安装

执行以下命令：

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行以下命令：

```bash
pip install --upgrade mindspore=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.6.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
