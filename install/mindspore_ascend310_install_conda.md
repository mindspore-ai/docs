# Conda方式安装MindSpore Ascend 310版本

<!-- TOC -->

- [Conda方式安装MindSpore Ascend 310版本](#conda方式安装mindspore-ascend-310版本)
    - [自动安装](#自动安装)
    - [手动安装](#手动安装)
        - [安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)
        - [安装Conda](#安装conda)
        - [安装GCC](#安装gcc)
        - [安装gmp](#安装gmp)
        - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
        - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend310_install_conda.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包与该计算平台需要的所有库。

本文档介绍如何在Ascend 310环境的Linux系统上，使用Conda方式快速安装MindSpore，Ascend 310版本仅支持推理。

- 如果想在一个已经配置好昇腾AI处理器配套软件包的EulerOS 2.8上通过Conda安装MindSpore，可以使用[自动安装脚本](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/euleros-ascend-conda.sh)进行一键式安装，参见[自动安装](#自动安装)小节。自动安装脚本会安装MindSpore及其所需的依赖。

- 如果您的系统已经安装了部分依赖，如CUDA，Conda，GCC等，则推荐参照[手动安装](#手动安装)小节的安装步骤手动安装。

## 自动安装

在使用自动安装脚本之前，需要确保系统正确安装了昇腾AI处理器配套软件包。如果没有安装，请先参考[安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)小节进行安装。

使用以下命令获取自动安装脚本并执行。自动安装脚本仅支持安装MindSpore>=1.6.0。

```bash
wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/euleros-ascend-conda.sh
# 默认安装Python 3.7以及最新版本的MindSpore
# 默认LOCAL_ASCEND路径为/usr/local/Ascend
bash -i ./euleros-ascend-conda.sh
# 如需指定Python和MindSpore版本，以Python 3.9和MindSpore 1.6.0为例
# 且指定LOCAL_ASCEND路径为/home/xxx/Ascend，使用以下方式
# LOCAL_ASCEND=/home/xxx/Ascend PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash -i ./euleros-ascend-conda.sh
```

该脚本会执行以下操作：

- 安装MindSpore所需的依赖，如GCC，gmp。
- 安装Conda并为MindSpore创建虚拟环境。
- 通过Conda安装MindSpore Ascend 310版本。

在脚本执行完成后，需要重新打开终端窗口，然后参照[配置环境变量](#配置环境变量)中的说明设置相关环境变量。

自动安装脚本会为MindSpore创建名为`mindspore_pyXX`的虚拟环境。其中`XX`为Python版本，如Python 3.7则虚拟环境名为`mindspore_py37`。执行以下命令查看所有虚拟环境。

```bash
conda env list
```

要激活虚拟环境，以Python 3.7为例，执行以下命令。

```bash
conda activate mindspore_py37
```

更多的用法请参看脚本头部的说明。

## 手动安装

下表列出了安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/OpenEuler 20.03/KylinV10 SP1|-|编译和运行MindSpore的操作系统|
|[昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)|-|MindSpore使用的Ascend平台AI计算库|
|[Conda](#安装conda)|Anaconda3或Miniconda3|Python环境管理工具|
|[GCC](#安装gcc)|7.3.0|用于编译MindSpore的C++编译器|
|[gmp](#安装gmp)|6.1.2|MindSpore使用的多精度算术库|

下面给出第三方依赖的安装方法。

### 安装昇腾AI处理器配套软件包

详细安装方法请参考[Ascend Data Center Solution 21.1.0安装指引文档]。

安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。

### 安装Conda

执行以下指令安装Miniconda。

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

### 安装GCC

- Ubuntu 18.04可以使用以下命令安装。

    ```bash
    sudo apt-get install gcc-7 -y
    ```

- CentOS 7可以使用以下命令安装。

    ```bash
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    ```

    安装完成后，需要使用如下命令切换到GCC 7。

    ```bash
    scl enable devtoolset-7 bash
    ```

- EulerOS可以使用以下命令安装。

    ```bash
    sudo yum install gcc -y
    ```

### 安装gmp

- Ubuntu 18.04可以使用以下命令安装。

    ```bash
    sudo apt-get install libgmp-dev -y
    ```

- CentOS 7和EulerOS可以使用以下命令安装。

    ```bash
    sudo yum install gmp-devel -y
    ```

### 创建并进入Conda虚拟环境

根据您希望使用的Python版本创建对应的Conda虚拟环境并进入虚拟环境。

如果您希望使用Python3.7.5版本：

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

如果希望使用其他版本Python，只需更改以上命令中的Python版本。当前支持Python 3.7，Python 3.8和Python 3.9。

在虚拟环境中安装昇腾AI处理器配套软件包提供的whl包，whl包随配套软件包发布，升级配套软件包之后需要重新安装。

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*-py3-none-any.whl
```

如果升级了昇腾AI处理器配套软件包，配套的whl包也需要重新安装，先将原来的安装包卸载，再参考上述命令重新安装。

```bash
pip uninstall te topi hccl -y
```

### 安装MindSpore

确认您处于Conda虚拟环境中，并执行如下命令安装最新版本的MindSpore。如需安装其他版本，可参考[版本列表](https://www.mindspore.cn/versions)在`mindspore-ascend=`后指定版本号。

```bash
conda install mindspore-ascend -c mindspore -c conda-forge
```

在联网状态下，安装Conda安装包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py)中的required_package），其余情况需自行安装。运行模型时，需要根据[ModelZoo](https://gitee.com/mindspore/models/tree/master/)中不同模型指定的requirements.txt安装额外依赖，常见依赖可以参考[requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt)。

## 配置环境变量

安装好MindSpore之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

## 验证是否成功安装

创建目录放置样例代码工程，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample`，代码可以从[官网示例下载](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/sample_resources/ascend310_single_op_sample.zip)获取，这是一个`[1, 2, 3, 4]`与`[2, 3, 4, 5]`相加的简单样例，代码工程目录结构如下:

```text

└─ascend310_single_op_sample
    ├── CMakeLists.txt                    // 编译脚本
    ├── README.md                         // 使用说明
    ├── main.cc                           // 主函数
    └── tensor_add.mindir                 // MindIR模型文件
```

进入样例工程目录，按照实际情况修改路径路径：

```bash
cd /home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample
```

参照`README.md`说明，构建工程，其中`pip3`需要按照实际情况修改。

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
make
```

构建成功后，执行用例。

```bash
./tensor_add_sample
```

如果输出：

```text
3
5
7
9
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
conda update mindspore-ascend -c mindspore -c conda-forge
```
