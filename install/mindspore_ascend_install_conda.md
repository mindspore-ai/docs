# Conda方式安装MindSpore Ascend版本

<!-- TOC -->

- [Conda方式安装MindSpore Ascend版本](#conda方式安装mindspore-ascend版本)
    - [安装MindSpore与依赖软件](#安装mindspore与依赖软件)
        - [安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)
        - [安装Conda](#安装conda)
        - [安装GCC](#安装gcc)
        - [创建并进入Conda虚拟环境](#创建并进入conda虚拟环境)
        - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/mindspore_ascend_install_conda.md)

[Conda](https://docs.conda.io/en/latest/)是一个开源跨平台语言无关的包管理与环境管理系统，允许用户方便地安装不同版本的二进制软件包，以及该计算平台需要的所有库。

本文档介绍如何在Ascend环境的Linux系统上，使用Conda方式快速安装MindSpore。

## 安装MindSpore与依赖软件

下表列出了安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu 18.04 / CentOS 7.6 / EulerOS 2.8 / openEuler 20.03 / KylinV10 SP1|-|编译和运行MindSpore的操作系统|
|[昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)|-|MindSpore使用的Ascend平台AI计算库|
|[Conda](#安装conda)|Anaconda3或Miniconda3|Python环境管理工具|
|[GCC](#安装gcc)|7.3.0|用于编译MindSpore的C++编译器|

下面给出第三方依赖的安装方法。

### 安装昇腾AI处理器配套软件包

昇腾软件包提供商用版和社区版两种下载途径：

- 商用版下载需要申请权限，下载链接即将发布。

- 社区版下载不受限制，下载链接请前往[CANN社区版](https://www.hiascend.com/developer/download/community/result?module=cann)，推荐优先选择`8.1.RC1.beta1`版本，以及在[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)链接中获取对应的固件和驱动安装包，安装包的选择与安装方式请参照上述的商用版安装指引文档。

安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。

### 安装Conda

执行以下命令安装Miniconda。

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
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

    安装完成后，需要使用以下命令切换到GCC 7。

    ```bash
    scl enable devtoolset-7 bash
    ```

- EulerOS和openEuler可以使用以下命令安装。

    ```bash
    sudo yum install gcc -y
    ```

### 创建并进入Conda虚拟环境

根据您希望使用的Python版本，创建对应的Conda虚拟环境，并进入虚拟环境。

如果您希望使用Python 3.9.11版本，执行以下命令：

```bash
conda create -c conda-forge -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

如果希望使用其他版本Python，只需更改以上命令中的Python版本。当前支持Python 3.9、Python 3.10和Python 3.11。

在虚拟环境中，安装昇腾AI处理器配套软件包提供的whl包，whl包随配套软件包发布，升级配套软件包之后需要重新安装。

```bash
pip install sympy
pip install "numpy>=1.20.0,<2.0.0"
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```

如果升级了昇腾AI处理器配套软件包，配套的whl包也需要重新安装，先将原来的安装包卸载，再参考以上命令重新安装。

```bash
pip uninstall te topi hccl -y
```

### 安装MindSpore

确认您处于Conda虚拟环境中，并执行以下命令安装最新版本的MindSpore。如需安装其他版本，可参考[版本列表](https://www.mindspore.cn/versions)，并在`conda install mindspore=`后指定版本号。

```bash
conda install mindspore -c mindspore -c conda-forge
```

在联网状态下，安装Conda安装包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/setup.py)中的required_package），其余情况需自行安装依赖。

## 配置环境变量

**如果昇腾AI处理器配套软件包没有安装在默认路径**，安装好MindSpore之后，需要导出Runtime相关的环境变量，以下命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# environment variables
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# set environmet variables using script provided by CANN, swap "ascend-toolkit" with "nnae" if you are using CANN-nnae package instead
source ${LOCAL_ASCEND}/ascend-toolkit/set_env.sh
```

## 验证是否成功安装

**方法一：**

执行以下命令：

```bash
python -c "import mindspore;mindspore.set_device('Ascend');mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
```

说明MindSpore安装成功了。

**方法二：**

执行以下代码：

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device("Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

如果输出：

```text
[[[[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]

  [[2. 2. 2. 2.]
   [2. 2. 2. 2.]
   [2. 2. 2. 2.]]]]
```

说明MindSpore安装成功了。

## 升级MindSpore版本

从MindSpore 1.x升级到MindSpore 2.x版本时，需要先手动卸载旧版本：

```bash
conda remove mindspore-ascend
```

然后安装新版本：

```bash
conda install mindspore -c mindspore -c conda-forge
```

从MindSpore 2.x版本升级时，执行以下命令：

```bash
conda update mindspore -c mindspore -c conda-forge
```

注意：升级MindSpore Ascend版本conda安装包后请重新安装昇腾AI处理器配套软件包提供的whl包。

首先卸载旧版本：

```bash
pip uninstall te topi hccl -y
```

然后重新安装：

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```
