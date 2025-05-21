# pip方式安装MindSpore Ascend版本

<!-- TOC -->

- [pip方式安装MindSpore Ascend版本](#pip方式安装mindspore-ascend版本)
    - [安装MindSpore与依赖软件](#安装mindspore与依赖软件)
        - [安装Python](#安装python)
        - [安装昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)
        - [安装GCC](#安装gcc)
        - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)

<!-- /TOC -->

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/install/mindspore_ascend_install_pip.md)

本文档介绍如何在Ascend环境的Linux系统上，使用pip方式快速安装MindSpore。

## 安装MindSpore与依赖软件

下表列出了安装MindSpore所需的系统环境和第三方依赖。

|软件名称|版本|作用|
|-|-|-|
|Ubuntu 18.04 / CentOS 7.6 / EulerOS 2.8 / openEuler 20.03 / KylinV10 SP1|-|编译和运行MindSpore的操作系统|
|[Python](#安装python)|3.9-3.11|MindSpore的使用依赖Python环境|
|[昇腾AI处理器配套软件包](#安装昇腾ai处理器配套软件包)|-|MindSpore使用的Ascend平台AI计算库|
|[GCC](#安装gcc)|7.3.0|用于编译MindSpore的C++编译器|

下面给出第三方依赖的安装方法。

### 安装Python

[Python](https://www.python.org/)可通过Conda进行安装。

安装Miniconda：

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

安装完成后，可以为Conda设置清华源加速下载，参考[此处](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)。

创建虚拟环境，以Python 3.9.11为例：

```bash
conda create -n mindspore_py39 python=3.9.11 -y
conda activate mindspore_py39
```

可以通过以下命令查看Python版本。

```bash
python --version
```

如果您的环境为ARM架构，请确认当前使用的Python配套的pip版本>=19.3。可以使用以下命令升级pip。

```bash
python -m pip install -U pip
```

### 安装昇腾AI处理器配套软件包

昇腾软件包提供商用版和社区版两种下载途径：

- 商用版下载需要申请权限，优选版本的下载链接与安装方式请参考[Ascend Training Solution 25.0.RC1 安装指引文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100472026)。

- 社区版下载不受限制，下载链接请前往[CANN社区版](https://www.hiascend.com/developer/download/community/result?module=cann)，推荐优先选择`8.1.RC1.beta1`版本，以及在[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)链接中获取对应的固件和驱动安装包，安装包的选择与安装方式请参照上述的商用版安装指引文档。

安装包默认安装路径为`/usr/local/Ascend`。安装后确认当前用户有权限访问昇腾AI处理器配套软件包的安装路径，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。

安装昇腾AI处理器配套软件所包含的whl包。如果之前已经安装过昇腾AI处理器配套软件包，需要先使用以下命令卸载对应的whl包。

```bash
pip uninstall te topi hccl -y
```

默认安装路径使用以下命令安装。如果安装路径不是默认路径，需要将命令中的路径替换为安装路径。

```bash
pip install sympy
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```

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

### 安装MindSpore

首先参考[版本列表](https://www.mindspore.cn/versions)，选择想要安装的MindSpore版本，并进行SHA-256完整性校验。以2.6.0版本为例，执行以下命令。

```bash
export MS_VERSION=2.6.0
```

然后根据系统架构及Python版本，执行以下命令安装MindSpore。

```bash
# x86_64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/x86_64/mindspore-${MS_VERSION/-/}-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.10
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/x86_64/mindspore-${MS_VERSION/-/}-cp310-cp310-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# x86_64 + Python3.11
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/x86_64/mindspore-${MS_VERSION/-/}-cp311-cp311-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.9
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp39-cp39-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.10
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp310-cp310-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
# aarch64 + Python3.11
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/aarch64/mindspore-${MS_VERSION/-/}-cp311-cp311-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[setup.py](https://gitee.com/mindspore/mindspore/blob/v2.6.0/setup.py)中的required_package），其余情况需自行安装依赖。

## 配置环境变量

**如果昇腾AI处理器配套软件包没有安装在默认路径**，安装好MindSpore之后，需要导出Runtime相关环境变量，以下命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

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
pip uninstall mindspore-ascend
```

然后安装新版本：

```bash
pip install mindspore=={version}
```

从MindSpore 2.x版本升级时，执行以下命令：

```bash
pip install --upgrade mindspore=={version}
```

其中：

- 升级到rc版本时，需要手动指定`{version}`为rc版本号，例如1.6.0rc1；如果升级到正式版本，`=={version}`字段可以缺省。
