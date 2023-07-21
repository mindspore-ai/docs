# pip方式安装MindSpore Ascend 910版本

<!-- TOC -->

- [pip方式安装MindSpore Ascend 910版本](#pip方式安装mindspore-ascend-910版本)
    - [确认系统环境信息](#确认系统环境信息)
    - [安装MindSpore](#安装mindspore)
    - [配置环境变量](#配置环境变量)
    - [验证是否成功安装](#验证是否成功安装)
    - [升级MindSpore版本](#升级mindspore版本)
    - [安装MindInsight](#安装mindinsight)
    - [安装MindArmour](#安装mindarmour)
    - [安装MindSpore Hub](#安装mindspore-hub)
    - [安装MindSpore Serving](#安装mindspore-serving)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.2/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/install/mindspore_ascend_install_pip.md)

本文档介绍如何在Ascend 910环境的Linux系统上，使用pip方式快速安装MindSpore。

## 确认系统环境信息

- 确认安装Ubuntu 18.04/CentOS 7.6/EulerOS 2.8/KylinV10 SP1是64位操作系统。
- 确认安装[GCC 7.3.0版本](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)。
- 确认安装[gmp 6.1.2版本](https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz)。
- 确认安装Python 3.7.5版本。
    - 如果未安装或者已安装其他版本的Python，可从[官网](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)或者[华为云](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)下载Python 3.7.5版本 64位，进行安装。
- 确认安装Ascend 910 AI处理器配套软件包（[Ascend Data Center Solution 21.0.1.SPC001](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910/software/253384063?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C252309113%7C251167910)）。
    - 软件包安装方式请参考[产品文档](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910)。
    - 配套软件包包括驱动/固件和CANN。
        - [驱动和固件A800-9000 1.0.10 ARM平台](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9000-pid-250702818/software/252727249?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702818) 或 [驱动和固件A800-9010 1.0.10 x86平台](https://support.huawei.com/enterprise/zh/ascend-computing/a800-9010-pid-250702809/software/252727265?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C250702809)
        - [CANN 5.0.1.SPC102](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software/253348209?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251168373)
    - 确认当前用户有权限访问Ascend 910 AI处理器配套软件包的安装路径`/usr/local/Ascend`，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组。
    - 安装Ascend 910 AI处理器配套软件包提供的whl包，whl包随配套软件包发布，参考如下命令完成安装。

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
        ```

    - 如果升级了Ascend 910 AI处理器配套软件包，配套的whl包也需要重新安装，先将原来的安装包卸载，再参考上述命令重新安装。

        ```bash
        pip uninstall te topi hccl -y
        ```

## 安装MindSpore

参考[版本列表](https://www.mindspore.cn/versions)先进行SHA-256完整性校验，校验一致后再执行如下命令安装MindSpore。

```bash
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}/MindSpore/ascend/{system}/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其中：

- 在联网状态下，安装whl包时会自动下载MindSpore安装包的依赖项（依赖项详情参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.2/requirements.txt)），其余情况需自行安装。  
- `{version}`表示MindSpore版本号，例如安装1.1.0版本MindSpore时，`{version}`应写为1.1.0。  
- `{arch}`表示系统架构，例如使用的Linux系统是x86架构64位时，`{arch}`应写为`x86_64`。如果系统是ARM架构64位，则写为`aarch64`。  
- `{system}`表示系统版本，例如使用的欧拉系统ARM架构，`{system}`应写为`euleros_aarch64`，目前Ascend版本可支持以下系统`euleros_aarch64`/`centos_aarch64`/`centos_x86`/`ubuntu_aarch64`/`ubuntu_x86`/`kylin_aarch64`。

## 配置环境变量

**如果Ascend 910 AI处理器配套软件包没有安装在默认路径**，安装好MindSpore之后，需要导出Runtime相关环境变量，下述命令中`LOCAL_ASCEND=/usr/local/Ascend`的`/usr/local/Ascend`表示配套软件包的安装路径，需注意将其改为配套软件包的实际安装路径。

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}                 # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                # Python library that TBE implementation depends on
```

## 验证是否成功安装

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x, y))
```

如果输出：

```text
[[[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]],

    [[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]],

    [[ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.],
    [ 2.  2.  2.  2.]]]
```

说明MindSpore安装成功了。

## 升级MindSpore版本

当需要升级MindSpore版本时，可执行如下命令：

```bash
pip install --upgrade mindspore-ascend
```

## 安装MindInsight

当您需要查看训练过程中的标量、图像、计算图以及模型超参等信息时，可以选装MindInsight。

具体安装步骤参见[MindInsight](https://gitee.com/mindspore/mindinsight/blob/r1.2/README_CN.md)。

## 安装MindArmour

当您进行AI模型安全研究或想要增强AI应用模型的防护能力时，可以选装MindArmour。

具体安装步骤参见[MindArmour](https://gitee.com/mindspore/mindarmour/blob/r1.2/README_CN.md)。

## 安装MindSpore Hub

当您想要快速体验MindSpore预训练模型时，可以选装MindSpore Hub。

具体安装步骤参见[MindSpore Hub](https://gitee.com/mindspore/hub/blob/r1.2/README_CN.md)。

## 安装MindSpore Serving

当您想要快速体验MindSpore在线推理服务时，可以选装MindSpore Serving。

具体安装步骤参见[MindSpore Serving](https://gitee.com/mindspore/serving/blob/r1.2/README_CN.md)。
