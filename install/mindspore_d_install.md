# 安装MindSpore

本文档介绍如何在Ascend AI处理器的环境上快速安装MindSpore。

<!-- TOC -->

- [安装MindSpore](#安装mindspore)
    - [环境要求](#环境要求)
        - [硬件要求](#硬件要求)
        - [系统要求和软件依赖](#系统要求和软件依赖)
        - [Conda安装（可选）](#conda安装可选)
        - [配套软件包依赖配置](#配套软件包依赖配置)
    - [安装指南](#安装指南)
        - [通过可执行文件安装](#通过可执行文件安装)
        - [从源码编译安装](#从源码编译安装)
    - [配置环境变量](#配置环境变量)
    - [安装验证](#安装验证)
- [安装MindInsight](#安装mindinsight)
- [安装MindArmour](#安装mindarmour)

<!-- /TOC -->

## 环境要求

### 硬件要求

- Ascend 910 AI处理器

  > - 需为每张卡预留至少32G内存。

### 系统要求和软件依赖

| 版本号 | 操作系统 | 可执行文件安装依赖 | 源码编译安装依赖 |
| ---- | :--- | :--- | :--- |
| MindSpore 0.6.0-beta | - Ubuntu 18.04 aarch64 <br> - Ubuntu 18.04 x86_64 <br> - EulerOS 2.8 aarch64 <br> - EulerOS 2.5 x86_64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - Ascend 910 AI处理器配套软件包（对应版本[Atlas Data Center Solution V100R020C10T200](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-data-center-solution-pid-251167910/software/251661816)） <br> - [gmp](https://gmplib.org/download/gmp/) 6.1.2 <br> - 其他依赖项参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r0.6/requirements.txt) | **编译依赖：**<br> - [Python](https://www.python.org/downloads/) 3.7.5 <br> - Ascend 910 AI处理器配套软件包（对应版本[Atlas Data Center Solution V100R020C10T200](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-data-center-solution-pid-251167910/software/251661816)） <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [GCC](https://gcc.gnu.org/releases.html) 7.3.0 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [patch](http://ftp.gnu.org/gnu/patch/) >= 2.5 <br> - [gmp](https://gmplib.org/download/gmp/) 6.1.2 <br> **安装依赖：**<br> 与可执行文件安装依赖相同 |

- 确认当前用户有权限访问Ascend 910 AI处理器配套软件包（对应版本[Atlas Data Center Solution V100R020C10T200](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-data-center-solution-pid-251167910/software/251661816)）的安装路径`/usr/local/Ascend`，若无权限，需要root用户将当前用户添加到`/usr/local/Ascend`所在的用户组，具体配置请详见配套软件包的说明文档。
- GCC 7.3.0可以直接通过apt命令安装。
- 在联网状态下，安装whl包时会自动下载`requirements.txt`中的依赖项，其余情况需自行安装。

### Conda安装（可选）

1. 针对不同的CPU架构，Conda安装包下载路径如下。

   - [X86 Anaconda](https://www.anaconda.com/distribution/) 或 [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - [ARM Anaconda](https://github.com/Archiconda/build-tools/releases/download/0.2.3/Archiconda3-0.2.3-Linux-aarch64.sh)

2. 创建并激活Python环境。

    ```bash
    conda create -n {your_env_name} python=3.7.5
    conda activate {your_env_name}
    ```

> Conda是强大的Python环境管理工具，建议初学者上网查阅更多资料。

### 配套软件包依赖配置

 - 安装Ascend 910 AI处理器配套软件包（对应版本[Atlas Data Center Solution V100R020C10T200](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-data-center-solution-pid-251167910/software/251661816)）提供的whl包，whl包随配套软件包发布，升级配套软件包之后需要重新安装。

    ```bash
    pip install /usr/local/Ascend/fwkacllib/lib64/topi-{version}-py3-none-any.whl
    pip install /usr/local/Ascend/fwkacllib/lib64/te-{version}-py3-none-any.whl
    pip install /usr/local/Ascend/fwkacllib/lib64/hccl-{version}-py3-none-any.whl
    ```

## 安装指南

### 通过可执行文件安装

- 从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，建议先进行SHA-256完整性校验，执行如下命令安装MindSpore。

    ```bash
    pip install mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    ```

### 从源码编译安装

必须在Ascend 910 AI处理器的环境上进行编译安装。

1. 从代码仓下载源码。

    ```bash
    git clone https://gitee.com/mindspore/mindspore.git -b r0.6
    ```

2. 在源码根目录下，执行如下命令编译MindSpore。

    ```bash
    bash build.sh -e ascend
    ```
    > - 在执行上述命令前，需保证可执行文件`cmake`和`patch`所在路径已加入环境变量PATH中。
    > - `build.sh`中会执行`git clone`获取第三方依赖库的代码，请提前确保git的网络设置正确可用。
    > - `build.sh`中默认的编译线程数为8，如果编译机性能较差可能会出现编译错误，可在执行中增加-j{线程数}来减少线程数量。如`bash build.sh -e ascend -j4`。

3. 执行如下命令安装MindSpore。

    ```bash
    chmod +x build/package/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    pip install build/package/mindspore_ascend-{version}-cp37-cp37m-linux_{arch}.whl
    ```

## 配置环境变量

- EulerOS操作系统，安装好MindSpore之后，需要导出Runtime相关环境变量。

    ```bash
    # control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
    export GLOG_v=2

    # Conda environmental options
    LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

    # lib libraries that the run package depends on
    export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/fwkacllib/lib64:${LD_LIBRARY_PATH}

    # Environment variables that must be configured
    export TBE_IMPL_PATH=${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe  # TBE operator implementation tool path
    export PATH=${LOCAL_ASCEND}/fwkacllib/ccec_compiler/bin/:${PATH}       # TBE operator compilation tool path
    export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                       # Python library that TBE implementation depends on
    ```

- Ubuntu操作系统，安装好MindSpore之后，需要导出Runtime相关环境变量，注意：需要将如下配置中{version}替换为环境上真实的版本号。

    ```bash
    # control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
    export GLOG_v=2

    # Conda environmental options
    LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

    # lib libraries that the run package depends on
    export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/ascend-toolkit/{version}/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LD_LIBRARY_PATH}

    # Environment variables that must be configured
    export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/{version}/opp/op_impl/built-in/ai_core/tbe  # TBE operator implementation tool path
    export PATH=${LOCAL_ASCEND}/ascend-toolkit/{version}/fwkacllib/ccec_compiler/bin/:${PATH}       # TBE operator compilation tool path
    export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                # Python library that TBE implementation depends on
    ```

## 安装验证

- 安装并配置好环境变量后，执行如下python脚本：

    ```bash
    import numpy as np
    from mindspore import Tensor
    from mindspore.ops import functional as F
    import mindspore.context as context

    context.set_context(device_target="Ascend")
    x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
    print(F.tensor_add(x, y))
    ```

- 若出现如下结果，即安装验证通过。

    ```
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

# 安装MindInsight

当您需要查看训练过程中的标量、图像、计算图以及模型超参等信息时，可以选装MindInsight。

## 环境要求

### 系统要求和软件依赖

| 版本号 | 操作系统 | 可执行文件安装依赖 | 源码编译安装依赖 |
| ---- | :--- | :--- | :--- |
| MindInsight 0.6.0-beta | - Ubuntu 18.04 aarch64 <br> - Ubuntu 18.04 x86_64 <br> - EulerOS 2.8 aarch64 <br> - EulerOS 2.5 x86_64 <br> | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore 0.6.0-beta <br> - 其他依赖项参见[requirements.txt](https://gitee.com/mindspore/mindinsight/blob/r0.6/requirements.txt) | **编译依赖：**<br> - [Python](https://www.python.org/downloads/) 3.7.5 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [GCC](https://gcc.gnu.org/releases.html) 7.3.0 <br> - [node.js](https://nodejs.org/en/download/) >= 10.19.0 <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [pybind11](https://pypi.org/project/pybind11/) >= 2.4.3 <br> **安装依赖：**<br> 与可执行文件安装依赖相同 |

- 在联网状态下，安装whl包时会自动下载`requirements.txt`中的依赖项，其余情况需自行安装。

## 安装指南

### 通过可执行文件安装

1. 从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，建议先进行SHA-256完整性校验，执行如下命令安装MindInsight。

    ```bash
    pip install mindinsight-{version}-cp37-cp37m-linux_{arch}.whl
    ```

2. 执行如下命令，如果提示`web address: http://127.0.0.1:8080`，则说明安装成功。

    ```bash
    mindinsight start
    ```

### 从源码编译安装

1. 从代码仓下载源码。

    ```bash
    git clone https://gitee.com/mindspore/mindinsight.git -b r0.6
    ```
    > **不能**直接在仓库主页下载zip包获取源码。

2. 可选择以下任意一种安装方式：

   (1) 进入源码的根目录，执行安装命令。

      ```bash
      cd mindinsight
      pip install -r requirements.txt
      python setup.py install
      ```

   (2) 构建whl包进行安装。

      进入源码的根目录，先执行`build`目录下的MindInsight编译脚本，再执行命令安装`output`目录下生成的whl包。

      ```bash
      cd mindinsight
      bash build/build.sh
      pip install output/mindinsight-{version}-cp37-cp37m-linux_{arch}.whl
      ```

3. 执行如下命令，如果提示`web address: http://127.0.0.1:8080`，则说明安装成功。

    ```bash
    mindinsight start
    ```

# 安装MindArmour

当您进行AI模型安全研究或想要增强AI应用模型的防护能力时，可以选装MindArmour。

## 环境要求

### 系统要求和软件依赖

| 版本号 | 操作系统 | 可执行文件安装依赖 | 源码编译安装依赖 |
| ---- | :--- | :--- | :--- |
| MindArmour 0.6.0-beta | - Ubuntu 18.04 aarch64 <br> - Ubuntu 18.04 x86_64 <br> - EulerOS 2.8 aarch64 <br> - EulerOS 2.5 x86_64 <br> | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore 0.6.0-beta <br> - 其他依赖项参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/r0.6/setup.py) | 与可执行文件安装依赖相同 |

- 在联网状态下，安装whl包时会自动下载`setup.py`中的依赖项，其余情况需自行安装。

## 安装指南

### 通过可执行文件安装

1. 从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，建议先进行SHA-256完整性校验，执行如下命令安装MindArmour。

   ```bash
   pip install mindarmour-{version}-cp37-cp37m-linux_{arch}.whl
   ```

2. 执行如下命令，如果没有提示`No module named 'mindarmour'`等加载错误的信息，则说明安装成功。

   ```bash
   python -c 'import mindarmour'
   ```

### 从源码编译安装

1. 从代码仓下载源码。

   ```bash
   git clone https://gitee.com/mindspore/mindarmour.git -b r0.6
   ```

2. 在源码根目录下，执行如下命令编译并安装MindArmour。

   ```bash
   cd mindarmour
   python setup.py install
   ```


3. 执行如下命令，如果没有提示`No module named 'mindarmour'`等加载错误的信息，则说明安装成功。

   ```bash
   python -c 'import mindarmour'
   ```
