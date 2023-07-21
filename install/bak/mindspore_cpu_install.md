# 安装MindSpore

本文档介绍如何在CPU环境的Ubuntu系统上快速安装MindSpore。

<!-- TOC -->

- [安装MindSpore](#安装mindspore)
    - [环境要求](#环境要求)
        - [系统要求和软件依赖](#系统要求和软件依赖)
        - [Conda安装（可选）](#conda安装可选)
    - [安装指南](#安装指南)
        - [通过可执行文件安装](#通过可执行文件安装)
        - [从源码编译安装](#从源码编译安装)
        - [版本升级](#版本升级)
- [安装MindArmour](#安装mindarmour)
- [安装MindSpore Hub](#安装mindspore-hub)

<!-- /TOC -->

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/install/mindspore_cpu_install.md)

本文档介绍如何在CPU环境的Ubuntu系统上快速安装MindSpore。

## 环境要求

### 系统要求和软件依赖

| 版本号 | 操作系统 | 可执行文件安装依赖 | 源码编译安装依赖 |
| ---- | :--- | :--- | :--- |
| MindSpore 1.0.1 | - Ubuntu 18.04 x86_64 <br> - Ubuntu 18.04 aarch64 | - [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5 <br> - 其他依赖项参见[requirements.txt](https://gitee.com/mindspore/mindspore/blob/r1.0/requirements.txt) | **编译依赖：**<br> - [Python](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) 3.7.5 <br> - [wheel](https://pypi.org/project/wheel/) >= 0.32.0 <br> - [GCC](http://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz) 7.3.0 <br> - [CMake](https://cmake.org/download/) >= 3.14.1 <br> - [patch](http://ftp.gnu.org/gnu/patch/) >= 2.5 <br> **安装依赖：**<br> 与可执行文件安装依赖相同 |

- GCC 7.3.0可以直接通过apt命令安装。
- 若环境中已经安装了Python，确保将Python添加到环境变量中，还可以通过命令`python --version`查看Python的版本是否符合要求。
- 请将pip添加到环境变量中，以保证可以通过pip直接安装Python相关的工具包。如果pip没有在当前环境中安装，可以在 `https://pypi.org/project/pip/` 中进行下载安装。
- 在联网状态下，安装whl包时会自动下载`requirements.txt`中的依赖项，其余情况需自行安装。

### Conda安装（可选）

1. Conda安装包下载路径如下。

   - [X86 Anaconda](https://www.anaconda.com/distribution/) 或 [X86 Miniconda](https://docs.conda.io/en/latest/miniconda.html)

2. 创建并激活Python环境。

    ```bash
    conda create -n {your_env_name} python=3.7.5
    conda activate {your_env_name}
    ```

> Conda是强大的Python环境管理工具，建议初学者上网查阅更多资料。

## 安装指南

### 通过可执行文件安装

1. 从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，建议先进行SHA-256完整性校验，执行如下命令安装MindSpore。

    ```bash
    pip install mindspore-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
    ```

2. 执行如下命令，如果没有提示`No module named 'mindspore'`等加载错误的信息，则说明安装成功。

    ```bash
    python -c 'import mindspore'
    ```

### 从源码编译安装

1. 从代码仓下载源码。

    ```bash
    git clone https://gitee.com/mindspore/mindspore.git -b r1.0
    ```

2. 在源码根目录下执行如下命令编译MindSpore。

    ```bash
    bash build.sh -e cpu -j4
    ```
    >
    > - 在执行上述命令前，需保证可执行文件cmake和patch所在路径已加入环境变量PATH中。
    > - 在执行上述命令前，需保证已经安装了[OpenSSL](https://github.com/openssl/openssl.git)，并设置环境变量`export OPENSSL_ROOT_DIR=“OpenSSL安装目录”`。
    > - `build.sh`中会执行`git clone`获取第三方依赖库的代码，请提前确保git的网络设置正确可用。
    > - 如果编译机性能较好，可在执行中增加-j{线程数}来增加线程数量。如`bash build.sh -e cpu -j12`。

3. 执行如下命令安装MindSpore。

    ```bash
    chmod +x build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    pip install build/package/mindspore-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
    ```

4. 执行如下命令，如果没有提示`No module named 'mindspore'`等加载错误的信息，则说明安装成功。

    ```bash
    python -c 'import mindspore'
    ```

### 版本升级

当您需要升级MindSpore版本时，可用如下命令进行版本升级。

- 直接在线升级

    ```bash
    pip install --upgrade mindspore
    ```

- 本地源码编译升级

    在源码根目录下执行编译脚本`build.sh`成功后，在`build/package`目录下找到编译生成的whl安装包，然后执行命令进行升级。

    ```bash
    pip install --upgrade mindspore-{version}-cp37-cp37m-linux_{arch}.whl
    ```

# 安装MindArmour

当您进行AI模型安全研究或想要增强AI应用模型的防护能力时，可以选装MindArmour。

## 环境要求

### 系统要求和软件依赖

| 版本号                 | 操作系统            | 可执行文件安装依赖                                           | 源码编译安装依赖         |
| ---------------------- | :------------------ | :----------------------------------------------------------- | :----------------------- |
| MindArmour 1.0.1 | - Ubuntu 18.04 x86_64 <br> - Ubuntu 18.04 aarch64 | - [Python](https://www.python.org/downloads/) 3.7.5 <br> - MindSpore 1.0.1<br> - 其他依赖项参见[setup.py](https://gitee.com/mindspore/mindarmour/blob/r1.0/setup.py) | 与可执行文件安装依赖相同 |

- 在联网状态下，安装whl包时会自动下载`setup.py`中的依赖项，其余情况需自行安装。

## 安装指南

### 通过可执行文件安装

1. 从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，建议先进行SHA-256完整性校验，执行如下命令安装MindArmour。

   ```bash
   pip install mindarmour-{version}-cp37-cp37m-linux_{arch}.whl -i https://mirrors.huaweicloud.com/repository/pypi/simple
   ```

2. 执行如下命令，如果没有提示`No module named 'mindarmour'`等加载错误的信息，则说明安装成功。

   ```bash
   python -c 'import mindarmour'
   ```

### 从源码编译安装

1. 从代码仓下载源码。

   ```bash
   git clone https://gitee.com/mindspore/mindarmour.git -b r1.0
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

# 安装MindSpore Hub

当您想要快速体验MindSpore预训练模型时，可以选装MindSpore Hub。

具体安装步骤参见[MindSpore Hub](https://gitee.com/mindspore/hub/blob/master/README_CN.md)。
