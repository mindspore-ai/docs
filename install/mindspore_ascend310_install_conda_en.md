# Installing MindSpore Ascend 310 by Conda

<!-- TOC -->

- [Installing MindSpore Ascend 310 by Conda](#installing-mindspore-ascend-310-by-conda)
    - [Automatic Installation](#automatic-installation)
    - [Manual Installation](#manual-installation)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing Conda](#installing-conda)
        - [Installing GCC](#installing-gcc)
        - [Installing gmp](#installing-gmp)
        - [Creating and Accessing the Conda Virtual Environment](#creating-and-accessing-the-conda-virtual-environment)
        - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Verifying the Installation](#verifying-the-installation)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend310_install_conda_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

[Conda](https://docs.conda.io/en/latest/) is an open-source, cross-platform, language-agnostic package manager and environment management system. It allows users to easily install different versions of binary software packages and any required libraries appropriate for their computing platform.

The following describes how to quickly install MindSpore by Conda on Linux in the Ascend 310 environment, MindSpore in Ascend 310 only supports inference.

- If you want to install MindSpore by Conda on an EulerOS 2.8 with Ascend AI processor software package installed, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/master/scripts/install/euleros-ascend-conda.sh) for one-click installation, see [Automatic Installation](#automatic-installation) section. The automatic installation script will install MindSpore and its required dependencies.

- If some dependencies, such as Python and GCC, have been installed in your system, it is recommended to install manually by referring to the installation steps in the [Manual Installation](#manual-installation) section.

## Automatic Installation

Before running the automatic installation script, you need to make sure that the Ascend AI processor software package is correctly installed on your system. If it is not installed, please refer to the section [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package) to install it.

Run the following command to obtain and run the automatic installation script. The automatic installation script only supports the installation of MindSpore>=1.6.0.

```bash
wget https://gitee.com/mindspore/mindspore/raw/master/scripts/install/euleros-ascend-conda.sh
# install Python 3.7 and the latest MindSpore by default
# the default value of LOCAL_ASCEND is /usr/local/Ascend
bash -i ./euleros-ascend-conda.sh
# to specify Python and MindSpore version, taking Python 3.9 and MindSpore 1.6.0 as examples
# and set LOCAL_ASCEND to /home/xxx/Ascend, use the following manners
# LOCAL_ASCEND=/home/xxx/Ascend PYTHON_VERSION=3.9 MINDSPORE_VERSION=1.6.0 bash -i ./euleros-ascend-conda.sh
```

This script performs the following operations:

- Install the dependencies required by MindSpore, such as GCC and gmp.
- Install Conda and create a virtual environment for MindSpore.
- Install MindSpore Ascend 310 by Conda.

After the script is executed, you need to reopen the terminal window, and then set the relevant environment variables according to the instructions in [Configuring Environment Variables](#configuring-environment-variables).

The automatic installation script creates a virtual environment named `mindspore_pyXX` for MindSpore. Where `XX` is the Python version, such as Python 3.7, the virtual environment name is `mindspore_py37`. Run the following command to show all virtual environments.

```bash
conda env list
```

To activate the virtual environment, take Python 3.7 as an example, execute the following command.

```bash
conda activate mindspore_py37
```

For more usage, see the script header description.

## Manual Installation

The following table lists the system environment and third-party dependencies required for installing MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8|-|OS for compiling and running MindSpore|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[Conda](#installing-conda)|Anaconda3 or Miniconda3|Python environment management tool|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|
|[gmp](#installing-gmp)|6.1.2|Multiple precision arithmetic library used by MindSpore|

The following describes how to install the third-party dependencies.

### Installing Ascend AI processor software package

For detailed installation guide, please refer to [Ascend Data Center Solution 21.0.4 Installation Guide](https://support.huawei.com/enterprise/zh/doc/EDOC1100235797?section=j003).

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

### Installing Conda

Run the following command to install Miniconda.

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

### Installing GCC

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install gcc-7 -y
    ```

- On CentOS 7, run the following commands to install.

    ```bash
    sudo yum install centos-release-scl
    sudo yum install devtoolset-7
    ```

    After installation, run the following commands to switch to GCC 7.

    ```bash
    scl enable devtoolset-7 bash
    ```

- On EulerOS, run the following commands to install.

    ```bash
    sudo yum install gcc -y
    ```

### Installing gmp

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install libgmp-dev -y
    ```

- On CentOS 7 and EulerOS, run the following commands to install.

    ```bash
    sudo yum install gmp-devel -y
    ```

### Creating and Accessing the Conda Virtual Environment

Create a Conda virtual environment based on the Python version you want to use and go to the virtual environment.

If you want to use Python 3.7.5:

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

If you wish to use another version of Python, just change the Python version in the above command. Python 3.7, Python 3.8 and Python 3.9 are currently supported.

Install the .whl package provided with the Ascend AI Processor software package in the virtual environment. The .whl package is released with the software package. After the software package is upgraded, you need to reinstall the .whl package.

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*-py3-none-any.whl
```

If the software package of the Ascend AI processor is upgraded, the matching .whl package also needs to be reinstalled. Uninstall the original installation package and reinstall the .whl package by referring to the preceding commands.

```bash
pip uninstall te topi hccl -y
```

### Installing MindSpore

Ensure that you are in the Conda virtual environment and run the following command to install the latest MindSpore. To install other versions, please refer to the specified version number of [Version List](https://www.mindspore.cn/versions) after `mindspore-ascend=`.

```bash
conda install mindspore-ascend -c mindspore -c conda-forge
```

When the network is connected, dependencies of MindSpore are automatically downloaded during the .whl package installation. (For details about the dependency, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py) .) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).

## Configuring Environment Variables

After MindSpore is installed, export Runtime-related environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package, and you need to replace it as the actual installation path of the software package.

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
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

## Verifying the Installation

Create a directory to store the sample code project, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample`. You can obtain the code from the [official website](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/sample_resources/ascend310_single_op_sample.zip). A simple example of adding `[1, 2, 3, 4]` to `[2, 3, 4, 5]` is used and the code project directory structure is as follows:

```text

└─ascend310_single_op_sample
    ├── CMakeLists.txt                    // Compile script
    ├── README.md                         // Usage description
    ├── main.cc                           // Main function
    └── tensor_add.mindir                 // MindIR model file
```

Go to the directory of the sample project and change the path based on the actual requirements.

```bash
cd /home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample
```

Build a project by referring to `README.md`, and modify `pip3` according to the actual situation.

```bash
cmake . -DMINDSPORE_PATH=`pip3 show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
make
```

After the build is successful, execute the case.

```bash
./tensor_add_sample
```

The following information is displayed:

```text
3
5
7
9
```

The preceding information indicates that MindSpore is successfully installed.

## Version Update

Use the following command if you need to update the MindSpore version:

```bash
conda update mindspore-ascend -c mindspore -c conda-forge
```
