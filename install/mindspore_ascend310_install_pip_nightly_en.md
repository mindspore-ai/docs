# Installing MindSpore Nightly in Ascend 310 by pip

<!-- TOC -->

- [Installing MindSpore Nightly in Ascend 310 by pip](#installing-mindspore-200-nightly-in-ascend-310-by-pip)
    - [Installing MindSpore and dependencies](#installing-mindspore-and-dependencies)
        - [Installing Python](#installing-python)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing GCC](#installing-gcc)
        - [Installing CMake](#installing-cmake)
        - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Verifying the Installation](#verifying-the-installation)

<!-- /TOC -->

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend310_install_pip_en.md)

MindSpore Nightly is a preview version which includes latest features and bugfixes, not fully supported and tested. Install MindSpore Nightly version if you wish to try out the latest features or bug fixes can use this version.

The following describes how to install MindSpore by pip on Linux in the Ascend 310 environment, MindSpore in Ascend 310 only supports inference.

## Installing MindSpore and dependencies

The following table lists the system environment and third-party dependencies required for installing MindSpore.

|Software|Version|Description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8|-|OS for running and compiling MindSpore|
|[Python](#installing-python)|3.7-3.9|Python environment that MindSpore depends|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|
|[CMake](#installing-cmake)|3.18.3 or later| Compilation tool that builds MindSpore                  |

The following describes how to install the third-party dependencies.

### Installing Python

[Python](https://www.python.org/) can be installed by Conda.

Install Miniconda:

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

After the installation is complete, you can set up Tsinghua source acceleration download for Conda, and see [here](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/).

Create a virtual environment, taking Python 3.7.5 as an example:

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

Run the following command to check the Python version.

```bash
python --version
```

If you are using ARM architecture, ensure that pip installed for current Python is 19.3 or later. If not, upgrade pip using the following command.

```bash
python -m pip install -U pip
```

### Installing Ascend AI processor software package

Ascend software package provides two distributions, commercial edition and community edition:

- Commercial edition needs approval from Ascend to download, for detailed installation guide, please refer to [Ascend Data Center Solution 23.0.RC2 Installation Guide].

- Community edition has no restrictions, choose `6.3.RC2.alpha005` in [CANN community edition](https://www.hiascend.com/software/cann/community-history), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers?tag=community). Please refer to the abovementioned commercial edition installation guide to choose which packages are to be installed and how to install them.

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

Install the .whl packages provided in Ascend AI processor software package. If the .whl packages have been installed before, you should uninstall the .whl packages by running the following command.

```bash
pip uninstall te topi hccl -y
```

Run the following command to install the .whl packages if the Ascend AI package has been installed in default path. If the installation path is not the default path, you need to replace the path in the command with the installation path.

```bash
pip install sympy
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-*-py3-none-any.whl
```

The LD_LIBRARY_PATH environment variable does not work when the installation package exists in the default path. The default path priority is: /usr/local/Ascend/nnae is higher than /usr/local/Ascend/ascend-toolkit. The reason is that MindSpore uses DT_RPATH to support startup without environment variables, reducing user settings. DT_RPATH has a higher priority than the LD_LIBRARY_PATH environment variable.

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

### Installing CMake

- On Ubuntu 18.04, run the following commands to install [CMake](https://cmake.org/).

    ```bash
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt-get install cmake -y
    ```

- Other Linux systems can be installed with the following commands.

    Choose different download links based on the system architecture.

    ```bash
    # x86 run
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-x86_64.sh
    # aarch64 run
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-aarch64.sh
    ```

    Run the script to install CMake, which is installed in the `/usr/local` by default.

    ```bash
    sudo mkdir /usr/local/cmake-3.19.8
    sudo bash cmake-3.19.8-Linux-*.sh --prefix=/usr/local/cmake-3.19.8 --exclude-subdir
    ```

    Finally, add CMake to the `PATH` environment variable. Run the following commands if it is installed in the default path, and other installation path need to be modified accordingly.

    ```bash
    echo -e "export PATH=/usr/local/cmake-3.19.8/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
    ```

### Installing MindSpore

Execute the following command to install MindSpore:

```bash
pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Of which,

- When the network is connected, dependencies are automatically downloaded during .whl package installation. (For details about the dependencies, see required_package in [setup.py](https://gitee.com/mindspore/mindspore/blob/master/setup.py).) In other cases, you need to install it by yourself. When running models, you need to install additional dependencies based on requirements.txt specified for different models in [ModelZoo](https://gitee.com/mindspore/models/tree/master/). For details about common dependencies, see [requirements.txt](https://gitee.com/mindspore/mindspore/blob/master/requirements.txt).
- pip will be installing the latest version of MindSpore Nightly automatically. If you wish to specify the version to be installed, please refer to the instruction below regarding to version update, and specify version manually.

## Configuring Environment Variables

After MindSpore is installed, export runtime environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package. Change it to the actual installation path.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# lib libraries that the mindspore depends on, modify "pip3" according to the actual situation
export LD_LIBRARY_PATH=`pip3 show mindspore | grep Location | awk '{print $2"/mindspore/lib"}' | xargs realpath`:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}
```

## Verifying the Installation

Create a directory to store the sample code project, for example, `/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_single_op_sample`. You can obtain the code from the [official website](https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/sample_resources/ascend310_single_op_sample.zip). A simple example of adding `[1, 2, 3, 4]` to `[2, 3, 4, 5]` is used and the code project directory structure is as follows:

```text

└─ascend310_single_op_sample
    ├── CMakeLists.txt                    // Build script
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
cmake . -DMINDSPORE_PATH=`pip3 show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`
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

When upgrading from MindSpore 1.x to MindSpore 2.x, you need to manually uninstall the old version first:

```bash
pip uninstall mindspore-ascend-dev
```

Then install MindSpore 2.x:

```bash
pip install mindspore-dev=={version}
```

When upgrading from MindSpore 2.x:

```bash
pip install --upgrade mindspore-dev=={version}
```

Of which,

- When updating to a release candidate (RC) version, set `{version}` to the RC version number, for example, 2.0.0.rc1. When updating to a stable release, you can remove `=={version}`.
