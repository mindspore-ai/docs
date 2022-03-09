# Installing MindSpore in Ascend 310 by Source Code

<!-- TOC -->

- [Installing MindSpore in Ascend 310 by Source Code Compilation](#installing-mindspore-in-ascend-310-by-source-code-compilation)
    - [Environment Preparation](#environment-preparation)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing Python](#installing-python)
        - [Installing GCC](#installing-gcc)
        - [Installing git, gmp, tclsh, patch and Flex](#installing-git-gmp-tclsh-patch-and-flex)
        - [Installing CMake](#installing-cmake)
    - [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository)
    - [Building MindSpore](#building-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Verifying the Installation](#verifying-the-installation)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/install/mindspore_ascend310_install_source_en.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

The following describes how to quickly install MindSpore by compiling the source code on Linux in the Ascend 310 environment, MindSpore in Ascend 310 only supports inference.

## Environment Preparation

The following table lists the system environment and third-party dependencies required for building and installing MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8|-|OS for compiling and running MindSpore|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[Python](#installing-python)|3.7.5 or 3.9.0|Python environment that MindSpore depends on|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|
|[git](#installing-git-gmp-tclsh-patch-and-flex)|-|Source code management tools used by MindSpore|
|[CMake](#installing-cmake)|3.18.3 or later|Build tools for MindSpore|
|[gmp](#installing-git-gmp-tclsh-patch-and-flex)|6.1.2|Multiple precision arithmetic library used by MindSpore|
|[Flex](#installing-git-gmp-tclsh-patch-and-flex)|2.5.35 or later|lexical analyzer used by MindSpore|
|[tclsh](#installing-git-gmp-tclsh-patch-and-flex)|-|MindSpore SQLite build dependency|
|[patch](#installing-git-gmp-tclsh-patch-and-flex)|2.5 or later|Source code patching tool used by MindSpore|

The following describes how to install the third-party dependencies.

### Installing Ascend AI processor software package

For detailed installation guide, please refer to [Ascend Software Installation Guide](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-data-center-solution-pid-251167910?category=installation-upgrade&subcategory=software-deployment-guide).

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path `/usr/local/Ascend` of Ascend AI processor software package, If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

Install the .whl packages provided in Ascend AI processor software package. The .whl packages are released with the software package. If the .whl packages have been installed before, you need to uninstall the packages by the following command.

```bash
pip uninstall te topi hccl -y
```

Run the following command to install the .whl packages if the Ascend AI package has been installed in default path. If the installation path is not the default path, you need to replace the path in the command with the installation path.

```bash
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/topi-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/te-*-py3-none-any.whl
pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/hccl-*-py3-none-any.whl
```

### Installing Python

[Python](https://www.python.org/) can be installed by Conda.

Install Miniconda:

```bash
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh
bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```

Create a Python 3.7.5 environment:

```bash
conda create -n mindspore_py37 python=3.7.5 -y
conda activate mindspore_py37
```

Or create a Python 3.9.0 environment:

```bash
conda create -n mindspore_py39 python=3.9.0 -y
conda activate mindspore_py39
```

Run the following command to check the Python version.

```bash
python --version
```

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

### Installing git, gmp, tclsh, patch and Flex

- On Ubuntu 18.04, run the following commands to install.

    ```bash
    sudo apt-get install git libgmp-dev tcl patch flex -y
    ```

- On CentOS 7 and EulerOS, run the following commands to install.

    ```bash
    sudo yum install git gmp-devel tcl patch flex -y
    ```

### Installing CMake

- On Ubuntu 18.04, run the following commands to install [CMake](https://cmake.org/).

    ```bash
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
    sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
    sudo apt-get install cmake -y
    ```

- Other Linux systems can be installed with the following commands.

    Choose one download link based on the system architecture.

    ```bash
    # x86 run
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-x86_64.sh
    # aarch64 run
    curl -O https://cmake.org/files/v3.19/cmake-3.19.8-Linux-aarch64.sh
    ```

    run the script to install CMake, which is installed in the `/usr/local` by default.

    ```bash
    sudo mkdir /usr/local/cmake-3.19.8
    sudo bash cmake-3.19.8-Linux-*.sh --prefix=/usr/local/cmake-3.19.8 --exclude-subdir
    ```

    Finally, add CMake to the `PATH` environment variable. Run the following commands if it is installed in the default path, other installation path need to be modified accordingly.

    ```bash
    echo -e "export PATH=/usr/local/cmake-3.19.8/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
    ```

## Downloading the Source Code from the Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git
```

## Building MindSpore

Go to the root directory of mindspore, then run the build script.

```bash
cd mindspore
bash build.sh -e ascend -V 310 -S on
```

Where:

- The default number of build threads is 8 in `build.sh`. If the compiler performance is poor, build errors may occur. You can add -j{Number of threads} to script to reduce the number of threads. For example, `bash build.sh -e ascend -V 310 -j4`.
- By default, the dependent source code is downloaded from GitHub. When `-S` is set to `on`, the source code is downloaded from the corresponding Gitee image.
- For details about how to use `build.sh`, see the script header description.

## Installing MindSpore

```bash
tar -zxf output/mindspore_ascend-*.tar.gz
```

## Configuring Environment Variables

After MindSpore is installed, export runtime environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package. Change it to the actual installation path.

```bash
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
## TBE operator implementation tool path
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
## OPP path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}

# Set path to extracted MindSpore accordingly
export LD_LIBRARY_PATH={mindspore_path}:${LD_LIBRARY_PATH}
```

Where:

- `{mindspore_path}` specifies the absolute path to which MindSpore package is extracted.

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

Build a project by referring to `README.md`, modify`{mindspore_path}`which specifies the absolute path to MindSpore.

```bash
cmake . -DMINDSPORE_PATH={mindspore_path}
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
