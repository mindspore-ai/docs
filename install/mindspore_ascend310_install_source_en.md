# Installing MindSpore in Ascend 310 by Source Code

<!-- TOC -->

- [Installing MindSpore in Ascend 310 by Source Code](#installing-mindspore-in-ascend-310-by-source-code)
    - [Environment Preparation-automatic, recommended](#environment-preparation-automatic-recommended)
    - [Environment Preparation-manual](#environment-preparation-manual)
        - [Installing Python](#installing-python)
        - [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package)
        - [Installing GCC](#installing-gcc)
        - [Installing git, gmp, tclsh, patch and Flex](#installing-git-gmp-tclsh-patch-and-flex)
        - [Installing CMake](#installing-cmake)
    - [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository)
    - [Compiling MindSpore](#compiling-mindspore)
    - [Installing MindSpore](#installing-mindspore)
    - [Configuring Environment Variables](#configuring-environment-variables)
    - [Verifying the Installation](#verifying-the-installation)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.7/install/mindspore_ascend310_install_source_en.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

The following describes how to quickly install MindSpore by compiling the source code on Linux in the Ascend 310 environment, MindSpore in Ascend 310 only supports inference.

- If you want to configure an environment that can compile MindSpore on a EulerOS 2.8 with Ascend AI processor software package installed, you may use [automatic installation script](https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/euleros-ascend-source.sh) for one-click configuration, see [Environment Preparation-automatic, recommended](#environment-preparation-automatic-recommended) section. The automatic installation script will install the dependencies required to compile MindSpore.

- If some dependencies, such as Python and GCC, have been installed in your system, it is recommended to install manually by referring to the installation steps in the [Environment Preparation-manual](#environment-preparation-manual) section.

## Environment Preparation-automatic, recommended

Before running the automatic installation script, you need to make sure that the Ascend AI processor software package is correctly installed on your system. If it is not installed, please refer to the section [Installing Ascend AI processor software package](#installing-ascend-ai-processor-software-package) to install it.

Run the following command to obtain and run the automatic installation script. The environment configured by the automatic installation script only supports compiling MindSpore>=1.6.0.

```bash
wget https://gitee.com/mindspore/mindspore/raw/r1.7/scripts/install/euleros-ascend-source.sh
# install Python 3.7 by default
# the default value of LOCAL_ASCEND is /usr/local/Ascend
bash -i ./euleros-ascend-source.sh
# to specify Python 3.9 installation, and set LOCAL_ASCEND to /home/xxx/Ascend, use the following manners
# LOCAL_ASCEND=/home/xxx/Ascend PYTHON_VERSION=3.9 bash -i ./euleros-ascend-source.sh
```

This script performs the following operations:

- Install the compilation dependencies required by MindSpore, such as GCC, CMake, etc.
- Install Python3 and pip3 and set them as default.

After the automatic installation script is executed, you need to reopen the terminal window to make the environment variables take effect.

The automatic installation script creates a virtual environment named `mindspore_pyXX` for MindSpore. Where `XX` is the Python version, such as Python 3.7, the virtual environment name is `mindspore_py37`. Run the following command to show all virtual environments.

```bash
conda env list
```

Take Python 3.7 as an example, run the following command to activate the virtual environment.

```bash
conda activate mindspore_py37
```

Now you can jump to the [Downloading the Source Code from the Code Repository](#downloading-the-source-code-from-the-code-repository) section to downloading and compiling MindSpore.

For more usage, see the script header description.

## Environment Preparation-manual

The following table lists the system environment and third-party dependencies required for compiling and installing MindSpore.

|software|version|description|
|-|-|-|
|Ubuntu 18.04/CentOS 7.6/EulerOS 2.8|-|OS for compiling and running MindSpore|
|[Python](#installing-python)|3.7-3.9|Python environment that MindSpore depends|
|[Ascend AI processor software package](#installing-ascend-ai-processor-software-package)|-|Ascend platform AI computing library used by MindSpore|
|[GCC](#installing-gcc)|7.3.0|C++ compiler for compiling MindSpore|
|[git](#installing-git-gmp-tclsh-patch-and-flex)|-|Source code management tools used by MindSpore|
|[CMake](#installing-cmake)|3.18.3 or later|Compilation tool that builds MindSpore|
|[gmp](#installing-git-gmp-tclsh-patch-and-flex)|6.1.2|Multiple precision arithmetic library used by MindSpore|
|[Flex](#installing-git-gmp-tclsh-patch-and-flex)|2.5.35 or later|lexical analyzer used by MindSpore|
|[tclsh](#installing-git-gmp-tclsh-patch-and-flex)|-|MindSpore SQLite compiling dependency|
|[patch](#installing-git-gmp-tclsh-patch-and-flex)|2.5 or later|Source code patching tool used by MindSpore|

The following describes how to install the third-party dependencies.

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

### Installing Ascend AI processor software package

For detailed installation guide, please refer to [Ascend Data Center Solution 21.1.0 Installation Guide].

The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.

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

    Choose different download links based on the system architecture.

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

    Finally, add CMake to the `PATH` environment variable. Run the following commands if it is installed in the default path, other installation paths need to be modified accordingly.

    ```bash
    echo -e "export PATH=/usr/local/cmake-3.19.8/bin:\$PATH" >> ~/.bashrc
    source ~/.bashrc
    ```

## Downloading the Source Code from the Code Repository

```bash
git clone https://gitee.com/mindspore/mindspore.git -b r1.7
```

## Compiling MindSpore

Go to the root directory of MindSpore, then run the compiling script.

```bash
cd mindspore
bash build.sh -e ascend -V 310 -S on
```

Where:

- The default number of build threads is 8 in `build.sh`. If the compiler performance is poor, compiling errors may occur. You can add -j{Number of threads} to script to reduce the number of threads, for example, `bash build.sh -e ascend -V 310 -j4`.
- By default, the dependent source code is downloaded from GitHub. When -S is set to `on`, the source code is downloaded from the corresponding Gitee image.
- For details about how to use `build.sh`, see the script header description.

## Installing MindSpore

```bash
tar -zxf output/mindspore_ascend-*.tar.gz
```

## Configuring Environment Variables

After MindSpore is installed, export Runtime-related environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package, and you need to replace it as the actual installation path of the software package.

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
## AICPU path
export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
## TBE operator compilation tool path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin/:${PATH}
## Python library that TBE implementation depends on
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}

# Set path to extracted MindSpore accordingly
export LD_LIBRARY_PATH={mindspore_path}:${LD_LIBRARY_PATH}
```

Where:

- `{mindspore_path}` specifies the absolute path to which MindSpore binary package is extracted.

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

Build a project by referring to `README.md`, and`{mindspore_path}` specifies the absolute path to which MindSpore binary package is extracted, and replaces according to the actual condition.

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
