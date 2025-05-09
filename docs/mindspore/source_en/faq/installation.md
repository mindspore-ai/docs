# Installation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/installation.md)

## Installing by Using Pip

### Q: What should I do if an error message `ERROR: mindspore_{VERSION}.whl is not a supported wheel on this platform` is displayed when I install MindSpore using pip?

A: pip checks compatibility of wheel package and current Python environment by verifying the file name. For example, when installing mindspore_ascend-1.2.0-cp37-cp37m-linux_aarch64.whl, pip checks whether:

1. Python version falls in 3.7.x
2. Current operating system is Linux
3. System architecture is arm64

Hence, if you run into the problem `is not a supported wheel on this platform`, please make sure the current environment fulfills the requirements of the MindSpore package you are installing, or a correct version of MindSpore package which matches your environment is being installed.

<br/>

### Q: What should I do if an error message `ERROR: mindspore-{VERSION}.whl is not a supported wheel on this platform` is displayed when I compile MindSpore form source and install it by using pip on the macOS system?

A: First, check the name of the installation package under the output directory, which is similar to mindspore-1.6.0-cp37-cp37m-macosx_11_1_x84_64.whl. The "11_1" in the package name means that the SDK version used when compiling is 11.1. If the SDK version used is 11.x, it may be that the SDK version used during compilation is too high and cannot be installed.

Solution 1: You can rename the installation package and then try to install it. For example, rename the above installation package to mindspore-1.6.0-cp37-cp37m-macosx_10_15_x84_64.whl.

Solution 2: Before compiling the source code, set the environment variable `MACOSX_DEPOLYMENT_TARGET` to `10.15` and recompile.

<br/>

### Q: What should I do if an error message `SSL:CERTIFICATE_VERIFY_FATLED` is displayed when I use pip to install MindSpore?

A: Add the `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com` parameter to the pip installation command and try again.

<br/>

### Q: Any specific requirements for Protobuf version when use MindSpore?

A: MindSpore installs version 3.13.0 of Protobuf by default. If it is not the version, there will be many warnings in the log when using pytest to test the code.  It is recommended that you use the command 'pip install protobuf==3.13.0' to reinstall version 3.13.0.

<br/>

### Q: What should I do when an error prompts `No matching distribution found for XXX` during pip installing dependencies?

A: Please execute `pip config list` to check the software library index `index-url`. Sometimes software library index updates are out of sync. You can try other software library indexes instead.

<br/>

### Q: What should I do if I cannot find whl package for MindSpore Armour on the installation page of MindSpore website?

A: You can download whl package from the official [MindSpore Website download page](https://www.mindspore.cn/versions/en) and manually install it via `pip install`.

<br/>

### Q: What should I do if error message `Could not find a version that satisfies the requirement` is displayed when I install MindSpore on an ARM architecture system by using pip?

A: The version of pip installed on your system is most likely lower than 19.3, which is too low to recognize `manylinux2014` label. Wrong versions of python packages such as `numpy` or `scipy` are downloaded in the pip install stage, and the issue of not being able to find build dependencies is raised. As such, please upgrade pip in the environment to a later version higher than 19.3 by typing `pip install --upgrade pip`, and then try installing MindSpore again.

<br/>

### Q: What should I do if error message `Running setup.py install for pillow: finished with status 'error' ... The headers or library files could not be found for jpeg, ...` is generated when I install MindSpore by using pip?

A: MindSpore relies on the third-party library `pillow` for some data processing operations, while `pillow` needs to rely on the `libjpeg` library already installed in the environment. Taking the Ubuntu environment as an example, you can use `sudo apt-get install libjpeg8-dev` to install the `libjpeg` library, and then install MindSpore.

<br/>

### Q: What should I do if error message `ImportError: dlopen ... no suitable image found.  Did find:..._psutil_osx.cpython-38-darwin.so: mach-o, but wrong architecture` is generated when I execute MindSpore on macOS for ARM with Python3.8?

A: The default version of psutil in Python3.8 for macOS(ARM architecture) has a bug that compiles its binaries for x86 architecture, causing conflitcts when running MindSpore. To fix this, run `pip uninstall psutil; conda install psutil` if you are using Conda, otherwise run `pip uninstall psutil; pip install --no-binary :all: psutil` to install psutil correctly.
For detailed reasons, please refer to [this post on stackoverflow](https://stackoverflow.com/questions/72619143/unable-to-import-psutil-on-m1-mac-with-miniforge-mach-o-file-but-is-an-incomp).

<br/>

### Q: What should I do if error message `error: metadata-generation-failed` is generated pypi installs dependency scipy when I install MindSpore on MacOS for ARM?

A: The later versions of scipy released on pypi only supports MacOS for ARM version 12 or above. For lower versions of MacOS for ARM, pypi downloads scipy source code package and attempts to compile it on spot, which is highly likely running into troubles. If you are using Conda, you may run `pip uninstall scipy; conda install scipy`, usage of Conda is strongly advised for users of MacOS for ARM version 11 and below to prevent such compatibility issues.

<br/>

### Q: When installing GPU, CUDA 10.1, 0.5.0-beta version of MindSpore, it prompts `cannot open shared object file:No such file or directory`, what should I do?

A: The error message indicates that the cuBLAS library is not found. Generally, the cause is that the cuBLAS library is not installed or is not added to the environment variable. Generally, cuBLAS is installed together with CUDA and the driver. After the installation, add the directory where cuBLAS is located to the `LD_LIBRARY_PATH` environment variable.

<br/>

## Installing by Using Conda

### Q: For Ascend users, what should I do when `RuntimeError: json.exception.parse_error.101 parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'T'` appears in personal Conda environment?

A: When you encounter the error, te or hccl packages in your personal Conda environment are most likely not updated when updating Ascend AI processor software packages, you may uninstall the abovementioned packages first, then execute environment variables configuration script provided in the Ascend packages to set PYTHONPATH that allocates the correct version of te and hccl packages automatically: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`.

<br/>

### Q: What should I do if error message `ImportError: dlopen ... Library not loaded: @rpath/libffi.7.dylib` is generated when I execute MindSpore on macOS for ARM with Python3.9?

A: libffi is not installed automatically on some devices when creating Python 3.9 virtual environments using Conda. You may fix it by manually running `conda install libffi` to install libffi.

<br/>

## Installing by Using Source

### Q: A cross compiler has been installed on Linux, but how do I write compilation commands?

A: To compile the Arm64 version, run `bash build.sh -I arm64`. To compile the Arm32 version, run `bash build.sh -I arm32`. To set environment variables, specify the Android NDK path:  `export ANDROID_NDK=/path/to/android-ndk`. After the compilation is successful, find the compiled package in the output directory.

<br/>

### Q: What should I do if the source code compilation process takes too long or is often interrupted?

A: MindSpore imports the third party dependency package through the submodule mechanism, among which `Protobuf` v3.13.0 might not have the optimal or steady download speed. It is recommended that you perform package cache in advance.

<br/>

### Q: What should I do when the error message `MD5 does not match` is displayed when the source code is compiled?

A: This error may be due to network problems at the time of compilation caused by some third-party library download interruption. After recompiling, the file exists but is incomplete, failed to verify MD5. The solution is to delete the relevant third-party libraries in the .mslib cache path, and then recompile.

<br/>

### Q: What should I do when I have installed Python 3.7.5 and set environment variables accordingly, but still failed compiling MindSpore, with error message `Python3 not found`?

A: It's probably due to the lack of shared libraries in current Python environment. Compiling MindSpore requires linking Python share libraries, hence you may need to compile and install Python 3.7.5 from source, using command `./configure --enable-shared`.

<br/>

### Q: How to change installation directory of the third party libraries?

A: The third party library packages will be installed in build/mindspore/.mslib, and you can change the installation directory by setting the environment variable MSLIBS_CACHE_PATH, eg. `export MSLIBS_CACHE_PATH = ~/.mslib`.

<br/>

### Q: What are the directories to be cleaned if the previous compilation failed, so as to prevent the following compilation being affected by previous remains?

A: While compiling MindSpore, if:

1. Failed while downloading or compiling third-party software. e.g. failed applying patch on icu4c, with error message `Cmake Error at cmake/utils.cmake:301 (message): Failed patch:`. In this case, go to `build/mindspore/.mslib` or the third-party software installation directory specified by the `MSLIBS_CACHE_PATH` environment variable and deletes the corresponding software from it.

2. Failed in other stages of compilation, or if you wish to clean all previous build results, before recompilation, simply delete `build` directory.

<br/>

### Q: What should I do if it prompts that pthread cannot be found in CMakeError.txt after the compilation fails?

A: The real reason for the failure will be showed in the stdout log. Errors in CMakeError.txt usually do not imply the true reason of compilation failures. Please look for the first error in the stdout log.

<br/>

### Q: After the compilation is successful, an error `undefined reference to XXXX` or `undefined symbol XXXX` occurs during runtime, what should I do?

A: The possible reasons are:

1. If the problem occurs after the code is updated by `git pull`, please delete the `build` directory to exclude the impact of the previous build.

2. If the problem occurs after modifying the code, you can use `c++filt XXXX` to view the meaning of the symbol. It may be caused by unimplemented functions, unimplemented virtual functions, and unlinked dependencies, etc.

3. If the problem occurs on the Ascend platform, after excluding the above reasons, it is likely to be caused by a mismatch between the MindSpore version and the CANN version. For the version matching relationship, please refer to [Installation Guide](https://www.mindspore.cn/install/en).

<br/>

### Q: A warning `SetuptoolsDeprecationWarning: setup.py install is deprecated ...` occurs when finishing compilation, what should I do?

A: Setuptools, a Python package which manages generation of Python whl packages, has deprecated direct calling of setup.py since 58.3.0, hence you may run into such warnings while using a Python environment with a newer version of setuptools installed. This does not affect the usage of MindSpore, and we will update our pacakaging methods in a future version.
For more details, please refer to [setuptools version history](https://setuptools.pypa.io/en/latest/history.html#v58-3-0)

<br/>

### Q: What should I do if the software version required by MindSpore is not the same with the Ubuntu default software version?

A: Currently, MindSpore only provides version matching relationships, which requires you to manually install and upgrade the companion software. (**Note**: MindSpore requires Python3.7.5 and gcc7.3, and the default version in Ubuntu 16.04 are Python3.5 and gcc5, whereas the default one in Ubuntu 18.04 are Python3.7.3 and gcc7.4)

<br/>

### Q: What should I do when the error message `No module named 'mindspore.version` is displayed when the use case is executed?

A: When there is such an error, it is possible to execute a use case in the path that created the same name as the MindSpore installation package, causing Python to preferentially find the current directory when importing the package, and the current directory does not version.py the file. The solution is to rename the directory or exit the one- or multi-level directory upwards.

<br/>

### Q: MindSpore installation: Version 0.6.0-beta + Ascend + Ubuntu_aarch64 + Python3.7.5, manually download the whl package of the corresponding version, compile and install gmp6.1.2. Other Python library dependencies have been installed, the execution of the sample fails, and an error shows that the so file cannot be found.

A: The `libdatatransfer.so` dynamic library is in the `fwkacllib/lib64` directory. Find the path of the library in the `/usr/local` directory, and then add the path to the `LD_LIBRARY_PATH` environment variable. After the settings take effect, execute the sample again.

<br/>

## Installing by Using Docker

### Q: Whether MindSpore can support Nvidia GPU discrete graphics + Windows operating system pc?

A: At present, the support of MindSpore is the combination configuration of GPU +Linux and CPU + Windows, and the support of Windows + GPU is still under development.

If you want to run on a GPU+Windows environment, you can try to use WSL+docker, the operation idea:

1. For installing Ubuntu18.04 in WSL mode, refer to [Install Linux on Windows with WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

2. For installing Nvidia drivers that support WSL and deploying in environments where containers run on WSL, refer to [WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

    > Since CUDA on WSL is still a preview feature, pay attention to the description of the Windows version requirements in the reference link, and the version is not enough to be upgraded.

3. Referring to [Docker Image](https://gitee.com/mindspore/mindspore/blob/master/README.md#docker-image), take MindSpore-GPU images. For example, take the MindSpore1.0.0 version container, and execute `docker pull mindspore/mindspore-gpu:1.0.0` to execute the container in WSL Ubuntu18.04:

    ```docker
    docker run -it --runtime=nvidia mindspore/mindspore-gpu:1.0.0 /bin/bash
    ```

The detailed steps can refer to the practice provided by the community [Zhang Xiaobai teaches you to install the GPU driver (CUDA and cuDNN)](https://www.hiascend.com/developer/blog/details/0235122004927955034). Thanks to the community member [Zhang Hui](https://www.hiascend.com/forum/otheruser?uid=110545f972cf4903b3deba9cfaac7fc8) for sharing.

<br/>

## Uninstalling

### Q: How to uninstall MindSpore?

A: First of all, please confirm the full name of MindSpore, for example, for the gpu version, and you can execute the command `pip uninstall mindspore-gpu` to uninstall.

<br/>

## Environment Variables

### Q: Some frequently-used environment settings need to be reset in the newly started terminal window, which is easy to be forgotten. What should I do?

A: You can write the frequently-used environment settings to `~/.bash_profile` or `~/.bashrc` so that the settings can take effect immediately when you start a new terminal window.

<br/>

### Q: How to set environment variable `DEVICE_ID` when using GPU version of MindSpore?

A: Normally, GPU version of MindSpore doesn't need to set `DEVICE_ID`. MindSpore automatically chooses visible GPU devices according to the cuda environment variable `CUDA_VISIBLE_DEVICES`. After setting `CUDA_VISIBLE_DEVICES`, `DEVICE_ID` refers to the ordinal of the GPU device:

- After `export CUDA_VISIBLE_DEVICES=1,3,5`, `DEVICE_ID` should be exported as `0`, `1` or `2`. If `3` is exported, MindSpore will fail to execute because of the invalid device ordinal.

<br/>

### Q: What should I do when error `/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found` prompts during application compiling?

A: Find the directory where the missing dynamic library file is located, add the path to the environment variable `LD_LIBRARY_PATH`.

<br/>

### Q: What should I do when error `ModuleNotFoundError: No module named 'te'` prompts during application running?

A: First confirm whether the system environment is installed correctly and whether the whl packages such as `te` and `topi` are installed correctly. If there are multiple Python versions in the user environment, such as Conda virtual environment, you need to execute `ldd name_of_your_executable_app` to confirm whether the application link `libpython3.so` is consistent with the current Python directory, if not, you need to adjust the order of the environment variable `LD_LIBRARY_PATH`. For example:

```bash
export LD_LIBRARY_PATH=`python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`:$LD_LIBRARY_PATH
```

Add the library directory of the program corresponding to the current `python` command to the top of `LD_LIBRARY_PATH`.

<br/>

### Q: What should I do when error `error while loading shared libraries: libpython3.so: cannot open shared object file: No such file or directory` prompts during application running?

A: This error usually occurs in an environment where multiple Python versions are installed. First, confirm whether the `lib` directory of Python is in the environment variable `LD_LIBRARY_PATH`, this command can set it:

```bash
export LD_LIBRARY_PATH=`python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`:$LD_LIBRARY_PATH
```

Then, if Python 3.7.6 or lower does not contain this dynamic library in the Conda virtual environment, you can run the command to upgrade the Python version in Conda, such as: `conda install python=3.7.11`.

<br/>

### Q: Ascend AI processor software package and other prerequisites have been installed, but executing MindSpore failed with error message `Cannot open shared objectfile: No such file or directory`, what should I do?

A: There are 2 common reasons: an incorrect version of Ascend AI processor software package is installed, or the packages are installed in customized paths, yet the environment varialbes are not set accordingly.

1. Go to directory where Ascend AI processor software package is installed, being `/usr/local/Ascend` by default, open `version.info` files which are located under directories of subpackages, and check if the version number matches the requirement of MindSpore. Please refer to [Install](https://www.mindspore.cn/install/en) for the detailed version required by MindSpore versions. If the version number does not match, please update the packages accordingly.

2. Check if Ascend AI processor software package is installed in the default directory. MindSpore attempts to load packages from default directory `/usr/local/Ascend`, if the packages are installed in a customized directory, please follow the instructions from `Configuring Environment Variables` section of [Install](https://www.mindspore.cn/install/en) and set environment variables to match the changes. If other dependencies are installed in customized directies, then please set `LD_LIBRARY_PATH` environment variable accordingly.

<br/>

### Q: What should I do when multiple versions of Ascend AI processor software are installed on the same environment, errors such as segmentation fault, cannot open shared object file: No such file or directory occurs?

A: Versions prior to MindSpore 2.4.0 sets environment path using RPATH, on environments where both CANN-nnae and CANN-toolkit packages are installed, this may result in CANN related binaries being loaded in wrong sequences. MindSpore has removed this limitation since version 2.4.0, and requires users to set environment paths before use manually. If you have to install the abovementioned CANN packages at the same time, please upgrade to MindSpore 2.4.0 or higher, and configure environment paths using script provided by CANN. Please refer to [Install](https://www.mindspore.cn/install/en) to set environment variables.

<br/>

## Verifying the Installation

### Q: Does MindSpore of the GPU version have requirements on the computing capability of devices?

A: Currently, MindSpore supports only devices with the computing capability version greater than 5.3.

<br/>

### Q: After MindSpore is installed on a CPU of a PC, an error message `the pointer[session] is null` is displayed during code verification. The specific code is as follows. How do I verify whether MindSpore is successfully installed?

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device(device_target="Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x,y))
```

A: To verify whether MindSpore is installed successfully, please follow the `Installation Verification` part under installation guide:

```bash
python -c "import mindspore;mindspore.run_check()"
```

The outputs should be the same as:

```text
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

It means MindSpore has been installed successfully.

<br/>

### Q: What should I do do when the errors prompts, such as `sh:1:python:not found`, `No module named mindspore._extends.remote` that the Python was linked to Python2.7 when the use case is executed on `linux` platform?

A: When you encounter such problem, it is most likely caused by Python environment. Use the following command to check whether the current Python environment meets the requirements of MindSpore.

- Enter `python` in the terminal window and check the following version information entered into the Python interactive environment. If an error is reported directly, there is no Python soft connection; if you enter a non-Python 3.7 environment, the current Python environment is not the MindSpore runtime needs.
- Execute the `sudo ln -sf /usr/bin/python3.7.x /usr/bin/python` command to create Python's soft connection, and then check the execution.

<br/>

### Q: Here in script when we import other python lib before `import mindspore`, error raised like follows (`/{your_path}/libgomp.so.1: cannot allocate memory in static TLS block`), how can we solve it?

A: Above question is relatively common, and there are two feasible solutions, you can choose one of them:

- Exchange the order of import, first `import mindspore` and then import other third party libraries.
- Before executing the program, we can add environment variables first (`export LD_PRELOAD=/{your_path}/libgomp.so.1`), where `{your_path}` is the path mentioned in above error.

<br/>

### Q: What should I do if an error message `ImportError: libgmpxx.so: cannot open shared object file: No such file or directory` is displayed when running `import mindspore` in a script after the source code of MindSpore and gmp are compiled and installed?

A: The reason is that we didn't set `--enable-cxx` when installing gmp. The correct steps for installing gmp is (suppose that we have download gmp installation repository):

```bash
$cd gmp-6.1.2
$./configure --enable-cxx
$make
$make check
$sudo make install
```

<br/>

### Q: What should I do if an warning message `UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.` is displayed when running MindSpore?

A: The above issue occurs in environments with a newer version of numpy (>=1.22.0) version of ARM python 3.9 installed. We observed such warnings on ARM environment, with python 3.9 and numpy >=1.22.0 installed. The wharnings come from numpy and not from MindSpore. If the wharnings affect the normal debugging of the code, you can consider manually installing an earlier version of numpy (<=1.21.2) to circumvent it.

<br/>

### Q: What should I do if an error message `AttributeError: module 'six' has no attribute 'ensure_text'` is displayed when running MindSpore?

A: The reason for the above issue is that a newer version of `asttokens` (>=2.0.6) is installed, but the version of `six` on which it depends does not match. It can be solved by updating the `six` version (>=1.12.0).

<br/>

### Q: What should I do if an error message `ModuleNotFoundError: No module named 'requests'` is displayed when running MindSpore?

A: The reason for the above issue is that python package requests is missing in current python environment, it can be solved by running `pip install requests`.
