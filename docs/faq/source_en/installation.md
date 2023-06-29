# Installation

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_en/installation.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Installing Using pip

<font size=3>**Q: What can I do if an error message `cannot open shared object file:file such file or directory` is displayed when I install MindSpore of the GPU, CUDA 10.1, 0.5.0-beta, or Ubuntu-x86 version?**</font>

A: The error message indicates that the cuBLAS library is not found. Generally, the cause is that the cuBLAS library is not installed or is not added to the environment variable. Generally, cuBLAS is installed together with CUDA and the driver. After the installation, add the directory where cuBLAS is located to the `LD_LIBRARY_PATH` environment variable.

<br/>

<font size=3>**Q: What should I do if an error message `SSL:CERTIFICATE_VERIFY_FATLED` is displayed when I use pip to install MindSpore?**</font>

A: Add the `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com` parameter to the pip installation command and try again.

<br/>

<font size=3>**Q: Any specific requirements for Python version when pip install MindSpore?**</font>

A: MindSpore utilizes many of the new features in Python3.7+，therefore we recommend you add Python3.7.5 develop environment via `conda`.

<br/>

<font size=3>**Q: Any specific requirements for protobuf version when use MindSpore?**</font>

A: MindSpore installs version 3.8.0 of protobuf by default. If you have installed 3.12.0 or later version of protobuf locally, there will be many warnings in the log when using pytest to test the code. It is recommended that you use the command 'pip install protobuf==3.8.0' to reinstall version 3.8.0.

<br/>

<font size=3>**Q: What should I do when error `ProxyError(Cannot connect to proxy)` prompts during pip install?**</font>

A: It is generally a proxy configuration problem, you can using `export http_proxy={your_proxy}` on Ubuntu environment, and using `set http_proxy={your_proxy}` in cmd on Windows environment to config your proxy.

<br/>

<font size=3>**Q: What should I do when error prompts during pip install?**</font>

A: Please execute `pip -V` to check if pip is linked to Python3.7+. If not, we recommend you
use `python3.7 -m pip install` instead of `pip install` command.

<br/>

<font size=3>**Q: What should I do if I cannot find whl package for MindInsight or MindArmour on the installation page of MindSpore website?**</font>

A: You can download whl package from the official [MindSpore Website download page](https://www.mindspore.cn/versions) and manually install it via `pip install`.

<br/>

## Source Code Compilation Installation

<font size=3>**Q: A cross compiler has been installed on Linux, but how do I write compilation commands?**</font>

A: Set environment variables and specify the Android NDK path using `export ANDROID_NDK=/path/to/android-ndk`. To compile the Arm64 version, run `bash build.sh -I arm64`. To compile the Arm32 version, run `bash build.sh -I arm32`. After the compilation is successful, find the compiled package in the output directory.

<br/>

<font size=3>**Q: A sample fails to be executed after I installed MindSpore 0.6.0 beta on Ascend 910 using Ubuntu_aarch64 and Python 3.7.5 and manually downloaded the .whl package of the corresponding version, compiled and installed GMP6.1.2, and installed other Python library dependencies. An error message is displayed, indicating that the .so file cannot be found. What can I do?**</font>

A: The `libdatatransfer.so` dynamic library is in the `fwkacllib/lib64` directory. Find the path of the library in the `/usr/local` directory, and then add the path to the `LD_LIBRARY_PATH` environment variable. After the settings take effect, execute the sample again.

<br/>

<font size=3>**Q: What should I do if the compilation time of MindSpore source code takes too long or the process is constantly interrupted by errors?**</font>

A: MindSpore imports third party dependencies through submodule mechanism, among which `protobuf` v3.8.0 might not have the optimal or steady download speed, it is recommended that you perform package cache in advance.

<br/>

<font size=3>**Q: How to change installation directory of the third party libraries?**</font>

A: The third party libraries will be installed in build/mindspore/.mslib, you can change the installation directory by setting the environment variable MSLIBS_CACHE_PATH, eg. `export MSLIBS_CACHE_PATH = ~/.mslib`.

<br/>

<font size=3>**Q: What should I do if the software version required by MindSpore is not the same with the Ubuntu default software version?**</font>

A: At the moment some software might need manual upgrade. (**Note**: MindSpore requires Python3.7.5 and gcc7.3，the default version in Ubuntu 16.04 are Python3.5 and gcc5，whereas the one in Ubuntu 18.04 are Python3.7.3 and gcc7.4)

<br/>

<font size=3>**Q: What should I do if there is a prompt `tclsh not found` when I compile MindSpore from source code?**</font>

A: Please install the software manually if there is any suggestion of certain `software not found`.

<br/>

<font size=3>**Q: what should I do when I have installed Python 3.7.5 and set environment variables accordingly, but still failed compiling MindSpore, with error message `Python3 not found`?**</font>

A:　It's probably due to the lack of shared libraries in current Python environment. Compiling MindSpore requires linking Python shared libraries, hence you may need to compile and install Python 3.7.5 from source, using command `./configure --enable-shared`.

<br/>

<font size=3>**Q: What are the directories to be cleaned if the previous compilation failed, so as to prevent the following compilation being affected by previous remains?**</font>

A: While compiling MindSpore, if:

1. Failed while downloading or compiling third-party software. e.g. Failed applying patch on icu4c, with error message `Cmake Error at cmake/utils.cmake:301 (message): Failed patch:`. In this case, go to `build/mindspore/.mslib` or directories specified by environment vairiable `MSLIBS_CACHE_PATH` where third-party software are installed, and delete affected software respectively.

2. Failed in other stages of compilation, or if you wish to clean all previous build results, simply delete `build` directory.

<br/>

## Uninstall

<font size=3>**Q: How to uninstall MindSpore?**</font>

A: Using `pip uninstall mindspore` to uninstall MindSpore.

<br/>

## Environment Variables

<font size=3>**Q: Some frequently-used environment settings need to be reset in the newly started terminal window, which is easy to be forgotten, What should I do?**</font>

A: You can write the frequently-used environment settings to `~/.bash_profile` or `~/.bashrc` so that the settings can take effect immediately when you start a new terminal window.

<br/>

<font size=3>**Q: Ascend AI processor software package and other prerequisites have been installed, but executing MindSpore failed with error message `Cannot open shared objectfile: No such file or directory`, what should I do?**</font>

A: There are 2 common reasons: an incorrect version of Ascend AI processor software package is installed, or the packages are installed in customized paths, yet the environment varialbes are not set accordingly.

1. Go to directory where Ascend AI processor software package is installed, being `/usr/local/Ascend` by default, open `version.info` files which are located under directories of subpackages, and check if the version number matches the requirement of MindSpore. Please refer to [Install](https://www.mindspore.cn/install/en) for the detailed version required by MindSpore versions. If the version number does not match, please update the packages accordingly.

2. Check if Ascend AI processor software package is installed in the default directory. MindSpore attempts to load packages from default directory `/usr/local/Ascend`, if the packages are installed in a customized directory, please follow the instructions from `Configuring Environment Variables` section of [Install](https://www.mindspore.cn/install/en) and set environment variables to match the changes. If other dependencies are installed in customized directies, then please set `LD_LIBRARY_PATH` environment variable accordingly.

<br/>

## Verifying the Installation

<font size=3>**Q: After MindSpore is installed on a CPU of a PC, an error message `the pointer[session] is null` is displayed during code verification. The specific code is as follows. How do I verify whether MindSpore is successfully installed?**</font>

```python
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x,y))
```

A: After MindSpore is installed on a CPU hardware platform, run the `python -c'import mindspore'` command to check whether MindSpore is successfully installed. If no error message such as `No module named'mindspore'` is displayed, MindSpore is successfully installed. The verification code is used only to verify whether a Ascend platform is successfully installed.

<br/>

<font size=3>**Q: What should I do do when the errors prompts, such as `sh:1:python:not found`, `No module named mindspore._extends.remote` that the Python was linked to Python2.7?**</font>

A: Use the following command to check whether the current Python environment meets the requirements of MindSpore.

- Text `python` in terminal window, check whether the version of Python interactive environment is `3.7.x`
- If not, execute the `sudo ln -sf /usr/bin/python3.7.x /usr/bin/python` command to create Python's soft connection.

<br/>

<font size=3>**Q: Here in script when we import other python lib before `import mindspore`, error raised like follows (`/your_path/libgomp.so.1: cannot allocate memory in static TLS block`), how can we solve it?**</font>

A: Above question is relatively common, and there are two feasible solutions, you can choose one of them:

- Exchange the order of import, first `import mindspore` and then import other third party libraries.
- Before executing the program, we can add environment variables first (`export LD_PRELOAD=/your_path/libgomp.so.1`), where `your_path` is the path mentioned in above error.
