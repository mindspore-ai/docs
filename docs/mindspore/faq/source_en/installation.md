# Installation

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/faq/source_en/installation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Installing Using pip

<font size=3>**Q: When installing GPU, CUDA 10.1, 0.5.0-beta version of MindSpore, it prompts `cannot open shared object file:No such file or directory`, what should I do?**</font>

A: The error message indicates that the cuBLAS library is not found. Generally, the cause is that the cuBLAS library is not installed or is not added to the environment variable. Generally, cuBLAS is installed together with CUDA and the driver. After the installation, add the directory where cuBLAS is located to the `LD_LIBRARY_PATH` environment variable.

<br/>

<font size=3>**Q: What should I do if an error message `ERROR: mindspore_{VERSION}.whl is not a supported wheel on this platform` is displayed when I install MindSpore using pip?**</font>

A: pip checks compatibility of wheel package and current Python environment by verifying the file name. For example, when installing mindspore_ascend-1.2.0-cp37-cp37m-linux_aarch64.whl, pip checks whether:

1. Python version falls in 3.7.x.
2. Current operating system is a linux distro.
3. System architecture is arm64.

Hence, if you run into such a problem, please make sure the current environment fulfills the requirements of the MindSpore package you are installing, or a correct version of MindSpore package which matches your environment is being installed.

<br/>

<font size=3>**Q: What should I do if an error message `SSL:CERTIFICATE_VERIFY_FATLED` is displayed when I use pip to install MindSpore?**</font>

A: Add the `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com` parameter to the pip installation command and try again.

<br/>

<font size=3>**Q: Any specific requirements for Protobuf version when use MindSpore?**</font>

A: MindSpore installs version 3.13.0 of Protobuf by default.  If it is not the version, there will be many warnings in the log when using pytest to test the code.  It is recommended that you use the command 'pip install protobuf==3.13.0' to reinstall version 3.13.0.

<br/>

<font size=3>**Q: What should I do when error `ProxyError(Cannot connect to proxy)` prompts during pip install?**</font>

A: It is generally a proxy configuration problem, you can using `export http_proxy={your_proxy}` on Ubuntu environment, and using `set http_proxy={your_proxy}` in cmd on Windows environment to config your proxy.

<br/>

<font size=3>**Q: What should I do when error prompts during pip install?**</font>

A: Please execute `pip -V` to check if pip is linked to Python3.7+. If not, we recommend you
use `python3.7 -m pip install` instead of `pip install` command.

<br/>

<font size=3>**Q: What should I do when error prompts `No matching distribution found for XXX` during pip install dependencies?**</font>

A: Please execute `pip config list` to check the package index `index-url`. Sometimes package index updates are out of sync. You can try other package index instead.

<br/>

<font size=3>**Q: What should I do if I cannot find whl package for MindInsight or MindArmour on the installation page of MindSpore website?**</font>

A: You can download whl package from the official [MindSpore Website download page](https://www.mindspore.cn/versions/en) and manually install it via `pip install`.

<br/>

<font size=3>**Q: For Ascend users, what should I do when `RuntimeError: json.exception.parse_error.101 parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'T'` appears in personal Conda environment?**</font>

A: When you encounter the error, you should update the `te/topi/hccl` python toolkits, unload them firstly and then using command `pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/{te/topi/hccl}-{version}-py3-none-any.whl` to reinstall.

<br/>

<font size=3>**Q: What should I do when I install both CPU and GPU versions of MindSpore, an error occurs while importing mindspore, saying `cannot import name 'context' from 'mindspore'`?**</font>

A: All versions of MindSpore are installed in the directory named `mindspore`, installing them in one Python environment may overwrite each other and cause failures. If you wish to use alternate versions of MindSpore for different platforms (e.g. CPU and GPU versions), please uninstall the unused version and install the new version afterwards.

<br/>

<font size=3>**Q: What should I do if error message `Could not find a version that satisfies the requirement` is generated when I install MindSpore on a ARM architecture system using pip?**</font>

A: The version of pip installed on your system is most likely lower than 19.3, which is too low to recognize `manylinux2014` label that identifies ARM64 architecture for pypi. Wrong versions of python packages such as `numpy` or `scipy` are downloaded, and dependencies are then found lacking while trying to build these packages. As such, please upgrade pip to a later version by typing `pip install --upgrade pip`, and then try installing MindSpore again.

<br/>

<font size=3>**Q: What should I do if error message `Running setup.py install for pillow: finished with status 'error' ... The headers or library files could not be found for jpeg, ...` is generated when I install MindSpore using pip?**</font>

A: MindSpore relies on the third-party library `pillow` for some data processing operations, while `pillow` needs to rely on the `libjpeg` library already installed in the environment. Take the Ubuntu environment as an example, you can use `sudo apt-get install libjpeg8-dev` to install the `libjpeg` library, and then install MindSpore.

<br/>

## Source Code Compilation Installation

<font size=3>**Q: What is the difference between `bash -p` and `bash -e` when an error is reported during application build?**</font>

A: MindSpore Serving build and running depend on MindSpore. Serving provides two build modes: 1. Use `bash -p {python site-packages}/mindspore/lib` to specify an installed MindSpore path to avoid building MindSpore when building Serving. 2. Build Serving and the corresponding MindSpore. Serving passes the `-e`, `-V`, and `-j` options to MindSpore.
For example, use `bash -e ascend -V 910 -j32` in the Serving directory as follows:

- Build MindSpore in the `third_party/mindspore` directory using `bash -e ascend -V 910 -j32`.
- Use the MindSpore build result as the Serving build dependency.

<br/>

<font size=3>**Q: A cross compiler has been installed on Linux, but how do I write compilation commands?**</font>

A: Set environment variables and specify the Android NDK path using `export ANDROID_NDK=/path/to/android-ndk`. To compile the Arm64 version, run `bash build.sh -I arm64`. To compile the Arm32 version, run `bash build.sh -I arm32`. After the compilation is successful, find the compiled package in the output directory.

<br/>

<font size=3>**Q: MindSpore installation: Version 0.6.0-beta + Ascend 910 + Ubuntu_aarch64 + Python3.7.5, manually download the whl package of the corresponding version, compile and install gmp6.1.2. Other Python library dependencies have been installed, the execution of the sample fails, and an error shows that the so file cannot be found.**</font>

A: The `libdatatransfer.so` dynamic library is in the `fwkacllib/lib64` directory. Find the path of the library in the `/usr/local` directory, and then add the path to the `LD_LIBRARY_PATH` environment variable. After the settings take effect, execute the sample again.

<br/>

<font size=3>**Q: What should I do if the compilation time of MindSpore source code takes too long or the process is constantly interrupted by errors?**</font>

A: MindSpore imports third party dependencies through submodule mechanism, among which `Protobuf` v3.13.0 might not have the optimal or steady download speed, it is recommended that you perform package cache in advance.

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

A: It's probably due to the lack of shared libraries in current Python environment. Compiling MindSpore requires linking Python shared libraries, hence you may need to compile and install Python 3.7.5 from source, using command `./configure --enable-shared`.

<br/>

<font size=3>**Q: What are the directories to be cleaned if the previous compilation failed, so as to prevent the following compilation being affected by previous remains?**</font>

A: While compiling MindSpore, if:

1. Failed while downloading or compiling third-party software. e.g. Failed applying patch on icu4c, with error message `Cmake Error at cmake/utils.cmake:301 (message): Failed patch:`. In this case, go to `build/mindspore/.mslib` or directories specified by environment vairiable `MSLIBS_CACHE_PATH` where third-party software are installed, and delete affected software respectively.

2. Failed in other stages of compilation, or if you wish to clean all previous build results, simply delete `build` directory.

<br/>

<font size=3>**Q: what should I do when an error message `No module named 'mindpore.version'` is displayed when I execute the case?**</font>

A: Maybe you execute the case in the path with the same name as the MindSpore installation package. Rename the directory or exit one or more levels of directory to solve the problem.

<br/>

<font size=3>**Q: what should I do when an error message `MD5 does not match` is displayed when I execute the case?**</font>

A: This kind of error may be caused by internet problem when some third party libraries are downloading. It fails to verify MD5 with a incomplete file when you recompile the project. Remove the related third-party file in .mslib cache path and recompile the project to solve the problem.

<br/>

<font size=3>**Q: What should I do if it prompts that `pthread not found` in CMakeError.txt after the compilation fails?**</font>

A: The real reason for the failure will be showed in the stdout log. CMakeError.txt has no reference value. Please look for the first error in the stdout log.

<br/>

<font size=3>**Q: After the compilation is successful, an error `undefined reference to XXXX` or `undefined symbol XXXX` occurs during runtime, what should I do?**</font>

A: The possible reasons are:

1. If the problem occurs after the code is updated by `git pull`, please delete the `build` directory to exclude the impact of the previous build.

2. If the problem occurs after modifying the code, you can use `c++filt XXXX` to view the meaning of the symbol. It may be caused by unimplemented functions, unimplemented virtual functions, and unlinked dependencies, etc.

3. If the problem occurs on the Ascend platform, after excluding the above reasons, it is likely to be caused by a mismatch between the MindSpore version and the CANN version. For the version matching relationship, please refer to [Installation Guide](https://www.mindspore.cn/install/en).

<br/>

## Uninstall

<font size=3>**Q: How to uninstall MindSpore?**</font>

A: First of all, please confirm the full name of MindSpore, for example, for the gpu version, you can execute the command `pip uninstall mindspore-gpu` to uninstall.

<br/>

## Environment Variables

<font size=3>**Q: Some frequently-used environment settings need to be reset in the newly started terminal window, which is easy to be forgotten, What should I do?**</font>

A: You can write the frequently-used environment settings to `~/.bash_profile` or `~/.bashrc` so that the settings can take effect immediately when you start a new terminal window.

<br/>

<font size=3>**Q: How to set environment variable `DEVICE_ID` when using GPU version of MindSpore**</font>

A: Normally, GPU version of MindSpore doesn't need to set `DEVICE_ID`. MindSpore automatically chooses visible GPU devices according to the cuda environment variable `CUDA_VISIBLE_DEVICES`. After setting `CUDA_VISIBLE_DEVICES`, `DEVICE_ID` refers to the ordinal of the GPU device:

- After `export CUDA_VISIBLE_DEVICES=1,3,5`, `DEVICE_ID` should be exported as `0`, `1` or `2`. If `3` is exported, MindSpore will fail to execute because of the invalid device ordinal.

<br/>

<font size=3>**Q: What should I do when error `/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found` prompts during application compiling?**</font>

A: Find the directory where the missing dynamic library file is located, add the path to the environment variable `LD_LIBRARY_PATH`, and refer to [Inference Using the MindIR Model on Ascend 310 AI Processors#Building Inference Code](https://www.mindspore.cn/docs/programming_guide/en/r1.6/multi_platform_inference_ascend_310_mindir.html#building-inference-code) for environment variable settings.

<br/>

<font size=3>**Q: What should I do when error `ModuleNotFoundError: No module named 'te'` prompts during application running?**</font>

A: First confirm whether the system environment is installed correctly and whether the whl packages such as `te` and `topi` are installed correctly. If there are multiple Python versions in the user environment, such as Conda virtual environment, you need to execute `ldd name_of_your_executable_app` to confirm whether the application link `libpython3.so` is consistent with the current Python directory, if not, you need to adjust the order of the environment variable `LD_LIBRARY_PATH`. For example:

```bash
export LD_LIBRARY_PATH=`python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`:$LD_LIBRARY_PATH
```

Add the library directory of the program corresponding to the current `python` command to the top of `LD_LIBRARY_PATH`.

<br/>

<font size=3>**Q: What should I do when error `error while loading shared libraries: libpython3.so: cannot open shared object file: No such file or directory` prompts during application running?**</font>

A: This error usually occurs in an environment where multiple Python versions are installed. First, confirm whether the `lib` directory of Python is in the environment variable `LD_LIBRARY_PATH`, this command can set it:

```bash
export LD_LIBRARY_PATH=`python -c "import distutils.sysconfig as sysconfig ; print(sysconfig.get_config_var('LIBDIR'))"`:$LD_LIBRARY_PATH
```

Then, if Python 3.7.6 or lower does not contain this dynamic library in the Conda virtual environment, you can run the command to upgrade the Python version in Conda, such as: `conda install python=3.7.11`.

<br/>

<font size=3>**Q: Ascend AI processor software package and other prerequisites have been installed, but executing MindSpore failed with error message `Cannot open shared objectfile: No such file or directory`, what should I do?**</font>

A: There are 2 common reasons: an incorrect version of Ascend AI processor software package is installed, or the packages are installed in customized paths, yet the environment varialbes are not set accordingly.

1. Go to directory where Ascend AI processor software package is installed, being `/usr/local/Ascend` by default, open `version.info` files which are located under directories of subpackages, and check if the version number matches the requirement of MindSpore. Please refer to [Install](https://www.mindspore.cn/install/en) for the detailed version required by MindSpore versions. If the version number does not match, please update the packages accordingly.

2. Check if Ascend AI processor software package is installed in the default directory. MindSpore attempts to load packages from default directory `/usr/local/Ascend`, if the packages are installed in a customized directory, please follow the instructions from `Configuring Environment Variables` section of [Install](https://www.mindspore.cn/install/en) and set environment variables to match the changes. If other dependencies are installed in customized directies, then please set `LD_LIBRARY_PATH` environment variable accordingly.

<br/>

## Verifying the Installation

<font size=3>**Q: Does MindSpore of the GPU version have requirements on the computing capability of devices?**</font>

A: Currently, MindSpore supports only devices with the computing capability version greater than 5.3.

<br/>

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

- Enter `python` in the terminal window and check the following version information entered into the Python interactive environment. If an error is reported directly, there is no Python soft connection; if you enter a non-Python 3.7 environment, the current Python environment is not the MindSpore runtime needs.
- If not, execute the `sudo ln -sf /usr/bin/python3.7.x /usr/bin/python` command to create Python's soft connection.

<br/>

<font size=3>**Q: Here in script when we import other python lib before `import mindspore`, error raised like follows (`/{your_path}/libgomp.so.1: cannot allocate memory in static TLS block`), how can we solve it?**</font>

A: Above question is relatively common, and there are two feasible solutions, you can choose one of them:

- Exchange the order of import, first `import mindspore` and then import other third party libraries.
- Before executing the program, we can add environment variables first (`export LD_PRELOAD=/{your_path}/libgomp.so.1`), where `{your_path}` is the path mentioned in above error.

<br/>

<font size=3>**Q: What should I do if an error message `ImportError: libgmpxx.so: cannot open shared object file: No such file or directory` is displayed
when running `import mindspore` in a script after the source code of mindspore and gmp are compiled and installed?**</font>

A: The reason is that we didn't set `--enable-cxx` when installing gmp. The correct steps for installing gmp is (suppose that
we have download gmp installation repository):

```bash
$cd gmp-6.1.2
$./configure --enable-cxx
$make
$make check
$sudo make install
```

<br/>

<font size=3>**What should I do if an warning message `UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.` is displayed when running Mindspore?**</font>

A: We observed such warnings on ARM environment, with python 3.9 and numpy >=1.22.0 installed. These wharnings come from numpy instead of MindSpore, if you wish to suppress these warnings, please consider manually switching to a lower version of numpy (<=1.21.2).
