# 安装

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/faq/installation.md)

## Pip安装

### Q: 使用pip安装时报错: `ERROR: mindspore_{VERSION}.whl is not a supported wheel on this platform`应该怎么办？

A: pip会通过wheel安装包的文件名来判断该安装包是否与当前Python环境兼容，例如安装mindspore_ascend-1.2.0-cp37-cp37m-linux_aarch64.whl时，pip会检查:

1. 当前python环境为3.7.x版本
2. 当前操作系统为Linux
3. 操作系统架构为arm64

因此，如果出现 `is not a supported wheel on this platform` 问题，请检查当前环境是否满足MindSpore安装要求，以及该MindSpore安装包版本是否正确。

<br/>

### Q: macOS系统源码编译后使用pip安装报错: `ERROR: mindspore-{VERSION}.whl is not a supported wheel on this platform`应该怎么办？

A: 首先检查output目录下编译得到的安装包名，类似mindspore-1.6.0-cp37-cp37m-macosx_11_1_x84_64.whl。包名中“11_1”的意思是编译时使用的SDK版本是11.1。如果使用的SDK版本为11.x，则可能是因为编译时使用的SDK版本过高导致无法安装。

解决方法一：可以重命名安装包后再尝试安装，例如将上述安装包重命名为mindspore-1.6.0-cp37-cp37m-macosx_10_15_x84_64.whl。

解决方法二：在源码编译前，设置环境变量`MACOSX_DEPOLYMENT_TARGET`为`10.15`并重新编译。

<br/>

### Q: 使用pip安装时报错: `SSL:CERTIFICATE_VERIFY_FATLED`应该怎么办？

A: 在pip安装命令后添加参数 `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com`重试即可。

<br/>

### Q: MindSpore对Protobuf版本是否有特别要求？

A: MindSpore默认安装Protobuf的3.13.0版本，如果不是该版本，在使用pytest测试代码时日志中会产生很多告警，建议您使用命令`pip install protobuf==3.13.0`重新安装3.13.0版本。

<br/>

### Q: 使用pip安装依赖库时提示`No matching distribution found for XXX`错误，应该怎么办？

A: 请执行`pip config list`，查看当前软件库索引路径`index-url`。某些情况下，软件库索引会出现更新滞后，可尝试设置其他软件库索引路径。

<br/>

### Q: MindSpore网站安装页面找不到MindSpore Armour的whl包，无法安装怎么办？

A: 您可以从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，通过`pip install`命令进行安装。

<br/>

### Q: 在ARM架构的环境上使用pip安装MindSpore时报错: `Could not find a version that satisfies the requirement`应该怎么办？

A: 大概率是因为pip版本低于19.3，无法识别`manylinux2014`标签，导致pip install阶段下载了错误版本的`numpy`或`scipy`等python软件包，进而引发了无法找到构建依赖的问题，请执行`pip install --upgrade pip`将环境中的pip升级到19.3以上，重新安装MindSpore。

<br/>

### Q: pip安装MindSpore时，报错 `Running setup.py install for pillow: finished with status 'error' ... The headers or library files could not be found for jpeg, ...`，应该怎么办？

A: MindSpore依赖三方库`pillow`进行部分的数据处理操作，而`pillow`需要依赖环境上已经安装`libjpeg`库，以Ubuntu环境为例，可以使用`sudo apt-get install libjpeg8-dev`来安装`libjpeg`库，然后再安装MindSpore。

<br/>

### Q: ARM版macOS在Python3.8环境中编译的mindspore包安装后执行报错 `ImportError: dlopen ... no suitable image found.  Did find:..._psutil_osx.cpython-38-darwin.so: mach-o, but wrong architecture`怎么办？

A: ARM版macOS上的Python3.8包含的psutil无法正确识别当前系统的架构，会自编译为适配x86架构的二进制，因此出现冲突问题。如果在Conda环境则执行`pip uninstall psutil; conda install psutil`，如果在非Conda环境中则执行`pip uninstall psutil; pip install --no-binary :all: psutil`以正确安装psutil。
具体原因可以参照该[stackoverflow帖子](https://stackoverflow.com/questions/72619143/unable-to-import-psutil-on-m1-mac-with-miniforge-mach-o-file-but-is-an-incomp)。

<br/>

### Q: ARM版macOS在安装MindSpore时，安装依赖库scipy报错 `error: metadata-generation-failed` 怎么办？

A: 对应ARM版macOS的scipy在pypi源的版本仅适配MacOS 12以上版本的操作系统，低版本操作系统会自动下载scipy源码包进行编译，大概率会遇到编译失败问题。如果使用Conda环境，建议执行`pip uninstall scipy; conda install scipy`，MacOS 11系统强烈建议使用Conda以规避类似的兼容性问题。

<br/>

### Q: 安装MindSpore版本: GPU、CUDA 10.1、0.5.0-beta，出现问题: `cannot open shared object file:No such file or directory`。

A: 从报错情况来看，是cuBLAS库没有找到。一般的情况下是cuBLAS库没有安装，或者是因为没有加入到环境变量中去。通常cuBLAS是随着CUDA以及驱动一起安装的，确认安装后把cuBLAS所在的目录加入`LD_LIBRARY_PATH`环境变量中即可。

<br/>

## Conda安装

### Q: Ascend硬件平台，在个人的Conda环境中，有时候出现报错RuntimeError: json.exception.parse_error.101 parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'T'，该怎么处理？

A: 出现这种类型的报错，大概率是昇腾AI处理器配套软件包更新后个人的Conda环境中没有更新te或hccl工具包，可以将当前Conda环境中的上述几个工具包卸载，然后执行软件包中提供的环境变量配置脚本，即可配置PYTHONPATH，自动找到安装目录中上述软件包的最新版本，无需手动安装到Conda环境: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。

<br/>

### Q: ARM版macOS在Python3.9环境中安装mindspore包执行报错 `ImportError: dlopen ... Library not loaded: @rpath/libffi.7.dylib`怎么办？

A: ARM版macOS上配置python3.9的conda环境时，在部分设备上没有自动安装libffi，导致该问题出现。手动执行安装命令`conda install libffi`即可修复。

<br/>

## Source安装

### Q: 在Linux中已经安装了交叉编译工具，但是编译命令要怎么写呢？

A: arm64版本编译: `bash build.sh -I arm64`；arm32版本编译: `bash build.sh -I arm32`；注意要先设置环境变量，指定Android NDK路径: `export ANDROID_NDK=/path/to/android-ndk`，编译成功后，在output目录可以找到编译出的包。

<br/>

### Q: 源码编译MindSpore过程时间过长，或时常中断该怎么办？

A: MindSpore通过submodule机制引入第三方依赖包，其中`Protobuf`依赖包（v3.13.0）下载速度不稳定，建议您提前进行包缓存。

<br/>

### Q: 源码编译时，报错`MD5 does not match`，应该怎么办？

A: 这种报错可能是在编译的时候由于网络问题导致一些第三方库下载中断，之后重新编译的时候，该文件已经存在但是不完整，在校验MD5的时候失败。解决方法是: 删除.mslib缓存路径中的相关第三方库，然后重新编译。

<br/>

### Q: 环境上安装了Python3.7.5，环境变量设置正确，编译MindSpore时仍然报错`Python3 not found`，应该怎么办？

A: 可能是因为当前环境上的Python未包含动态库。编译MindSpore需要动态链接Python库，因此需要使用开启动态库编译选项的Python3.7.5，即在源码编译Python时使用`./configure --enable-shared`命令。

<br/>

### Q: 如何改变第三方依赖库安装路径？

A: 第三方依赖库的包默认安装在build/mindspore/.mslib目录下，可以设置环境变量MSLIBS_CACHE_PATH来改变安装目录，比如 `export MSLIBS_CACHE_PATH = ~/.mslib`。

<br/>

### Q: 编译失败后，应该清理哪些路径以确保上次失败的编译结果不会影响到下一次编译？

A: 在编译MindSpore时，如果:

1. 第三方组件下载或编译失败，例如icu4c的patch动作失败返回错误信息`Cmake Error at cmake/utils.cmake:301 (message): Failed patch:`，则进入编译目录下的`build/mindspore/.mslib`目录，或由`MSLIBS_CACHE_PATH`环境变量指定的第三方软件安装目录，并删除其中的对应软件。

2. 其他阶段编译失败，或打算删除上一次编译结果，完全重新编译时，直接删除`build`目录即可。

<br/>

### Q: 编译时报错，打开CMakeError.txt提示pthread找不到怎么办？

A: 真正的失败原因会体现在打屏的日志里，CMakeError.txt报错内容一般情况下与编译失败的真正原因无关，请寻找打屏日志中的第一个报错。

<br/>

### Q: 编译成功后，运行时报错`undefined reference to XXXX`或`undefined symbol XXXX`怎么办？

A: 可能的原因有:

1. 如果问题是`git pull`更新代码后出现，请删除掉`build`文件夹，排除前次构建的影响。

2. 如果问题是修改代码后出现，可以使用`c++filt XXXX`查看该符号的意义，有可能是函数未实现、虚函数未实现、依赖未链接等原因引起。

3. 如果问题发生在Ascend平台，排除上述原因后，很可能是由于MindSpore版本与CANN版本不匹配引起的，版本匹配关系参考[安装说明](https://www.mindspore.cn/install)。

<br/>

### Q: 编译完成时告警，`SetuptoolsDeprecationWarning: setup.py install is deprecated ...`怎么办？

A: Python的setuptools组件从58.3.0版本宣布放弃直接调用setup.py的使用方式，因此在安装了较高版本的setuptools的Python环境中源码编译时会遇到类似告警。该告警暂不影响MindSpore的使用，我们将在后续的版本中更换打包方式。
详情可以参照[setuptools版本记录](https://setuptools.pypa.io/en/latest/history.html#v58-3-0)

<br/>

### Q: MindSpore要求的配套软件版本与Ubuntu默认版本不一致怎么办？

A: 当前MindSpore只提供版本配套关系，需要您手动进行配套软件的安装升级。（**注明**: MindSpore要求Python3.7.5和gcc7.3，Ubuntu 16.04默认为Python3.5和gcc5，Ubuntu 18.04默认自带Python3.7.3和gcc7.4）。

<br/>

### Q: 执行用例报错`No module named 'mindspore.version'`，应该怎么办？

A: 当有这种报错时，有可能是在创建了和MindSpore安装包相同名字的路径中执行用例，导致Python导入包的时候优先找到了当前目录下，而当前目录没有version.py这个文件。解决方法就是目录重命名或者向上退出一级或者多级目录。

<br/>

### Q: MindSpore安装: 版本0.6.0-beta + Ascend + Ubuntu_aarch64 + Python3.7.5，手动下载对应版本的whl包，编译并安装gmp6.1.2。其他Python库依赖已经安装完成，执行样例失败，报错显示找不到so文件。

A: `libdatatransfer.so`动态库是`fwkacllib/lib64`目录下的，请先在`/usr/local`目录查找到这个库所在的路径，然后把这个路径加到`LD_LIBRARY_PATH`环境变量中，确认设置生效后，再执行。

<br/>

## Docker安装

### Q: MindSpore是否支持Nvidia GPU独立显卡+Windows操作系统的个人电脑？

A: 目前MindSpore支持的情况是GPU+Linux与CPU+Windows的组合配置，Windows+GPU的支持还在开发中。
如果希望在GPU+Windows的环境上运行，可以尝试使用WSL+docker的方式，操作思路:

1. 以WSL方式安装起Ubuntu18.04，参考[Install Linux on Windows with WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10)。
2. 安装支持WSL的Nvidia驱动以及在WSL运行容器的环境部署，参考[CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。

   > 由于CUDA on WSL还是预览特性，注意参考链接里对Windows版本要求的说明，版本不够的需要做升级。
3. 参考[Docker镜像](https://gitee.com/mindspore/mindspore#docker镜像)，取MindSpore-GPU镜像。如取MindSpore1.0.0版本容器，在WSL Ubuntu18.04中执行`docker pull mindspore/mindspore-gpu:1.0.0`运行容器:

    ```docker
    docker run -it --runtime=nvidia mindspore/mindspore-gpu:1.0.0 /bin/bash
    ```

详细步骤可以参考社区提供的实践[张小白教你安装Windows10的GPU驱动（CUDA和cuDNN）](https://www.hiascend.com/developer/blog/details/0235122004927955034)。
在此感谢社区成员[张辉](https://www.hiascend.com/forum/otheruser?uid=110545f972cf4903b3deba9cfaac7fc8)的分享。

<br/>

## 卸载

### Q: 如何卸载MindSpore？

A: 首先请确定MindSpore的全称，例如gpu版本的MindSpore，可以执行命令`pip uninstall mindspore-gpu`进行卸载。

<br/>

## 环境变量

### Q: 一些常用的环境变量设置，在新启动的终端窗口中需要重新设置，容易忘记应该怎么办？

A: 常用的环境变量设置写入到`~/.bash_profile` 或 `~/.bashrc`中，可让环境变量设置在新启动的终端窗口中立即生效。

<br/>

### Q: 使用GPU版本MindSpore时，如何设置`DEVICE_ID`环境变量？

A: MindSpore GPU模式一般无需设置`DEVICE_ID`环境变量，MindSpore会根据cuda环境变量`CUDA_VISIBLE_DEVICES`，自动选择可见的GPU设备。设置`CUDA_VISIBLE_DEVICES`环境变量后，则`DEVICE_ID`环境变量代表可见GPU设备的下标:

- 执行`export CUDA_VISIBLE_DEVICES=1,3,5`后，`DEVICE_ID`应当被设置为`0`，`1`或`2`，若设置为`3`及以上，MindSpore会由于设备ID不合法而运行失败。

<br/>

### Q: 编译应用时报错`/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found`怎么办？

A: 寻找缺少的动态库文件所在目录，添加该路径到环境变量`LD_LIBRARY_PATH`中。

<br/>

### Q: 运行应用时出现`ModuleNotFoundError: No module named 'te'`怎么办？

A: 首先确认环境安装是否正确，`te`、`topi`等whl包是否正确安装。如果用户环境中有多个Python版本，如Conda虚拟环境中，需`ldd name_of_your_executable_app`确认应用所链接的`libpython3.so`是否与当前Python路径一致，如果不一致需要调整环境变量`LD_LIBRARY_PATH`顺序，例如

```bash
export LD_LIBRARY_PATH=`python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`:$LD_LIBRARY_PATH
```

将当前的`python`命令对应程序的运行库路径加入到`LD_LIBRARY_PATH`的最前面。

<br/>

### Q: 运行应用时出现`error while loading shared libraries: libpython3.so: cannot open shared object file: No such file or directory`怎么办？

A: 该报错通常出现在装有多个Python版本的环境中，首先确认Python的`lib`目录是否在环境变量`LD_LIBRARY_PATH`中，可执行以下命令进行设置：

```bash
export LD_LIBRARY_PATH=`python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`:$LD_LIBRARY_PATH
```

另外如果在Conda虚拟环境中，Python 3.7.6以下版本不含该动态库，可以执行命令升级Conda中Python的版本，如：`conda install python=3.7.11`。

<br/>

### Q: Ascend AI处理器配套软件包与其他依赖软件已安装，但是执行MindSpore时提示`Cannot open shared objectfile: No such file or directory`该怎么办？

A: 常见原因有两种: Ascend AI处理器配套软件包或固件/驱动包版本不正确，或没有安装在默认位置且未配置相应的环境变量。

1. 打开Ascend AI处理器配套软件包安装目录，默认`/usr/local/Ascend`下，各个子目录中的`version.info`文件，观察其版本号是否与当前使用的MindSpore版本一直，参照[安装页面](https://www.mindspore.cn/install/)中关于Ascend AI处理器配套软件包版本的描述。如果版本不配套，请更换软件包或MindSpore版本。

2. 检查Ascend AI处理器配套软件包与其他依赖软件是否安装在默认位置，MindSpore会尝试从默认安装位置`/usr/local/Ascend`自动加载，如果将Ascend软件包安装在自定义位置，请参照[安装页面](https://www.mindspore.cn/install/)页面的安装指南一栏设置环境变量。如果将其他依赖软件安装在自定义位置，请根据其位置关系设置`LD_LIBRARY_PATH`环境变量。

<br/>

### Q: 同一环境安装多个昇腾AI处理器配套软件包时，运行MindSpore出现卡死、segmentation fault、cannot open shared object file: No such file or directory等问题时该怎么办？

A: MindSpore 2.4.0之前的版本通过RPATH指定环境变量，可能导致同时安装CANN-nnae和CANN-toolkit包的环境出现二进制加载顺序错误问题，MindSpore 2.4.0已经删除该限制并要求用户手动设置环境变量。如果需要同时安装上述CANN软件包，请升级MindSpore到2.4.0或更高版本，通过CANN提供的环境变量配置脚本设置环境变量。请参照[安装页面](https://www.mindspore.cn/install/)进行环境变量设置。

<br/>

## 安装验证

### Q: MindSpore的GPU版本对设备的计算能力有限制吗？

A: 目前MindSpore仅支持计算能力大于5.3的设备。

<br/>

### Q: 个人电脑CPU环境安装MindSpore后验证代码时报错: `the pointer[session] is null`，具体代码如下，该如何验证是否安装成功呢？

```python
import numpy as np
import mindspore as ms
import mindspore.ops as ops

ms.set_device(device_target="Ascend")
x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.add(x,y))
```

A: 验证安装是否成功，可以参照对应版本的安装指南页面中 `验证是否成功安装` 段落的描述：

```bash
python -c "import mindspore;mindspore.run_check()"
```

如果输出：

```text
MindSpore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。

<br/>

### Q: `Linux`平台下执行用例的时候会报错`sh:1:python:not found`或者由于链接到了Python2.7的版本中而报错`No module named mindspore._extends.remote`，该怎么处理？

A: 遇到类似的问题，大多是由于Python的环境问题，可以通过如下方式检查Python环境是否是MindSpore运行时所需要的环境。

- 在终端窗口中输入`python`，检查以下进入Python交互环境中的版本信息，如果直接报错则是没有Python的软连接；如果进入的是非Python3.7版本的环境，则当前Python环境不是MindSpore运行所需要的。
- 执行`sudo ln -sf /usr/bin/python3.7.x /usr/bin/python`创建Python的软连接，然后再检查执行。

<br/>

### Q: 在脚本中`import mindspore`之前import了其他三方库，提示如下错误(`/{your_path}/libgomp.so.1: cannot allocate memory in static TLS block`)该怎么解决?

A: 上述问题较为常见，当前有两种可行的解决方法，可任选其一:

- 交换import的顺序，先`import mindspore`再import其他三方库。
- 执行程序之前先添加环境变量（`export LD_PRELOAD=/{your_path}/libgomp.so.1`），其中`{your_path}`是上述报错提示的路径。

<br/>

### Q: MindSpore和gmp都已经通过源码编译安装后，在脚本中执行`import mindspore`，提示如下错误(`ImportError: libgmpxx.so: cannot open shared object file: No such file or directory`)该怎么解决?

A: 上述问题的原因是在编译安装gmp库的时候没有设置`--enable-cxx`，正确的gmp编译安装方式如下（假设已经下载了gmp6.1.2安装包）：

```bash
$cd gmp-6.1.2
$./configure --enable-cxx
$make
$make check
$sudo make install
```

<br/>

### Q: 运行MindSpore时出现告警 `UserWarning: The value of the smallest subnormal for <class 'numpy.float64'> type is zero.` 应该怎么解决？

A: 上述问题出现在安装了较新版本的numpy(>=1.22.0)版本的ARM python3.9环境上。告警来自numpy而非MindSpore。如果告警影响到了代码的正常调测，可以考虑手动安装较低版本的numpy(<=1.21.2)来规避。

<br/>

### Q: 运行MindSpore时出现报错 `AttributeError: module 'six' has no attribute 'ensure_text'` 应该怎么解决？

A: 上述问题的原因是环境安装了较新版本的`asttokens`(>=2.0.6)，其依赖的`six`版本不匹配。更新`six`版本(>=1.12.0)即可解决。

<br/>

### Q: 运行MindSpore时出现报错 `ModuleNotFoundError: No module named 'requests'` 应该怎么解决？

A: 上述问题的原因是当前python环境缺少requests包，pip install requests即可解决。
