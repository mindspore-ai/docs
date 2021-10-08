# 安装

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<!-- TOC -->

- [安装](#安装)
    - [pip安装](#pip安装)
    - [源码编译安装](#源码编译安装)
    - [卸载](#卸载)
    - [环境变量](#环境变量)
    - [安装验证](#安装验证)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/faq/source_zh_cn/installation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## pip安装

<font size=3>**Q: 安装MindSpore版本: GPU、CUDA 10.1、0.5.0-beta，出现问题: `cannot open shared object file:No such file or directory`。**</font>

A: 从报错情况来看，是cuBLAS库没有找到。一般的情况下是cuBLAS库没有安装，或者是因为没有加入到环境变量中去。通常cuBLAS是随着CUDA以及驱动一起安装的，确认安装后把cuBLAS所在的目录加入`LD_LIBRARY_PATH`环境变量中即可。

<br/>

<font size=3>**Q: 使用pip安装时报错: `ERROR: mindspore_{VERSION}.whl is not a supported wheel on this platform`应该怎么办？**</font>

A: pip会通过wheel安装包的文件名来判断该安装包是否与当前Python环境兼容，例如安装mindspore_ascend-1.2.0-cp37-cp37m-linux_aarch64.whl时，pip会检查:

1. 当前python环境为3.7.x版本
2. 当前操作系统为Linux
3. 操作系统架构为arm64

因此，如果出现is not a supported wheel on this platform问题，请检查当前环境是否满足MindSpore安装要求，以及该MindSpore安装包版本是否正确。

<br/>

<font size=3>**Q: 使用pip安装时报错: `SSL:CERTIFICATE_VERIFY_FATLED`应该怎么办？**</font>

A: 在pip安装命令后添加参数 `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com`重试即可。

<br/>

<font size=3>**Q: pip安装MindSpore对Python版本是否有特别要求？**</font>

A: MindSpore开发过程中用到了Python3.7+的新特性，因此建议您通过`conda`工具添加Python3.7.5的开发环境。

<br/>

<font size=3>**Q: MindSpore对Protobuf版本是否有特别要求？**</font>

A: MindSpore默认安装Protobuf的3.13.0版本，如果不是该版本，在使用pytest测试代码时日志中会产生很多告警，建议您使用命令`pip install protobuf==3.13.0`重新安装3.13.0版本。

<br/>

<font size=3>**Q: 使用pip安装时报错`ProxyError(Cannot connect to proxy)`，应该怎么办？**</font>

A: 此问题一般是代理配置问题，Ubuntu环境下可通过`export http_proxy={your_proxy}`设置代理；Windows环境可以在cmd中通过`set http_proxy={your_proxy}`进行代理设置。

<br/>

<font size=3>**Q: 使用pip安装时提示错误，应该怎么办？**</font>

A: 请执行`pip -V`查看是否绑定了Python3.7+。如果绑定的版本不对，建议使用`python3.7 -m pip install`代替`pip install`命令。

<br/>

<font size=3>**Q: 使用pip安装依赖库时提示`No matching distribution found for XXX`错误，应该怎么办？**</font>

A: 请执行`pip config list`，查看当前软件库索引路径`index-url`。某些情况下，软件库索引会出现更新滞后，可尝试设置其它软件库索引路径。

<br/>

<font size=3>**Q: MindSpore网站安装页面找不到MindInsight和MindArmour的whl包，无法安装怎么办？**</font>

A: 您可以从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，通过`pip install`命令进行安装。

<br/>

<font size=3>**Q: MindSpore是否支持Nvidia GPU独立显卡+Windows操作系统的个人电脑？**</font>

A: 目前MindSpore支持的情况是GPU+Linux与CPU+Windows的组合配置，Windows+GPU的支持还在开发中。
如果希望在GPU+Windows的环境上运行，可以尝试使用WSL+docker的方式，操作思路:

1. 以WSL方式安装起Ubuntu18.04，参考<https://docs.microsoft.com/en-us/windows/wsl/install-win10>。
2. 安装支持WSL的Nvidia驱动以及在WSL运行容器的环境部署，参考<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>。

   > 由于CUDA on WSL还是预览特性，注意参考链接里对Windows版本要求的说明，版本不够的需要做升级。
3. 参考<https://gitee.com/mindspore/mindspore#docker%E9%95%9C%E5%83%8F>，取MindSpore-GPU镜像。如取MindSpore1.0.0版本容器，在WSL Ubuntu18.04中执行`docker pull mindspore/mindspore-gpu:1.0.0`运行容器:

    ```docker
    docker run -it --runtime=nvidia mindspore/mindspore-gpu:1.0.0 /bin/bash
    ```

详细步骤可以参考社区提供的实践[张小白教你安装Windows10的GPU驱动（CUDA和cuDNN）](https://bbs.huaweicloud.com/blogs/212446)。
在此感谢社区成员[张辉](https://bbs.huaweicloud.com/community/usersnew/id_1552550689252345)的分享。

<br/>

<font size=3>**Q: Ascend硬件平台，在个人的Conda环境中，有时候出现报错RuntimeError: json.exception.parse_error.101 parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'T'，该怎么处理？**</font>

A: 出现这种类型的报错，大概率是run包更新后个人的Conda环境中没有更新te或topi或hccl工具包，可以将当前Conda环境中的上述几个工具包卸载，然后使用如下命令再重新安装: `pip install /usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64/{te/topi/hccl}-{version}-py3-none-any.whl`。

<br/>

<font size=3>**Q: pip同时安装MindSpore CPU和GPU版本，import时报错 `cannot import name 'context' from 'mindspore'`，应该怎么办？**</font>

A: MindSpore不同版本的安装目录名同为`mindspore`，安装在同一个Python环境可能产生目录相互覆盖问题，导致无法使用，如果需要使用多平台版本的MindSpore时（例如同时使用CPU和GPU版本），请先卸载其他版本再安装新版本。

<br/>

## 源码编译安装

<font size=3>**Q: 编译时`bash -p`方式和 `bash -e`方式的区别？**</font>

A: MindSpore Serving的编译和运行依赖MindSpore，Serving提供两种编译方式: 一种指定已安装的MindSpore路径，即`bash -p {python site-packages}/mindspore/lib`，避免编译Serving时再编译MindSpore；另一种，编译Serving时，编译配套的MindSpore，Serving会将`-e`、`-V`和`-j`选项透传给MindSpore。
比如，在Serving目录下，`bash -e ascend -V 910 -j32`:

- 首先将会以`bash -e ascend -V 910 -j32`方式编译`third_party/mindspore`目录下的MindSpore；
- 其次，编译脚本将MindSpore编译结果作为Serving的编译依赖。

<br/>

<font size=3>**Q: 在Linux中已经安装了交叉编译工具，但是编译命令要怎么写呢？**</font>

A: arm64版本编译: `bash build.sh -I arm64`；arm32版本编译: `bash build.sh -I arm32`；注意要先设置环境变量，指定Android NDK路径: `export ANDROID_NDK=/path/to/android-ndk`，编译成功后，在output目录可以找到编译出的包。

<br/>

<font size=3>**Q: MindSpore安装: 版本0.6.0-beta + Ascend 910 + Ubuntu_aarch64 + Python3.7.5，手动下载对应版本的whl包，编译并安装gmp6.1.2。其他Python库依赖已经安装完成，执行样例失败，报错显示找不到so文件。**</font>

A: `libdatatransfer.so`动态库是`fwkacllib/lib64`目录下的，请先在`/usr/local`目录查找到这个库所在的路径，然后把这个路径加到`LD_LIBRARY_PATH`环境变量中，确认设置生效后，再执行。

<br/>

<font size=3>**Q: 源码编译MindSpore过程时间过长，或时常中断该怎么办？**</font>

A: MindSpore通过submodule机制引入第三方依赖包，其中`Protobuf`依赖包（v3.13.0）下载速度不稳定，建议您提前进行包缓存。

<br/>

<font size=3>**Q: 如何改变第三方依赖库安装路径？**</font>

A: 第三方依赖库的包默认安装在build/mindspore/.mslib目录下，可以设置环境变量MSLIBS_CACHE_PATH来改变安装目录，比如 `export MSLIBS_CACHE_PATH = ~/.mslib`。

<br/>

<font size=3>**Q: MindSpore要求的配套软件版本与Ubuntu默认版本不一致怎么办？**</font>

A: 当前MindSpore只提供版本配套关系，需要您手动进行配套软件的安装升级。（**注明**: MindSpore要求Python3.7.5和gcc7.3，Ubuntu 16.04默认为Python3.5和gcc5，Ubuntu 18.04默认自带Python3.7.3和gcc7.4）。

<br/>

<font size=3>**Q: 当源码编译MindSpore，提示`tclsh not found`时，应该怎么办？**</font>

A: 当有此提示时说明要用户安装`tclsh`；如果仍提示缺少其他软件，同样需要安装其他软件。

<br/>

<font size=3>**Q: 执行用例报错`No module named 'mindpore.version'`，应该怎么办？**</font>

A: 当有这种报错时，有可能是在创建了和MindSpore安装包相同名字的路径中执行用例，导致Python导入包的时候优先找到了当前目录下，而当前目录没有version.py这个文件。解决方法就是目录重命名或者向上退出一级或者多级目录。

<br/>

<font size=3>**Q: 源码编译时，报错`MD5 does not match`，应该怎么办？**</font>

A: 这种报错可能是在编译的时候由于网络问题导致一些第三方库下载中断，之后重新编译的时候，该文件已经存在但是不完整，在校验MD5的时候失败。解决方法是: 删除.mslib缓存路径中的相关第三方库，然后重新编译。

<br/>

<font size=3>**Q: 环境上安装了Python3.7.5，环境变量设置正确，编译MindSpore时仍然报错`Python3 not found`，应该怎么办？**</font>

A: 可能是因为当前环境上的Python未包含动态库。编译MindSpore需要动态链接Python库，因此需要使用开启动态库编译选项的Python3.7.5，即在源码编译Python时使用`./configure --enable-shared`命令。

<br/>

<font size=3>**Q: 编译失败后，应该清理哪些路径以确保上次失败的编译结果不会影响到下一次编译？**</font>

A: 在编译MindSpore时，如果:

1. 第三方组件下载或编译失败，例如icu4c的patch动作失败返回错误信息`Cmake Error at cmake/utils.cmake:301 (message): Failed patch:`，则进入编译目录下的`build/mindspore/.mslib`目录，或由`MSLIBS_CACHE_PATH`环境变量指定的第三方软件安装目录，并删除其中的对应软件。

2. 其他阶段编译失败，或打算删除上一次编译结果，完全重新编译时，直接删除`build`目录即可。

<br/>

<font size=3>**Q: 编译时报错，打开CMakeError.txt提示pthread找不到怎么办？**</font>

A: 真正的失败原因会体现在打屏的日志里，CMakeError.txt无参考价值，请寻找打屏日志中的第一个报错。

<br/>

<font size=3>**Q: 编译成功后，运行时报错`undefined reference to XXXX`或`undefined symbol XXXX`怎么办？**</font>

A: 可能的原因有:

1. 如果问题是`git pull`更新代码后出现，请删除掉`build`文件夹，排除前次构建的影响。

2. 如果问题是修改代码后出现，可以使用`c++filt XXXX`查看该符号的意义，有可能是函数未实现、虚函数未实现、依赖未链接等原因引起。

3. 如果问题发生在Ascend平台，排除上述原因后，很可能是由于MindSpore版本与CANN版本不匹配引起的，版本匹配关系参考[安装说明](https://www.mindspore.cn/install)。

<br/>

<font size=3>**Q: 编译应用时报错`bash -p`方式和 `bash -e`方式的区别？**</font>

A: MindSpore Serving的编译和运行依赖MindSpore，Serving提供两种编译方式: 一种指定已安装的MindSpore路径，即`bash -p {python site-packages}/mindspore/lib`，避免编译Serving时再编译MindSpore；另一种，编译Serving时，编译配套的MindSpore，Serving会将`-e`、`-V`和`-j`选项透传给MindSpore。
比如，在Serving目录下，`bash -e ascend -V 910 -j32`:

- 首先将会以`bash -e ascend -V 910 -j32`方式编译`third_party/mindspore`目录下的MindSpore；
- 其次，编译脚本将MindSpore编译结果作为Serving的编译依赖。

<br/>

## 卸载

<font size=3>**Q: 如何卸载MindSpore？**</font>

A: 首先请确定MindSpore的全称，例如gpu版本的MindSpore，可以执行命令`pip uninstall mindspore-gpu`进行卸载。

<br/>

## 环境变量

<font size=3>**Q: 一些常用的环境变量设置，在新启动的终端窗口中需要重新设置，容易忘记应该怎么办？**</font>

A: 常用的环境变量设置写入到`~/.bash_profile` 或 `~/.bashrc`中，可让环境变量设置在新启动的终端窗口中立即生效。

<br/>

<font size=3>**Q: 使用GPU版本MindSpore时，如何设置`DEVICE_ID`环境变量**</font>

A: MindSpore GPU模式一般无需设置`DEVICE_ID`环境变量，MindSpore会根据cuda环境变量`CUDA_VISIBLE_DEVICES`，自动选择可见的GPU设备。设置`CUDA_VISIBLE_DEVICES`环境变量后，则`DEVICE_ID`环境变量代表可见GPU设备的下标:

- 执行`export CUDA_VISIBLE_DEVICES=1,3,5`后，`DEVICE_ID`应当被设置为`0`，`1`或`2`，若设置为`3`及以上，MindSpore会由于设备ID不合法而运行失败。

<br/>

<font size=3>**Q: 编译应用时报错`/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found`怎么办？**</font>

A: 寻找缺少的动态库文件所在目录，添加该路径到环境变量`LD_LIBRARY_PATH`中，环境变量设置参考[Ascend 310 AI处理器上使用MindIR模型进行推理#编译推理代码](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_ascend_310_mindir.html#id6)。

<br/>

<font size=3>**Q: 运行应用时出现`ModuleNotFoundError: No module named 'te'`怎么办？**</font>

A: 首先确认环境安装是否正确，`te`、`topi`等whl包是否正确安装。如果用户环境中有多个Python版本，如Conda虚拟环境中，需`ldd name_of_your_executable_app`确认应用所链接的`libpython3.so`是否与当前Python路径一致，如果不一致需要调整环境变量`LD_LIBRARY_PATH`顺序，例如

```bash
export LD_LIBRARY_PATH=`python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))"`:$LD_LIBRARY_PATH
```

将当前的`python`命令对应程序的运行库路径加入到`LD_LIBRARY_PATH`的最前面。

<br/>

<font size=3>**Q: Ascend AI处理器配套软件包与其他依赖软件已安装，但是执行MindSpore时提示`Cannot open shared objectfile: No such file or directory`该怎么办？**</font>

A: 常见原因有两种: Ascend AI处理器配套软件包或固件/驱动包版本不正确，或没有安装在默认位置且未配置相应的环境变量。

1. 打开Ascend AI处理器配套软件包安装目录，默认`/usr/local/Ascend`下，各个子目录中的`version.info`文件，观察其版本号是否与当前使用的MindSpore版本一直，参照[安装页面](https://www.mindspore.cn/install/)中关于Ascend AI处理器配套软件包版本的描述。如果版本不配套，请更换软件包或MindSpore版本。

2. 检查Ascend AI处理器配套软件包与其他依赖软件是否安装在默认位置，MindSpore会尝试从默认安装位置`/usr/local/Ascend`自动加载，如果将Ascend软件包安装在自定义位置，请参照[安装页面](https://www.mindspore.cn/install/)页面的安装指南一栏设置环境变量。如果将其他依赖软件安装在自定义位置，请根据其位置关系设置`LD_LIBRARY_PATH`环境变量。

<br/>

## 安装验证

<font size=3>**Q: MindSpore的GPU版本对设备的计算能力有限制吗？**</font>

A: 目前MindSpore仅支持计算能力大于5.3的设备。

<br/>

<font size=3>**Q: 个人电脑CPU环境安装MindSpore后验证代码时报错: `the pointer[session] is null`，具体代码如下，该如何验证是否安装成功呢？**</font>

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

A: CPU硬件平台安装MindSpore后测试是否安装成功,只需要执行命令: `python -c 'import mindspore'`，如果没有显示`No module named 'mindspore'`等错误即安装成功。问题中的验证代码仅用于验证Ascend平台安装是否成功。

<br/>

<font size=3>**Q: `Linux`平台下执行用例的时候会报错`sh:1:python:not found`或者由于链接到了Python2.7的版本中而报错`No module named mindspore._extends.remote`，该怎么处理？**</font>

A: 遇到类似的问题，大多是由于Python的环境问题，可以通过如下方式检查Python环境是否是MindSpore运行时所需要的环境。

- 在终端窗口中输入`python`，检查以下进入Python交互环境中的版本信息，如果直接报错则是没有Python的软连接；如果进入的是非Python3.7版本的环境，则当前Python环境不是MindSpore运行所需要的。
- 执行`sudo ln -sf /usr/bin/python3.7.x /usr/bin/python`创建Python的软连接，然后再检查执行。

<br/>

<font size=3>**Q: 在脚本中`import mindspore`之前import了其他三方库，提示如下错误(`/{your_path}/libgomp.so.1: cannot allocate memory in static TLS block`)该怎么解决?**</font>

A: 上述问题较为常见，当前有两种可行的解决方法，可任选其一:

- 交换import的顺序，先`import mindspore`再import其他三方库。
- 执行程序之前先添加环境变量（`export LD_PRELOAD=/{your_path}/libgomp.so.1`），其中`{your_path}`是上述报错提示的路径。

<br/>

<font size=3>**Q: 训练nlp类网络，当使用第三方组件gensim时，可能会报错: ValueError，如何解决？**</font>

A: 以下为报错信息:

```bash
>>> import gensim
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/__init__.py", line 11, in <module>
    from gensim import parsing, corpora, matutils, interfaces, models, similarities, utils  # noqa:F401
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/corpora/__init__.py", line 6, in <module>
    from .indexedcorpus import IndexedCorpus  # noqa:F401 must appear before the other classes
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/corpora/indexedcorpus.py", line 14, in <module>
    from gensim import interfaces, utils
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/interfaces.py", line 19, in <module>
    from gensim import utils, matutils
  File "/home/miniconda3/envs/ci39_cj/lib/python3.9/site-packages/gensim/matutils.py", line 1024, in <module>
    from gensim._matutils import logsumexp, mean_absolute_difference, dirichlet_expectation
  File "gensim/_matutils.pyx", line 1, in init gensim._matutils
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
```

报错原因请参考[gensim](https://github.com/RaRe-Technologies/gensim/issues/3095)官网，或者[numpy](https://github.com/numpy/numpy/issues/18709)官网:

解决方案:

方法一: 重新安装numpy及gensim, 执行命令: `pip uninstall gensim numpy -y && pip install numpy gensim` ；

方法二: 如果还是有问题，请删除wheel安装包的缓存文件，然后执行方法一（wheel安装包缓存目录为: `~/.cache/pip/wheels`）。

<br/>

<font size=3>**Q: mindspore和gmp都已经通过源码编译安装后，在脚本中执行`import mindspore`，
提示如下错误(`ImportError: libgmpxx.so: cannot open shared object file: No such file or directory`)该怎么解决?**</font>

A: 上述问题的原因是在编译安装gmp库的时候没有设置`--enable-cxx`，正确的gmp编译安装方式如下（假设已经下载了gmp6.1.2安装包）：

```bash
$cd gmp-6.1.2
$./configure --enable-cxx
$make
$make check
$sudo make install
```