# 安装类

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_zh_cn/installation.md" target="_blank"><img src="./_static/logo_source.png"></a>

## pip安装

<font size=3>**Q：安装MindSpore版本：GPU、CUDA 10.1、0.5.0-beta、Ubuntu-x86，出现问题：`cannot open shared object file:file such file or directory`。**</font>

A：从报错情况来看，是cublas库没有找到。一般的情况下是cublas库没有安装，或者是因为没有加入到环境变量中去。通常cublas是随着cuda以及驱动一起安装的，确认安装后把cublas所在的目录加入`LD_LIBRARY_PATH`环境变量中即可。

<br/>

<font size=3>**Q：使用pip安装时报错：`SSL:CERTIFICATE_VERIFY_FATLED`应该怎么办？**</font>

A：在pip安装命令后添加参数 `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com`重试即可。

<br/>

<font size=3>**Q：pip安装MindSpore对Python版本是否有特别要求？**</font>

A：MindSpore开发过程中用到了Python3.7+的新特性，因此建议您通过`conda`工具添加Python3.7.5的开发环境。

<br/>

<font size=3>**Q：MindSpore对protobuf版本是否有特别要求？**</font>

A：MindSpore默认安装protobuf的3.8.0版本，如果您本地已安装protobuf的3.12.0或更高版本，在使用pytest测试代码时日志中会产生很多告警，建议您使用命令`pip install protobuf==3.8.0`重新安装3.8.0版本。

<br/>

<font size=3>**Q：使用pip安装时报错`ProxyError(Cannot connect to proxy)`，应该怎么办？**</font>

A：此问题一般是代理配置问题，Ubuntu环境下可通过`export http_proxy={your_proxy}`设置代理；Windows环境可以在cmd中通过`set http_proxy={your_proxy}`进行代理设置。

<br/>

<font size=3>**Q：使用pip安装时提示错误，应该怎么办？**</font>

A：请执行`pip -V`查看是否绑定了Python3.7+。如果绑定的版本不对，建议使用`python3.7 -m pip install`代替`pip install`命令。

<br/>

<font size=3>**Q：MindSpore网站安装页面找不到MindInsight和MindArmour的whl包，无法安装怎么办？**</font>

A：您可以从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，通过`pip install`命令进行安装。

<br/>

<font size=3>**Q：MindSpore是否支持Nvidia GPU独立显卡+Windows操作系统的个人电脑？**</font>

A：目前MindSpore支持的情况是GPU+Linux与CPU+Windows的组合配置，Windows+GPU的支持还在开发中。
如果希望在GPU+Windows的环境上运行，可以尝试使用WSL+docker的方式，操作思路：

1. 以WSL方式安装起Ubuntu18.04，参考<https://docs.microsoft.com/en-us/windows/wsl/install-win10>。
2. 安装支持WSL的Nvidia驱动以及在WSL运行容器的环境部署，参考<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>。

   > 由于CUDA on WSL还是预览特性，注意参考链接里对Windows版本要求的说明，版本不够的需要做升级。
3. 参考<https://gitee.com/mindspore/mindspore#docker%E9%95%9C%E5%83%8F>，取MindSpore-GPU镜像。如取MindSpore1.0.0版本容器，在WSL Ubuntu18.04中执行`docker pull mindspore/mindspore-gpu:1.0.0`运行容器：

    ```docker
    docker run -it --runtime=nvidia mindspore/mindspore-gpu:1.0.0 /bin/bash
    ```

详细步骤可以参考社区提供的实践[张小白教你安装Windows10的GPU驱动（CUDA和cuDNN）](https://bbs.huaweicloud.com/blogs/212446)。
在此感谢社区成员[张辉](https://bbs.huaweicloud.com/community/usersnew/id_1552550689252345)的分享。

<br />

## 源码编译安装

<font size=3>**Q：在Linux中已经安装了交叉编译工具，但是编译命令要怎么写呢？**</font>

A：arm64版本编译：`bash build.sh -I arm64`；arm32版本编译：`bash build.sh -I arm32`；注意要先设置环境变量，指定Android NDK路径：`export ANDROID_NDK=/path/to/android-ndk`，编译成功后，在output目录可以找到编译出的包。

<br/>

<font size=3>**Q：MindSpore安装：版本0.6.0-beta + Ascend 910 + Ubuntu_aarch64 + Python3.7.5，手动下载对应版本的whl包，编译并安装gmp6.1.2。其他Python库依赖已经安装完成，执行样例失败，报错显示找不到so文件。**</font>

A：`libdatatransfer.so`动态库是`fwkacllib/lib64`目录下的，请先在`/usr/local`目录find到这个库所在的路径，然后把这个路径加到`LD_LIBRARY_PATH`环境变量中，确认设置生效后，再执行。

<br/>

<font size=3>**Q：源码编译MindSpore过程时间过长，或时常中断该怎么办？**</font>

A：MindSpore通过submodule机制引入第三方依赖包，其中`protobuf`依赖包（v3.8.0）下载速度不稳定，建议您提前进行包缓存。

<br/>

<font size=3>**Q：如何改变第三方依赖库安装路径？**</font>

A：第三方依赖库的包默认安装在build/mindspore/.mslib目录下，可以设置环境变量MSLIBS_CACHE_PATH来改变安装目录，比如 `export MSLIBS_CACHE_PATH = ~/.mslib`。

<br/>

<font size=3>**Q：MindSpore要求的配套软件版本与Ubuntu默认版本不一致怎么办？**</font>

A：当前MindSpore只提供版本配套关系，需要您手动进行配套软件的安装升级。（**注明**：MindSpore要求Python3.7.5和gcc7.3，Ubuntu 16.04默认为Python3.5和gcc5，Ubuntu 18.04默认自带Python3.7.3和gcc7.4）。

<br/>

<font size=3>**Q：当源码编译MindSpore，提示`tclsh not found`时，应该怎么办？**</font>

A：当有此提示时说明要用户安装`tclsh`；如果仍提示缺少其他软件，同样需要安装其他软件。

<br/>

<font size=3>**Q：环境上安装了Python3.7.5，环境变量设置正确，编译MindSpore时仍然报错`Python3 not found`，应该怎么办？**</font>

A：可能是因为当前环境上的Python未包含动态库。编译MindSpore需要动态链接Python库，因此需要使用开启动态库编译选项的Python3.7.5，即在源码编译Python时使用`./configure --enable-shared`命令。

<br/>

<font size=3>**Q：编译失败后，应该清理哪些路径以确保上次失败的编译结果不会影响到下一次编译？**</font>

A：在编译MindSpore时，如果：

1. 第三方组件下载或编译失败，例如icu4c的patch动作失败返回错误信息`Cmake Error at cmake/utils.cmake:301 (message): Failed patch:`，则进入编译目录下的`build/mindspore/.mslib`目录，或由`MSLIBS_CACHE_PATH`环境变量指定的第三方软件安装目录，并删除其中的对应软件。

2. 其他阶段编译失败，或打算删除上一次编译结果，完全重新编译时，直接删除`build`目录即可。

<br/>

## 卸载

<font size=3>**Q：如何卸载MindSpore？**</font>

A：执行命令`pip uninstall mindspore`可卸载MindSpore。

<br/>

## 环境变量

<font size=3>**Q：一些常用的环境变量设置，在新启动的终端窗口中需要重新设置，容易忘记应该怎么办？**</font>

A：常用的环境变量设置写入到`~/.bash_profile` 或 `~/.bashrc`中，可让环境变量设置在新启动的终端窗口中立即生效。

<br/>

<font size=3>**Q：Ascend AI处理器配套软件包与其他依赖软件已安装，但是执行MindSpore时提示`Cannot open shared objectfile: No such file or directory`该怎么办？**</font>

A：常见原因有两种：Ascend AI处理器配套软件包或固件/驱动包版本不正确，或没有安装在默认位置且未配置相应的环境变量。

1. 打开Ascend AI处理器配套软件包安装目录，默认`/usr/local/Ascend`下，各个子目录中的`version.info`文件，观察其版本号是否与当前使用的MindSpore版本一直，参照[安装页面](https://www.mindspore.cn/install/)中关于Ascend AI处理器配套软件包版本的描述。如果版本不配套，请更换软件包或MindSpore版本。

2. 检查Ascend AI处理器配套软件包与其他依赖软件是否安装在默认位置，MindSpore会尝试从默认安装位置`/usr/local/Ascend`自动加载，如果将Ascend软件包安装在自定义位置，请参照[安装页面](https://www.mindspore.cn/install/)页面的安装指南一栏设置环境变量。如果将其他依赖软件安装在自定义位置，请根据其位置关系设置`LD_LIBRARY_PATH`环境变量。

<br/>

## 安装验证

<font size=3>**Q：个人电脑CPU环境安装MindSpore后验证代码时报错：`the pointer[session] is null`，具体代码如下，该如何验证是否安装成功呢？**</font>

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

A：CPU硬件平台安装MindSpore后测试是否安装成功,只需要执行命令：`python -c 'import mindspore'`，如果没有显示`No module named 'mindspore'`等错误即安装成功。问题中的验证代码仅用于验证Ascend平台安装是否成功。

<br/>

<font size=3>**Q：`Linux`平台下执行用例的时候会报错`sh:1:python:not found`或者由于链接到了Python2.7的版本中而报错`No module named mindspore._extends.remote`，该怎么处理？**</font>

A：遇到类似的问题，大多是由于Python的环境问题，可以通过如下方式检查Python环境是否是MindSpore运行时所需要的环境。

- 在终端窗口中输入`python`，检查以下进入Python交互环境中的版本信息，如果直接报错则是没有Python的软连接；如果进入的是非Python3.7版本的环境，则当前Python环境不是MindSpore运行所需要的。
- 执行`sudo ln -sf /usr/bin/python3.7.x /usr/bin/python`创建Python的软连接，然后再检查执行。

<br/>

<font size=3>**Q: 在脚本中`import mindspore`之前import了其他三方库，提示如下错误(`/your_path/libgomp.so.1: cannot allocate memory in static TLS block`)该怎么解决?**</font>

A: 上述问题较为常见，当前有两种可行的解决方法，可任选其一：

- 交换import的顺序，先`import mindspore`再import其他三方库。
- 执行程序之前先添加环境变量（`export LD_PRELOAD=/your_path/libgomp.so.1`），其中`your_path`是上述报错提示的路径。
