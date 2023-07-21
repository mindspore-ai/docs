# 安装类

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/tree/r1.0/docs/faq/source_zh_cn/installation.md)

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

详细步骤可以参考社区提供的实践[张小白GPU安装MindSpore给你看（Ubuntu 18.04.5）](https://bbs.huaweicloud.com/blogs/198357)。
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

## 卸载

<font size=3>**Q：如何卸载MindSpore？**</font>

A：执行命令`pip uninstall mindspore`可卸载MindSpore。

<br/>

## 环境变量

<font size=3>**Q：一些常用的环境变量设置，在新启动的终端窗口中需要重新设置，容易忘记应该怎么办？**</font>

A：常用的环境变量设置写入到`~/.bash_profile` 或 `~/.bashrc`中，可让环境变量设置在新启动的终端窗口中立即生效。

<br/>

## 安装验证

<font size=3>**Q:个人电脑CPU环境安装MindSpore后验证代码时报错：`the pointer[session] is null`，具体代码如下，该如何验证是否安装成功呢？**</font>

```python
import numpy as np
from mindspore import Tensor
imort mindspore.ops as ops
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(ops.tensor_add(x,y))
```

A：CPU硬件平台安装MindSpore后测试是否安装成功,只需要执行命令：`python -c 'import mindspore'`，如果没有显示`No module named 'mindspore'`等错误即安装成功。问题中的验证代码仅用于验证Ascend平台安装是否成功。