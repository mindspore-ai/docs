# 安装类

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<!-- TOC -->

- [安装类](#安装类)
    - [pip安装](#pip安装)
    - [源码编译安装](#源码编译安装)
    - [卸载](#卸载)
    - [环境变量](#环境变量)
    - [安装验证](#安装验证)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/installation.md" target="_blank"><img src="./_static/logo_source.png"></a>

## pip安装

Q：安装MindSpore版本：GPU、CUDA 10.1、0.5.0-beta、Ubuntu-x86，出现问题：`cannot open shared object file:file such file or directory`。

A：从报错情况来看，是cublas库没有找到。一般的情况下是cublas库没有安装，或者是因为没有加入到环境变量中去。通常cublas是随着cuda以及驱动一起安装的，确认安装后把cublas所在的目录加入`LD_LIBRARY_PATH`环境变量中即可。

<br/>

Q：使用pip安装时报错：`SSL:CERTIFICATE_VERIFY_FATLED`应该怎么办？

A：在pip安装命令后添加参数 `--trusted-host=ms-release.obs.cn-north-4.myhuaweicloud.com`重试即可。

<br/>

Q：pip安装MindSpore对Python版本是否有特别要求？

A：MindSpore开发过程中用到了Python3.7+的新特性，因此建议您通过`conda`工具添加Python3.7.5的开发环境。

<br/>

Q：使用pip安装时报错`ProxyError(Cannot connect to proxy)`，应该怎么办？

A：此问题一般是代理配置问题，Ubuntu环境下可通过`export http_proxy={your_proxy}`设置代理；Windows环境可以在cmd中通过`set http_proxy={your_proxy}`进行代理设置。

<br/>

Q：使用pip安装时提示错误，应该怎么办？

A：请执行`pip -V`查看是否绑定了Python3.7+。如果绑定的版本不对，建议使用`python3.7 -m pip install`代替`pip install`命令。

<br/>

Q：MindSpore网站安装页面找不到MindInsight和MindArmour的whl包，无法安装怎么办？

A：您可以从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，通过`pip install`命令进行安装。

<br/>

## 源码编译安装

Q：MindSpore安装：版本0.6.0-beta + Ascend 910 + Ubuntu_aarch64 + Python3.7.5，手动下载对应版本的whl包，编译并安装gmp6.1.2。其他Python库依赖已经安装完成，执行样例失败，报错显示找不到so文件。

A：`libdatatransfer.so`动态库是`fwkacllib/lib64`目录下的，请先在`/usr/local`目录find到这个库所在的路径，然后把这个路径加到`LD_LIBRARY_PATH`环境变量中，确认设置生效后，再执行。

<br/>

Q：源码编译MindSpore过程时间过长，或时常中断该怎么办？

A：MindSpore通过submodule机制引入第三方依赖包，其中`protobuf`依赖包（v3.8.0）下载速度不稳定，建议您提前进行包缓存。

<br/>

Q：如何改变第三方依赖库安装路径？

A：第三方依赖库的包默认安装在build/mindspore/.mslib目录下，可以设置环境变量MSLIBS_CACHE_PATH来改变安装目录，比如 `export MSLIBS_CACHE_PATH = ~/.mslib`。

<br/>

Q：MindSpore要求的配套软件版本与Ubuntu默认版本不一致怎么办？

A：当前MindSpore只提供版本配套关系，需要您手动进行配套软件的安装升级。（**注明**：MindSpore要求Python3.7.5和gcc7.3，Ubuntu 16.04默认为Python3.5和gcc5，Ubuntu 18.04默认自带Python3.7.3和gcc7.4）。

<br/>

Q：当源码编译MindSpore，提示`tclsh not found`时，应该怎么办？

A：当有此提示时说明要用户安装`tclsh`；如果仍提示缺少其他软件，同样需要安装其他软件。

<br/>

## 卸载

Q：如何卸载MindSpore？

A：执行命令`pip uninstall mindspore`可卸载MindSpore。

<br/>

## 环境变量

Q：一些常用的环境变量设置，在新启动的终端窗口中需要重新设置，容易忘记应该怎么办？

A：常用的环境变量设置写入到`~/.bash_profile` 或 `~/.bashrc`中，可让环境变量设置在新启动的终端窗口中立即生效。

<br/>

## 安装验证

Q:个人电脑CPU环境安装MindSpore后验证代码时报错：`the pointer[session] is null`，具体代码如下，该如何验证是否安装成功呢？

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