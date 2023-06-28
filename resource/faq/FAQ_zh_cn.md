# FAQ

## 安装

### pip安装

Q：pip安装MindSpore对Python版本是否有特别要求？

A：MindSpore开发过程中用到了Python3.7+的新特性，因此建议您通过`conda`工具添加Python3.7.5的开发环境。

<br/>

Q：使用pip安装时提示错误，应该怎么办？

A：请执行`pip -V`查看是否绑定了Python3.7+。如果绑定的版本不对，建议使用`python3.7 -m pip install`代替`pip install`命令。

<br/>

Q：MindSpore网站安装页面找不到MindInsight和MindArmour的whl包，无法安装怎么办？

A：您可以从[MindSpore网站下载地址](https://www.mindspore.cn/versions)下载whl包，通过`pip install`命令进行安装。

### 源码编译安装

Q：源码编译MindSpore过程时间过长，或时常中断该怎么办？

A：MindSpore通过submodule机制引入第三方依赖包，其中`protobuf`依赖包（v3.8.0）下载速度不稳定，建议您提前进行包缓存。

<br/>

Q：如何改变第三方依赖库安装路径？

A：第三方依赖库的包默认安装在build/mindspore/.mslib目录下，可以设置环境变量MSLIBS_CACHE_PATH来改变安装目录，比如 `export MSLIBS_CACHE_PATH = ~/.mslib`。

<br/>

Q：MindSpore要求的配套软件版本与Ubuntu默认版本不一致怎么办？

A：当前MindSpore只提供版本配套关系，需要您手动进行配套软件的安装升级。（**注明**：MindSpore要求Python3.7.5和gcc7.3，Ubuntu 16.04默认为Python3.5和gcc5，Ubuntu 18.04默认自带Python3.7.3和gcc7.4）

<br/>

Q：当源码编译MindSpore，提示`tclsh not found`时，应该怎么办？

A：当有此提示时说明要用户安装`tclsh`；如果仍提示缺少其他软件，同样需要安装其他软件。

## 支持

### 模型支持

Q：MindSpore支持哪些模型的训练？

A：MindSpore针对典型场景均有模型训练支持，支持情况详见[Release note](https://gitee.com/mindspore/mindspore/blob/r0.2/RELEASE.md)。

### 后端支持

Q：安装运行MindSpore时，是否要求平台有GPU、NPU等计算单元？

A：MindSpore当前支持CPU/GPU/NPU，重点支持Ascend AI处理器。CPU支持Lenet模型，您可以尝试使用。

<br/>

Q：针对异构计算单元的支持，MindSpore有什么计划？

A：MindSpore提供了可插拔式的设备管理接口，其他计算单元（比如FPGA）可快速灵活地实现与MindSpore的对接，欢迎您参与社区进行异构计算后端的开发工作。

### 编程语言扩展

Q：最近出来的taichi编程语言有Python扩展，类似`import taichi as ti`就能直接用了，MindSpore是否也支持？

A：MindSpore支持Python原生表达，`import mindspore`相关包即可使用。

<br/>

Q：MindSpore是否（计划）支持多语言扩展？

A：MindSpore目前支持Python扩展，针对C++、Rust、Julia等语言的支持正在开发中。

### 其他

Q：MindSpore在语义协同和处理上是如何实现的？是否利用当前学术界流行的FCA理论？

A：MindSpore框架本身并不需要支持FCA。对于语义类模型，用户可以调用第三方的工具在数据预处理阶段做FCA数学分析。MindSpore本身支持Python语言，`import FCA`相关包即可使用。

## 特性

Q：当前在云上MindSpore的训练和推理功能是比较完备的，至于边端场景（尤其是终端设备）MindSpore有什么计划？

A：MindSpore是端边云统一的训练和推理框架，支持将云侧训练的模型导出到Ascend AI处理器和终端设备进行推理。当前推理阶段支持的优化包括量化、算子融合、内存复用等。

## 能力

Q：MindSpore有没有类似基于TensorFlow实现的对象检测算法的模块？

A：TensorFlow的对象检测Pipeline接口属于TensorFlow Model模块。待MindSpore检测类模型完备后，会提供类似的Pipeline接口。
