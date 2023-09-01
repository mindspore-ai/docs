# 推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/faq/inference.md)

<font size=3>**Q: 编译应用时报错`/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found`怎么办？**</font>

A: 寻找缺少的动态库文件所在目录，添加该路径到环境变量`LD_LIBRARY_PATH`中。

<br/>

<font size=3>**Q: 更新MindSpore版本后，编译应用报错`WARNING: Package(s) not found: mindspore-ascend`、`CMake Error: The following variables are use in this project, but they are set to NOTFOUND. Please set them or make sure they are set and tested correctly in the CMake files: MS_LIB`怎么办？**</font>

A: MindSpore 2.0开始统一了各平台的安装包，不再以`-ascend`、`-gpu`等后缀区分不同安装包，因此旧编译命令或旧`build.sh`中的``MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"``需要修改为``MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"``。

<br/>

<font size=3>**Q: 运行应用时报错`error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory`怎么办？**</font>

A: 安装MindSpore所依赖的Ascend 310 AI处理器配套软件包时，`CANN`包不能安装`nnrt`版本，而是需要安装功能完整的`toolkit`版本。

<br/>

<font size=3>**Q: AIPP文件怎么配置？**</font>

A: AIPP（Artificial Intelligence Pre-Processing）AI预处理，用于在AI Core上完成图像预处理，包括改变图像尺寸、色域转换（转换图像格式）、减均值/乘系数（改变图像像素），数据处理之后再进行真正的模型推理。相关的配置介绍比较复杂，可以参考[ATC工具的AIPP使能章节](https://www.hiascend.com/document/detail/zh/canncommercial/51RC2/inferapplicationdev/atctool/atctool_0017.html)。
