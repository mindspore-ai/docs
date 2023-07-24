# 推理

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/faq/source_zh_cn/inference.md)

<font size=3>**Q: 编译应用时报错`/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found`怎么办？**</font>

A: 寻找缺少的动态库文件所在目录，添加该路径到环境变量`LD_LIBRARY_PATH`中，环境变量设置参考[Ascend 310 AI处理器上使用MindIR模型进行推理#编译推理代码](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/multi_platform_inference_ascend_310_mindir.html#id6)。

<br/>

<font size=3>**Q: 运行应用时出现`ModuleNotFoundError: No module named 'te'`怎么办？**</font>

A: 首先确认环境安装是否正确，`te`、`topi`等whl包是否正确安装。如果用户环境中有多个Python版本，如Conda虚拟环境中，需`ldd name_of_your_executable_app`确认应用所链接的`libpython3.7m.so.1.0`是否与当前Python路径一致，如果不一致需要调整环境变量`LD_LIBRARY_PATH`顺序。

<br/>

<font size=3>**Q: 运行应用时报错`error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory`怎么办？**</font>

A: 安装MindSpore所依赖的Ascend 310 AI处理器配套软件包时，`CANN`包不能安装`nnrt`版本，而是需要安装功能完整的`toolkit`版本。
