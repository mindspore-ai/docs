# 性能调优

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<!-- TOC -->

- [性能调优](#性能调优)
    - [参数调优](#参数调优)
    - [MindInsight使用](#MindInsight使用)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/performance_tuning.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 参数调优

<font size=3>**Q：MindSpore安装完成，执行训练时发现网络性能异常，权重初始化耗时过长，怎么办？**</font>  

A：可能与环境中使用了`scipy 1.4`系列版本有关，通过`pip list | grep scipy`命令可查看scipy版本，建议改成MindSpore要求的`scipy`版本。版本第三方库依赖可以在`requirement.txt`中查看。
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>
> 其中version替换为MindSpore具体的版本分支。

## MindInsight使用

<font size=3>**Q：MindInsight启动失败并且提示:`ImportError: libcrypto.so.1.0.0: cannot open shared object file: No such file or directory` 如何处理？**</font>

A：需要在命令行中使用”export LD_LIBRARY_PATH=dir:$LD_LIBRARY_PATH”来导入LD_LIBRARY_PATH变量。

<br />

<font size=3>**Q：卸载MindInsight后，在MindInsight的运行日志中出现：`No module named 'mindinsight'` 如何处理？**</font>

A：MindInsight启动后，会变成一个后台服务。卸载MindInsight后，已启动的MindInsight后台服务不会自行停止。
当MindInsight后台服务启动新的进程加载新数据或者做其他操作时，则会触发`No module named 'mindinsight'`的异常信息，并记录到日志中。

此时可以通过下面两种方式进行处理：

- 重新安装MindInsight，并使用`mindinsight stop --port <PORT>`命令停止已启动的MindInsight后台服务。
- 通过`kill -9 <PID>`命令，将MindInsight涉及的相关进程杀死。

<br />

<font size=3>**Q：MindInsight成功启动后，在谷歌浏览器中访问时，提示：`ERR_UNSAFE_PORT` 如何处理？**</font>

A：谷歌浏览器内核禁止将某些端口作为`HTTP`服务，你需要在谷歌浏览器的属性中新增配置`--explicitly-allowed-ports=port`。或者，你可以更换端口或者更换为IE浏览器。

<br />

<font size=3>**Q：在Ascend机器上启动Mindinsight并开启调试器后，训练脚本连接调试器时，提示：`Exeption calling application: Field number 0 is illegal` 如何处理？**</font>

A：说明安装的protobuf版本错误，需要安装正确版本的protobuf，安装方法请参照[安装python版本的proto](https://support.huaweicloud.com/instg-cli-cann/atlascli_03_0046.html)。
