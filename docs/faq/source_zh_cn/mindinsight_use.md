# 可视化组件MindInsight使用类

`Linux` `Ascend` `GPU` `环境准备`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/mindinsight_use.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q：MindInsight启动失败并且提示:`ImportError: libcrypto.so.1.0.0: cannnot open shared object file: No such file or directory` 如何处理？

A：需要在命令行中使用”export LD_LIBRARY_PATH=dir:$LD_LIBRARY_PATH”来导入LD_LIBRARY_PATH变量。

<br />

Q：卸载MindInsight后，在MindInsight的运行日志中出现：`No module named 'mindinsight'` 如何处理？

A：MindInsight启动后，会变成一个后台服务。卸载MindInsight后，已启动的MindInsight后台服务不会自行停止。
当MindInsight后台服务启动新的进程加载新数据或者做其他操作时，则会触发`No module named 'mindinsight'`的异常信息，并记录到日志中。

此时可以通过下面两种方式进行处理：

- 重新安装MindInsight，并使用`mindinsight stop --port <PORT>`命令停止已启动的MindInsight后台服务。
- 通过`kill -9 <PID>`命令，将MindInsight涉及的相关进程杀死。
