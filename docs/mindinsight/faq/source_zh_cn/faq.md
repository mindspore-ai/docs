# FAQ

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindinsight/faq/source_zh_cn/faq.md)

<font size=3>**Q：MindInsight启动失败并且提示:`ImportError: libcrypto.so.1.0.0: cannot open shared object file: No such file or directory` 如何处理？**</font>

A：需要在命令行中使用”export LD_LIBRARY_PATH=dir:$LD_LIBRARY_PATH”来导入LD_LIBRARY_PATH变量。

<br />

<font size=3>**Q：MindInsight启动失败并且提示:`bash: mindinsight: command not found` 如何处理？**</font>

A：当使用Python源码编译安装在自定义路径下会出现该问题，pip安装MindInsight时可执行文件会安装在该路径下，若使用`echo $PATH`查询到的bash环境变量中没有该安装目录会导致系统找不到安装的可执行文件。需要在命令行中使用`export PATH=$PATH:$YourPythonPath$/bin`来导入PATH变量。
(`$YourPythonPath$`请更换为你的安装路径)。注：该命令只在当前终端有效，若想永久有效请在`~/.bashrc`文件中加入该命令。

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

A：说明安装的Protobuf版本错误，需要安装正确版本的Protobuf，安装方法请参照[安装python版本的proto](https://support.huaweicloud.com/instg-cli-cann/atlascli_03_0046.html)。

<br />

<font size=3>**Q: MindInsight成功启动后，开启离线调试器，提示: `未找到调试器离线服务器模块` 如何处理？**</font>

A: 调试器离线服务需要启用MindSpore，请安装正确版本的MindSpore，安装方法请参照[安装MindSpore](https://www.mindspore.cn/install)。

<br/>

<font size=3>**Q：MindInsight成功启动后，在谷歌浏览器中访问失败,提示：`ERR_CONNECTION_REFUSED` 如何处理？**</font>

A：检查后台服务器与网络设备的防火墙策略配置，确保浏览器与MindInsight服务间的通讯连接，不受相关设备的配置规则限制。
