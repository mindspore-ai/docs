# MindSpore Serving类

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_zh_cn/mindspore_serving.md)

<font size=3>**Q：MindSpore Serving是否支持热更新，避免推理服务中断？**</font>

A：MindSpore Serving当前不支持热更新，需要用户重启；当前建议跑多个Serving服务，升级模型版本时，重启部分服务以避免服务中断。

<br/>

<font size=3>**Q：MindSpore Serving是否支持一个模型启动多个Worker，以支持多卡单模型并发？**</font>

A：MindSpore Serving暂未支持分流，即不支持一个模型启动多个Worker，这个功能正在开发中；当前建议跑多个Serving服务，通过对接多个Serving服务的服务器进行分流和负载均衡。另外，为了避免`master`和`worker`之间的消息转发，可以使用接口`start_servable_in_master`使`master`和`worker`执行在同一进程，实现Serving服务轻量级部署。

<br/>

<font size=3>**Q：MindSpore Serving的版本和MindSpore的版本如何配套？**</font>

A：MindSpore Serving配套相同版本号的MindSpore的版本，比如Serving `1.1.1`版本配套 MindSpore `1.1.1`版本。

<br/>

<font size=3>**Q：编译应用时报错`bash -p`方式和 `bash -e`方式的区别？**</font>

A：MindSpore Serving的编译和运行依赖MindSpore，Serving提供两种编译方式：一种指定已安装的MindSpore路径，即`bash -p {python site-packages}/mindspore/lib`，避免编译Serving时再编译MindSpore；另一种，编译Serving时，编译配套的MindSpore，Serving会将`-e`、`-V`和`-j`选项透传给MindSpore。
比如，在Serving目录下，`bash -e ascend -V 910 -j32`：

- 首先将会以`bash -e ascend -V 910 -j32`方式编译`third_party/mindspore`目录下的MindSpore；
- 其次，编译脚本将MindSpore编译结果作为Serving的编译依赖。

<br/>

<font size=3>**Q：运行应用时报错`libmindspore.so: cannot open shared object file: No such file or directory`怎么办？**</font>

A：首先，需要确认是否安装MindSpore Serving所依赖的MindSpore；其次，Serving 1.1需要配置`LD_LIBRARY_PATH`，显式指定`libmindspore.so`所在路径，`libmindspore.so`当前在MindSpore Python安装路径的`lib`目录下；Serving 1.2后不再需要显示指定`libmindspore.so`所在路径，Serving会基于MindSpore安装路径查找并追加配置`LD_LIBRARY_PATH`，用户不再需要感知。
