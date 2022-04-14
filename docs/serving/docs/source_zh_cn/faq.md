# FAQ

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/serving/docs/source_zh_cn/faq.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

<font size=3>**Q：MindSpore Serving是否支持热更新，避免推理服务中断？**</font>

A：MindSpore Serving当前不支持热更新，需要用户重启；当前建议跑多个Serving服务，升级模型版本时，重启部分服务以避免服务中断。

<br/>

<font size=3>**Q：MindSpore Serving是否支持一个模型启动多个Worker，以支持多卡单模型并发？**</font>

A：MindSpore Serving1.3版本后支持一个模型在多卡部署多个副本，实现多卡单模型并发执行。详细可以参考[Add样例](https://gitee.com/mindspore/serving/blob/r1.7/example/tensor_add/serving_server.py)。

<br/>

<font size=3>**Q：MindSpore Serving的版本和MindSpore的版本如何配套？**</font>

A：MindSpore Serving配套相同版本号的MindSpore的版本，比如Serving `1.1.1`版本配套 MindSpore `1.1.1`版本。

<br/>

<font size=3>**Q：编译时使用`bash -p`方式和 `bash -e`方式有什么区别？**</font>

A：MindSpore Serving的编译和运行依赖MindSpore，Serving提供两种编译方式：一种指定已安装的MindSpore路径，即`bash -p {python site-packages}/mindspore/lib`，避免编译Serving时再编译MindSpore；另一种，编译Serving时，编译配套的MindSpore，Serving会将`-e`、`-V`和`-j`选项透传给MindSpore。
比如，在Serving目录下，`bash -e ascend -V 910 -j32`：

- 首先将会以`bash -e ascend -V 910 -j32`方式编译`third_party/mindspore`目录下的MindSpore；
- 其次，编译脚本将MindSpore编译结果作为Serving的编译依赖。

<br/>

<font size=3>**Q：运行应用时报错`libmindspore.so: cannot open shared object file: No such file or directory`怎么办？**</font>

A：首先，需要确认是否安装MindSpore Serving所依赖的MindSpore；其次，Serving 1.1需要配置`LD_LIBRARY_PATH`，显式指定`libmindspore.so`所在路径，`libmindspore.so`当前在MindSpore Python安装路径的`lib`目录下；Serving 1.2后不再需要显示指定`libmindspore.so`所在路径，Serving会基于MindSpore安装路径查找并追加配置`LD_LIBRARY_PATH`，用户不再需要感知。

<br/>

<font size=3>**Q：如何控制Serving日志输出？**</font>

A：MindSpore Serving采用glog来输出日志，详细可参考[日志相关的环境变量和配置](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/debug/custom_debug.html#日志相关的环境变量和配置)，在此基础上，额外补充的内容：

- MS_SUBMODULE_LOG_v

该环境变量除了指定MindSpore C++各子模块日志级别，也可用于控制MindSpore Serving的日志级别。

可以通过GLOG_v=2 MS_SUBMODULE_LOG_v="{SERVING:1}"把Serving模块的日志级别设为INFO，其他模块的日志级别设为WARNING。

<br/>

<font size=3>**Q: 运行应用时报错`libmindspore.so: cannot open shared object file: No such file or directory`怎么办？**</font>

A: 首先，需要确认是否安装MindSpore Serving所依赖的MindSpore；其次，Serving 1.1需要配置`LD_LIBRARY_PATH`，显式指定`libmindspore.so`所在路径，`libmindspore.so`当前在MindSpore Python安装路径的`lib`目录下；Serving 1.2后不再需要显示指定`libmindspore.so`所在路径，Serving会基于MindSpore安装路径查找并追加配置`LD_LIBRARY_PATH`，用户不再需要感知。

<font size=3>**Q: 通过MindSpore Serving gRPC客户端发送超大消息报'assertion failed: slice_buffer->length <= UINT32_MAX'，什么原因?**</font>

具体报错信息：

```text
test_serving_client_grpc.py::test_serving_grpc_pressure_big_message E0413 20:03:08.764913058 122601 byte_stream.cc:40] assertion failed: slice_buffer->length <= UINT32_MAX
Fatal Python error: Aborted
Current thread 0x0000ffffb4884010 (most recent call first):
File ".../python3.7/site-packages/grpc/_channel.py", line 909 in _blocking
File ".../python3.7/site-packages/grpc/_channel.py", line 922 in call
File ".../python3.7/site-packages/mindspore_serving/client/python/client.py", line 217 in infer
```

A: MindSpore Serving提供Python Client封装gRPC消息通信，根据报错信息可知，gRPC底层校验发现消息大小大于4G（UINT32_MAX）。

进一步，MindSpore Serving默认设置了服务器接受消息大小为100MB，在接口`def start_grpc_server(address, max_msg_mb_size=100, ssl_config=None)`和`def start_restful_server(address, max_msg_mb_size=100, ssl_config=None)`可配置`max_msg_mb_size`参数设置服务器可接受的最大消息。

max_msg_mb_size接受[1,512]范围的整数数值，即可控制最大接收到的消息大小为1~512MB。如果超过了服务器的限定值，客户端将报类似以下错误，其中104857600即是服务器默认的限定100MB：

```text
Received message larger than max (419429033 vs. 104857600)
RESOURCE_EXHAUSTED
(8, 'resource exhausted')
```

客户端已通过参数限定最大发送消息大小为512MB。如果超过这个数值，客户端将会报类似这个错误：

```text
Sent message larger than max (838858033 vs. 536870912)
RESOURCE_EXHAUSTED
(8, 'resource exhausted')
```
