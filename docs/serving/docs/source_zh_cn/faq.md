# FAQ

<a href="https://gitee.com/mindspore/docs/blob/master/docs/serving/docs/source_en/faq.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: Does MindSpore Serving support hot update to avoid inference service interruption?**</font>

A: MindSpore Serving does not support hot update. You need to restart MindSpore Serving. You are advised to run multiple Serving services. When updating a model, restart some services to avoid service interruption.

<br/>

<font size=3>**Q: Does MindSpore Serving allow multiple workers to be started for one model to support multi-device and single-model concurrency?**</font>

A: After MindSpore Serving version 1.3, it supports the deployment of multiple copies of a model in multiple cards to achieve concurrent execution of multiple cards and single models. For details, please refer to [Add Sample](https://gitee.com/mindspore/serving/blob/master/example/tensor_add/serving_server.py).

<br/>

<font size=3>**Q: How does the MindSpore Serving version match the MindSpore version?**</font>

A: MindSpore Serving matches MindSpore in the same version. For example, Serving `1.1.1` matches MindSpore `1.1.1`.

<br/>

<font size=3>**Q: What is the difference between using `bash -p` method and `bash -e` method when compiling?**</font>

A: MindSpore Serving build and running depend on MindSpore. Serving provides two build modes: 1. Use `bash -p {python site-packages}/mindspore/lib` to specify an installed MindSpore path to avoid building MindSpore when building Serving. 2. Build Serving and the corresponding MindSpore. Serving passes the `-e`, `-V`, and `-j` options to MindSpore.
For example, use `bash -e ascend -V 910 -j32` in the Serving directory as follows:

- Build MindSpore in the `third_party/mindspore` directory using `bash -e ascend -V 910 -j32`.
- Use the MindSpore build result as the Serving build dependency.

<br/>

<font size=3>**Q: What can I do if an error `libmindspore.so: cannot open shared object file: No such file or directory` is reported during application running?**</font>

A: Check whether MindSpore that MindSpore Serving depends on is installed. In Serving 1.1, `LD_LIBRARY_PATH` needs to be configured to explicitly specify the path of `libmindspore.so`. `libmindspore.so` is in the `lib` directory of the MindSpore Python installation path. In Serving 1.2 or later, the path of `libmindspore.so` does not need to be specified. Serving searches for and adds `LD_LIBRARY_PATH` based on the MindSpore installation path, which does not need to be perceived by users.

<font size=3>**Q：How to control the output of Serving log?**</font>

A：MindSpore Serving uses glog to output logs, for more details, please refer to [Log-related Environment Variables and Configurations](https://www.mindspore.cn/docs/programming_guide/en/master/custom_debugging_info.html#log-related-environment-variables-and-configurations). On this basis, additional supplementary contents are as follows:

- MS_SUBMODULE_LOG_v

This environment variable can also be used to control the log level of MindSpore Serving in addition to specifying the log level of each sub module of MindSpore C++.

We can use GLOG_v=2 MS_SUBMODULE_LOG_v="{SERVING:1}" to set the log level of the Serving module to INFO, and the log level of other modules to WARNING.

<br/>

<font size=3>**Q: What can I do if an error `libmindspore.so: cannot open shared object file: No such file or directory` is reported during application running?**</font>

A: Check whether MindSpore that MindSpore Serving depends on is installed. In Serving 1.1, `LD_LIBRARY_PATH` needs to be configured to explicitly specify the path of `libmindspore.so`. `libmindspore.so` is in the `lib` directory of the MindSpore Python installation path. In Serving 1.2 or later, the path of `libmindspore.so` does not need to be specified. Serving searches for and adds `LD_LIBRARY_PATH` based on the MindSpore installation path, which does not need to be perceived by users.

<font size=3>**Q: Error 'assertion failed: slice_buffer->length <= UINT32_MAX' is reported when an extra large meesage is send through the MindSpore Serving gPRC Client.**</font>

Detailed error information:

```text
test_serving_client_grpc.py::test_serving_grpc_pressure_big_message E0413 20:03:08.764913058 122601 byte_stream.cc:40] assertion failed: slice_buffer->length <= UINT32_MAX
Fatal Python error: Aborted
Current thread 0x0000ffffb4884010 (most recent call first):
File ".../python3.7/site-packages/grpc/_channel.py", line 909 in _blocking
File ".../python3.7/site-packages/grpc/_channel.py", line 922 in call
File ".../python3.7/site-packages/mindspore_serving/client/python/client.py", line 217 in infer
```

A: MindSpore Serving provides Python Client to encapsulate gRPC communication. According to the error information above, the message size exceeds 4GB(UINT32_MAX).

Further, MindSpore Serving sets the size of the message accepted by the server to 100MB by default. Parameter `max_msg_mb_size` can be configured in `def start_grpc_server(address, max_msg_mb_size=100, ssl_config=None)` and `def start_restful_server(address, max_msg_mb_size=100, ssl_config=None)` interfaces to set the maximum message accepted by the server.

Parameter `max_msg_mb_size` accepts an integer ranging from 1 to 512 to control the maximum size of a received message. If the message size exceeds the value, the client reports an error similar to the following, 104857600 indicates the default limit of 100MB on the server:

```text
Received message larger than max (419429033 vs. 104857600)
RESOURCE_EXHAUSTED
(8, 'resource exhausted')
```

The maximum size of a message sent by the client is limited to 512MB. If the maximum size is exceeded, the client reports an error similar to the following:

```text
Sent message larger than max (838858033 vs. 536870912)
RESOURCE_EXHAUSTED
(8, 'resource exhausted')
```
