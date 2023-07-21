# FAQ

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.5/docs/serving/faq/source_en/faq.md)

<font size=3>**Q: Does MindSpore Serving support hot update to avoid inference service interruption?**</font>

A: MindSpore Serving does not support hot update. You need to restart MindSpore Serving. You are advised to run multiple Serving services. When updating a model, restart some services to avoid service interruption.

<br/>

<font size=3>**Q: Does MindSpore Serving allow multiple workers to be started for one model to support multi-device and single-model concurrency?**</font>

A: After MindSpore Serving version 1.3, it supports the deployment of multiple copies of a model in multiple cards to achieve concurrent execution of multiple cards and single models. For details, please refer to [Add Sample](https://gitee.com/mindspore/serving/blob/r1.5/example/tensor_add/serving_server.py).

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

A：MindSpore Serving uses glog to output logs, for more details, please refer to [Log-related Environment Variables and Configurations](https://www.mindspore.cn/docs/programming_guide/en/r1.5/custom_debugging_info.html#log-related-environment-variables-and-configurations). On this basis, additional supplementary contents are as follows:

- MS_SUBMODULE_LOG_v

This environment variable can also be used to control the log level of MindSpore Serving in addition to specifying the log level of each sub module of MindSpore C++.

We can use GLOG_v=2 MS_SUBMODULE_LOG_v="{SERVING:1}" to set the log level of the Serving module to INFO, and the log level of other modules to WARNING.

<br/>

<font size=3>**Q: What can I do if an error `libmindspore.so: cannot open shared object file: No such file or directory` is reported during application running?**</font>

A: Check whether MindSpore that MindSpore Serving depends on is installed. In Serving 1.1, `LD_LIBRARY_PATH` needs to be configured to explicitly specify the path of `libmindspore.so`. `libmindspore.so` is in the `lib` directory of the MindSpore Python installation path. In Serving 1.2 or later, the path of `libmindspore.so` does not need to be specified. Serving searches for and adds `LD_LIBRARY_PATH` based on the MindSpore installation path, which does not need to be perceived by users.
