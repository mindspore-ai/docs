# MindSpore Serving Class

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_en/mindspore_serving.md)

<font size=3>**Q: Does MindSpore Serving support hot update to avoid inference service interruption?**</font>

A: MindSpore Serving does not support hot update. You need to restart MindSpore Serving. You are advised to run multiple Serving services. When updating a model, restart some services to avoid service interruption.

<br/>

<font size=3>**Q: Does MindSpore Serving allow multiple workers to be started for one model to support multi-device and single-model concurrency?**</font>

A: MindSpore Serving does not support distribution and this function is being developed. That is, multiple workers cannot be started for one model. It is recommended that multiple Serving services be deployed to implement distribution and load balancing. In addition, to avoid message forwarding between `master` and `worker`, you can use the `start_servable_in_master` API to enable `master` and `worker` to be executed in the same process, implementing lightweight deployment of the Serving services.

<br/>

<font size=3>**Q: How does the MindSpore Serving version match the MindSpore version?**</font>

A: MindSpore Serving matches MindSpore in the same version. For example, Serving `1.1.1` matches MindSpore `1.1.1`.

<br/>

<font size=3>**Q: What is the difference between `bash -p` and `bash -e` when an error is reported during application build?**</font>

A: MindSpore Serving build and running depend on MindSpore. Serving provides two build modes: 1. Use `bash -p {python site-packages}/mindspore/lib` to specify an installed MindSpore path to avoid building MindSpore when building Serving. 2. Build Serving and the corresponding MindSpore. Serving passes the `-e`, `-V`, and `-j` options to MindSpore.
For example, use `bash -e ascend -V 910 -j32` in the Serving directory as follows:

- Build MindSpore in the `third_party/mindspore` directory using `bash -e ascend -V 910 -j32`.
- Use the MindSpore build result as the Serving build dependency.

<br/>

<font size=3>**Q: What can I do if an error `libmindspore.so: cannot open shared object file: No such file or directory` is reported during application running?**</font>

A: Check whether MindSpore that MindSpore Serving depends on is installed. In Serving 1.1, `LD_LIBRARY_PATH` needs to be configured to explicitly specify the path of `libmindspore.so`. `libmindspore.so` is in the `lib` directory of the MindSpore Python installation path. In Serving 1.2 or later, the path of `libmindspore.so` does not need to be specified. Serving searches for and adds `LD_LIBRARY_PATH` based on the MindSpore installation path, which does not need to be perceived by users.
