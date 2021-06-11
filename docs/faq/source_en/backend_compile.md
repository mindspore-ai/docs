# Banckend Compile

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_en/backend_compile.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: What framework models and formats can be directly read by MindSpore? Can the PTH Model Obtained Through Training in PyTorch Be Loaded to the MindSpore Framework for Use?**</font>

A: MindSpore uses protocol buffers (protobuf) to store training parameters and cannot directly read framework models. A model file stores parameters and their values. You can use APIs of other frameworks to read parameters, obtain the key-value pairs of parameters, and load the key-value pairs to MindSpore. If you want to use the .ckpt file trained by a framework, read the parameters and then call the `save_checkpoint` API of MindSpore to save the file as a .ckpt file that can be read by MindSpore.

<br/>

<font size=3>**Q：What should I do if a Protobuf memory limit error is reported during the process of using ckpt or exporting a model?**</font>

A：When a single Protobuf data is too large, because Protobuf itself limits the size of the data stream, a memory limit error will be reported. At this time, the restriction can be lifted by setting the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

<br/>

<font size=3>**Q: What is the difference between the PyNative and Graph modes?**</font>

A: In terms of efficiency, operators used in the two modes are the same. Therefore, when the same network and operators are executed in the two modes, the accuracy is the same. The network execution performance varies according to the execution mechanism. Theoretically, operators provided by MindSpore support both the PyNative and Graph modes.

In terms of application scenarios, Graph mode requires the network structure to be built at the beginning, and then the framework performs entire graph optimization and execution. This mode is suitable to scenarios where the network is fixed and high performance is required.

The two modes are supported on different hardware (such as `Ascend`, `GPU`, and `CPU`).

In terms of code debugging, since operators are executed line by line in PyNative mode, you can directly debug the Python code and view the `/api` output or execution result of the corresponding operator at any breakpoint in the code. In Graph mode, the network is built but not executed in the constructor function. Therefore, you cannot obtain the output of the corresponding operator at breakpoints in the `construct` function. The output can be viewed only after the network execution is complete.

<br/>

<font size=3>**Q: What is the difference between `c_transforms` and `py_transforms`? Which one is recommended?**</font>

A: `c_transforms` is recommended. Its performance is better because it is executed only at the C layer.

Principle: The underlying layer of `c_transform` uses `opencv/jpeg-turbo` of the C version for data processing, and `py_transform` uses `Pillow` of the Python version for data processing.

<br/>

<font size=3>**Q: What is the difference between `bash -p` and `bash -e` when an error is reported during application build?**</font>

A: MindSpore Serving build and running depend on MindSpore. Serving provides two build modes: 1. Use `bash -p {python site-packages}/mindspore/lib` to specify an installed MindSpore path to avoid building MindSpore when building Serving. 2. Build Serving and the corresponding MindSpore. Serving passes the `-e`, `-V`, and `-j` options to MindSpore.
For example, use `bash -e ascend -V 910 -j32` in the Serving directory as follows:

- Build MindSpore in the `third_party/mindspore` directory using `bash -e ascend -V 910 -j32`.
- Use the MindSpore build result as the Serving build dependency.

<br/>

<font size=3>**Q: What can I do if an error `libmindspore.so: cannot open shared object file: No such file or directory` is reported during application running?**</font>

A: Check whether MindSpore that MindSpore Serving depends on is installed. In Serving 1.1, `LD_LIBRARY_PATH` needs to be configured to explicitly specify the path of `libmindspore.so`. `libmindspore.so` is in the `lib` directory of the MindSpore Python installation path. In Serving 1.2 or later, the path of `libmindspore.so` does not need to be specified. Serving searches for and adds `LD_LIBRARY_PATH` based on the MindSpore installation path, which does not need to be perceived by users.
