# Inference

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<!-- TOC -->

- [Inference](#inference)
    - [MindSpore C++ Library Use](#mindspore-c++-library-use)
    - [MindSpore Serving](#mindspore-serving)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_en/inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## MindSpore C++ Library Use

<font size=3>**Q: What should I do when error `/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found` prompts during application compiling?**</font>

A: Find the directory where the missing dynamic library file is located, add the path to the environment variable `LD_LIBRARY_PATH`, and refer to [Inference Using the MindIR Model on Ascend 310 AI Processors#Building Inference Code](https://www.mindspore.cn/tutorial/inference/en/master/multi_platform_inference_ascend_310_mindir.html#building-inference-code) for environment variable settings.

<br/>

<font size=3>**Q: What should I do when error `ModuleNotFoundError: No module named 'te'` prompts during application running?**</font>

A: First confirm whether the system environment is installed correctly and whether the whl packages such as `te` and `topi` are installed correctly. If there are multiple Python versions in the user environment, such as Conda virtual environment, you need to execute `ldd name_of_your_executable_app` to confirm whether the application link `libpython3.7m.so.1.0` is consistent with the current Python directory, if not, you need to adjust the order of the environment variable `LD_LIBRARY_PATH` .

<br/>

<font size=3>**Q: What should I do when error `error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory` prompts during application running?**</font>

A: While installing Ascend 310 AI Processor software packages，the `CANN` package should install the full-featured `toolkit` version instead of the `nnrt` version.

## MindSpore Serving

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

<font size=3>**Q: How to control the output of Serving log?**</font>

A: MindSpore Serving uses glog to output logs, for more details, please refer to [Log-related Environment Variables and Configurations](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/custom_debugging_info.html?highlight=GLOG#log-related-environment-variables-and-configurations). On this basis, additional supplementary contents are as follows:

- MS_SUBMODULE_LOG_v

This environment variable can also be used to control the log level of MindSpore Serving in addition to specifying the log level of each sub module of MindSpore C++.

We can use GLOG_v=2 MS_SUBMODULE_LOG_v="{SERVING:1}" to set the log level of the Serving module to INFO, and the log level of other modules to WARNING.
