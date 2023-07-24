# MindSpore C++ Library Use

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.2/docs/faq/source_en/mindspore_cpp_library.md)

<font size=3>**Q：What should I do when error `/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found` prompts during application compiling?**</font>

<br/>

A：Find the directory where the missing dynamic library file is located, add the path to the environment variable `LD_LIBRARY_PATH`, and refer to [Inference Using the MindIR Model on Ascend 310 AI Processors#Building Inference Code](https://www.mindspore.cn/tutorial/inference/en/r1.2/multi_platform_inference_ascend_310_mindir.html#building-inference-code) for environment variable settings.

<br/>

<font size=3>**Q：What should I do when error `ModuleNotFoundError: No module named 'te'` prompts during application running?**</font>

A：First confirm whether the system environment is installed correctly and whether the whl packages such as `te` and `topi` are installed correctly. If there are multiple Python versions in the user environment, such as Conda virtual environment, you need to execute `ldd name_of_your_executable_app` to confirm whether the application link `libpython3.7m.so.1.0` is consistent with the current Python directory, if not, you need to adjust the order of the environment variable `LD_LIBRARY_PATH` .

<br/>

<font size=3>**Q：What should I do when error `error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory` prompts during application running?**</font>

A：While installing Ascend 310 AI Processor software packages，the `CANN` package should install the full-featured `toolkit` version instead of the `nnrt` version.