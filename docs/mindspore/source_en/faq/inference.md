# Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/inference.md)

<font size=3>**Q: What should I do when an error `/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found` prompts during application compiling?**</font>

A: Find the directory where the missing dynamic library file is located.

<br/>

<font size=3>**Q: After updating MindSpore version, the application compilation reports errors `WARNING: Package(s) not found: mindspore-ascend`, `CMake Error: The following variables are use in this project, but they are set to NOTFOUND. Please set them or make sure they are set and tested correctly in the CMake files: MS_LIB`. What should I do?**</font>

A: MindSpore 2.0 has unified the installation packages of various platforms and no longer distinguishes different installation packages with suffixes such as `-ascend`, `-gpu`, etc. Therefore, the old compilation command or the old `build.sh` with ``MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"`` needs to be modified to ``MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"``.

<br/>

<font size=3>**Q: What should I do when error `error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory` prompts during application running?**</font>

A: While Ascend 310 AI Processor software packages relied by MindSpore is installed, the `CANN` package should install the full-featured `toolkit` version instead of the `nnrt` version.

<br/>

<font size=3>**Q: How to configure AIPP files?**</font>

A: AIPP (artistic intelligence pre-processing) AI preprocessing is used to complete image preprocessing on AI core, including changing image size, color gamut conversion (converting image format), subtracting mean / multiplication coefficient (changing image pixels). Real-time inference is performed after data processing. The related configuration introduction is complex. Please refer to [AIPP enable chapter of ATC tool](https://www.hiascend.com/document/detail/zh/canncommercial/51RC2/inferapplicationdev/atctool/atctool_0017.html).
