# Inference

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/faq/inference.md)

## Q: In the previous version, Atlas 200/300/500 inference product inference is performed based on the MindSpore installation package. However, the MindSpore release package of the new version does not support Atlas 200/300/500 inference product inference. How do I use Atlas 200/300/500 inference product for inference? (Changes in the MindSpore Atlas 200/300/500 Inference Product Inference Release Package)

A: The MindSpore inference function is provided by MindSpore Lite, a core component of MindSpore. Since version 2.0, the Atlas 200/300/500 inference product inference package is released by MindSpore Lite and provides continuous maintenance and evolution of related functions. The corresponding interfaces in the MindSpore main release package are not maintained or evolved. Since version 2.2, the MindSpore main release package does not provide the inference interface enabling for the Atlas 200/300/500 inference product. If you need to use the inference interface, install the MindSpore Lite release package or download the MindSpore version earlier than 2.0. For details about how to install and use MindSpore Lite, see <https://www.mindspore.cn/lite/en>.

HUAWEI Atlas 200/300/500 inference product is an energy-efficient and highly integrated AI processor for edge scenarios. It supports inference on MindIR models. In the earlier version, MindSpore provides two methods for enabling inference on the Atlas 200/300/500 inference product hardware:

1. The MindSpore main release package provides the matching Atlas 200/300/500 inference product version that supports C++ inference interfaces.
2. The MindSpore Lite release package provides the matching Ascend version and supports C++ and Java inference.

The C++ APIs provided by the two solutions are basically the same. In the future, MindSpore Lite is used instead of building and maintaining two sets of interfaces.

The original Atlas 200/300/500 inference product inference service built based on the MindSpore main release package can be switched to MindSpore Lite with a few modifications. The following is an example:

1. compiling a C++ Project

    You do not need to use the MindSpore installation package. You need to download the MindSpore Lite C++ version package and decompress it to any working directory. The directory structure is as follows:

    ```text
    mindspore-lite-{version}-linux-{arch}
    ├── runtime
    │   ├── include
    │   ├── lib
    ```

    The path to the include header file is changed to the include directory, and the link to the dynamic library is changed to lib/libmindspore-lite.so during compilation. If minddata is required, also link lib/libminddata-lite.so.

    For example, set the environment variable ``MINDSPORE_PATH=/path/to/mindspore-lite``, and cmake is rewritten in the following way:

    ```cmake
    ...
    include_directories(${MINDSPORE_PATH})
    include_directories(${MINDSPORE_PATH}/include)
    ...

    if(EXISTS ${MINDSPORE_PATH}/lib/libmindspore-lite.so)
        message(--------------- Compile-with-MindSpore-Lite ----------------)
        set(MS_LIB ${MINDSPORE_PATH}/lib/libmindspore-lite.so)
        set(MD_LIB ${MINDSPORE_PATH}/lib/libminddata-lite.so)
    endif()

    add_executable(main src/main.cc)
    target_link_libraries(main ${MS_LIB} ${MD_LIB})
    ```

2. inference

    Except that the usage of the following two classes is different, all the methods for obtaining and structuring input and output, and executing inference are the same.

    - 2.1 structuring context

      The method for structuring context is modified as follows: `Ascend310DeviceInfo` is replaced with `AscendDeviceInfo`.

      ```c++
      // Original MindSpore
      - auto context = std::make_shared<Context>();
      - auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
      - ascend310->SetDeviceID(device_id);
      - context->MutableDeviceInfo().push_back(ascend310);

      // MindSpore lite
      + auto context = std::make_shared<Context>();
      + auto ascend = std::make_shared<mindspore::AscendDeviceInfo>();
      + ascend->SetDeviceID(device_id);
      + context->MutableDeviceInfo().push_back(ascend);
      ```

    - 2.2 graph compilation

      The graph compilation interface is adjusted as follows: You do not need to construct and load graph objects. Instead, you can directly transfer the mindir model file through the `Build` interface.

      ```c++
      // Original MindSpore
      -  mindspore::Graph graph;
      -  Serialization::Load(mindir_path, mindspore::kMindIR, &graph);
      -  auto ret = model->Build(GraphCell(graph), context);

      // MindSpore lite
      +  auto ret = model->Build(mindir_path, mindspore::kMindIR, context);
      ```

<br/>

## Q: What should I do when an error `/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found` prompts during application compiling?

A: Find the directory where the missing dynamic library file is located.

<br/>

## Q: After updating MindSpore version, the application compilation reports errors `WARNING: Package(s) not found: mindspore-ascend`, `CMake Error: The following variables are use in this project, but they are set to NOTFOUND. Please set them or make sure they are set and tested correctly in the CMake files: MS_LIB`. What should I do?

A: MindSpore 2.0 has unified the installation packages of various platforms and no longer distinguishes different installation packages with suffixes such as `-ascend`, `-gpu`, etc. Therefore, the old compilation command or the old `build.sh` with ``MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"`` needs to be modified to ``MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"``.

<br/>

## Q: What should I do when error `error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory` prompts during application running?

A: While Atlas 200/300/500 inference product software packages relied by MindSpore is installed, the `CANN` package should install the full-featured `toolkit` version instead of the `nnrt` version.

<br/>

## Q: How to configure AIPP files?

A: AIPP (artistic intelligence pre-processing) AI preprocessing is used to complete image preprocessing on AI core, including changing image size, color gamut conversion (converting image format), subtracting mean / multiplication coefficient (changing image pixels). Real-time inference is performed after data processing. The related configuration introduction is complex. Please refer to [AIPP enable chapter of ATC tool](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0017.html).
