# 推理

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/faq/inference.md)

## Q: 原先基于MindSpore安装包进行Atlas 200/300/500推理产品推理，新版本MindSpore发布包不支持Atlas 200/300/500推理产品平台的推理？如何使用Atlas 200/300/500推理产品进行推理？（MindSpore Atlas 200/300/500推理产品推理功能发布包变更说明）

A: 由于MindSpore推理功能统一由MindSpore核心组件 - MindSpore lite提供。自2.0版本起，统一由MindSpore lite发布Atlas 200/300/500推理产品推理包，并提供相关功能的持续维护演进，而MindSpore主发布包里的对应接口不再维护和演进。自2.2版本起MindSpore主发布包不再提供配套Atlas 200/300/500推理产品的推理接口使能，如需使用请切换安装MindSpore Lite发布包或下载MindSpore 2.0之前的版本。MindSpore lite的安装部署与用法详见 <https://www.mindspore.cn/lite>。

Atlas 200/300/500推理产品是面向边缘场景的高能效高集成度AI处理器，支持对MindIR格式模型进行推理。原先MindSpore提供了两种在Atlas 200/300/500推理产品硬件上的推理使能用法：

1. 由MindSpore主发布包提供配套Atlas 200/300/500推理产品的版本，支持C++推理接口。
2. 由MindSpore Lite发布包提供配套Ascend的版本，支持C++/Java两种语言进行推理。

这两种方案提供的C++ API基本一致，后续不再构建和维护两套接口，而是归一使用MindSpore Lite。

原有基于MindSpore主发布包构建的Atlas 200/300/500推理产品推理业务，可以少量修改切换到MindSpore Lite，示例如下：

1. 编译C++工程

    不再使用MindSpore 安装包，需下载MindSpore Lite C++版本包并解压在任意工作目录。目录结构如下：

    ```text
    mindspore-lite-{version}-linux-{arch}
    ├── runtime
    │   ├── include
    │   ├── lib
    ```

    编译时包含头文件路径改为include目录，链接动态库改为lib/libmindspore-lite.so，若需使用minddata，需同时链接lib/libminddata-lite.so。  
    举例，设环境变量``MINDSPORE_PATH=/path/to/mindspore-lite``，cmake改写为如下方式：

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

2. 推理

    除如下两类写法存在差异，其余输入输出获取、构造方式，执行推理接口等保持一致。

    - 2.1 context构造

      context构造方式改为如下，Ascend310DeviceInfo统一替换为AscendDeviceInfo

      ```c++
      // 原MindSpore
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

    - 2.2 图编译

      图编译接口调整为如下，无需构造Graph对象、序列化加载，Build接口直接传入mindir模型文件即可。

      ```c++
      // 原MindSpore
      -  mindspore::Graph graph;
      -  Serialization::Load(mindir_path, mindspore::kMindIR, &graph);
      -  auto ret = model->Build(GraphCell(graph), context);

      // MindSpore lite
      +  auto ret = model->Build(mindir_path, mindspore::kMindIR, context);
      ```

<br/>

## Q: 编译应用时报错`/usr/bin/ld: warning: libxxx.so, needed by libmindspore.so, not found`怎么办？

A: 寻找缺少的动态库文件所在目录，添加该路径到环境变量`LD_LIBRARY_PATH`中。

<br/>

## Q: 更新MindSpore版本后，编译应用报错`WARNING: Package(s) not found: mindspore-ascend`、`CMake Error: The following variables are use in this project, but they are set to NOTFOUND. Please set them or make sure they are set and tested correctly in the CMake files: MS_LIB`怎么办？

A: MindSpore 2.0开始统一了各平台的安装包，不再以`-ascend`、`-gpu`等后缀区分不同安装包，因此旧编译命令或旧`build.sh`中的``MINDSPORE_PATH="`pip show mindspore-ascend | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"``需要修改为``MINDSPORE_PATH="`pip show mindspore | grep Location | awk '{print $2"/mindspore"}' | xargs realpath`"``。

<br/>

## Q: 运行应用时报错`error while loading shared libraries: libge_compiler.so: cannot open shared object file: No such file or directory`怎么办？

A: 安装MindSpore所依赖的Atlas 200/300/500推理产品配套软件包时，`CANN`包不能安装`nnrt`版本，而是需要安装功能完整的`toolkit`版本。

<br/>

## Q: AIPP文件怎么配置？

A: AIPP（Artificial Intelligence Pre-Processing）AI预处理，用于在AI Core上完成图像预处理，包括改变图像尺寸、色域转换（转换图像格式）、减均值/乘系数（改变图像像素），数据处理之后再进行真正的模型推理。相关的配置介绍比较复杂，可以参考[ATC工具的AIPP使能章节](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/atc/atlasatc_16_0017.html)。
