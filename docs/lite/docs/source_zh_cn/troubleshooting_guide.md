# 问题定位指南

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/troubleshooting_guide.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

在MindSpore Lite使用中遇到问题时，可首先查看日志，多数场景下的问题可以通过日志报错信息直接定位（通过设置环境变量[GLOG_v](https://mindspore.cn/docs/programming_guide/zh-CN/master/custom_debugging_info.html#id11) 调整日志等级可以打印更多调试日志），这里简单介绍几种常见报错场景的问题定位与解决方法。

> 1. 因不同版本中日志行号可能存在差异，下述示例日志报错信息中的行号信息均用”**”表示；
> 2. 示例日志中只列出了通用信息，其他涉及具体场景的信息均用“****”表示。

1. 日志报错信息：

    ```cpp
    [mindspore/lite/tools/converter/converter.cc:**] BuildFuncGraph] Get funcGraph failed for fmk: ****
    [mindspore/lite/tools/converter/converter.cc:**] Converter] Parser/Import model return nullptr
    [mindspore/lite/tools/converter/converter_context.h:**] PrintOps] ===========================================
    [mindspore/lite/tools/converter/converter_context.h:**] PrintOps] UNSUPPORTED OP LIST:
    [mindspore/lite/tools/converter/converter_context.h:**] PrintOps] FMKTYPE: ****, OP TYPE: ****
    [mindspore/lite/tools/converter/converter_context.h:**] PrintOps] ===========================================
    [mindspore/lite/tools/converter/converter.cc:**] RunConverter] CONVERT RESULT FAILED:-300 Failed to find operator.
    ```

    - 问题分析：模型中存在MindSpore Lite转换工具不支持的算子导致转换失败。
    - 解决方法：对于不支持的算子可以尝试通过继承API接口[NodeParser](https://mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_converter.html#nodeparser) 自行添加parser并通过[NodeParserRegistry](https://mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_registry.html#nodeparserregistry) 进行Parser注册；或者在社区提[ISSUE](https://gitee.com/mindspore/mindspore/issues) 给MindSpore Lite开发人员处理。

2. 日志报错信息：

    ```cpp
    [mindspore/lite/tools/converter/parser/caffe/caffe_model_parser.cc:**] ConvertLayers] parse node **** failed.
    ```

    - 问题分析：转换工具支持该算子转换，但是不支持该算子的某种特殊属性或参数导致模型转换失败（示例日志以caffe为例，其他框架日志信息相同）。
    - 解决方法：可以尝试通过继承API接口[NodeParser](https://mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_converter.html#nodeparser) 添加自定义算子parser并通过[NodeParserRegistry](https://mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore_registry.html#nodeparserregistry) 进行Parser注册；或者在社区提[ISSUE](https://gitee.com/mindspore/mindspore/issues) 给MindSpore Lite开发人员处理。

3. 日志报错信息：

    ```cpp
    [mindspore/lite/src/lite_model.cc:**] ConstructModel] The model buffer is invalid and fail to create graph.
    [mindspore/lite/src/lite_model.cc:**] ImportFromBuffer] construct model failed.
    ```

    - 问题分析：从ms模型文件读取的缓存内容无效，导致图加载失败。
    - 解决方法：确认推理使用的模型是直接通过转换工具转出的ms模型文件，若模型文件为经过传输或下载得到的，可以通过检查md5值进行校验以查看ms模型文件是否损坏。

4. 日志报错信息：

    ```cpp
    [mindspore/lite/src/lite_model.cc:**] ConstructModel] Maybe this is a model transferred out using the conversion tool before 1.1.0.
    [mindspore/lite/src/lite_model.cc:**] ImportFromBuffer] construct model failed.
    ```

    - 问题分析：该ms模型文件所使用的转换工具版本较低，导致图加载失败。
    - 解决方法：请使用MindSpore Lite 1.1.0 以上的版本重新转出ms模型。

5. 日志报错信息：

    ```cpp
    WARNING [mindspore/lite/src/lite_model.cc:**] ConstructModel] model version is MindSpore Lite 1.2.0, inference version is MindSpore Lite 1.5.0 not equal
    [mindspore/lite/src/runtime/infer_manager.cc:**] KernelInferShape] Get infershape func failed! type: ****
    [mindspore/lite/src/scheduler.cc:**] ScheduleNodeToKernel] FindBackendKernel return nullptr, name: ****, type: ****
    [mindspore/lite/src/scheduler.cc:**] ScheduleSubGraphToKernels] schedule node return nullptr, name: ****, type: ****
    [mindspore/lite/src/scheduler.cc:**] ScheduleMainSubGraphToKernels] Schedule subgraph failed, index: 0
    [mindspore/lite/src/scheduler.cc:**] ScheduleGraphToKernels] ScheduleSubGraphToSubGraphKernel failed
    [mindspore/lite/src/scheduler.cc:**] Schedule] Schedule graph to kernels failed.
    [mindspore/lite/src/lite_session.cc:**] CompileGraph] Schedule kernels failed: -1.
    ```

    - 问题分析：推理使用的MindSpore Lite版本高于模型转换使用的转换工具版本，导致存在兼容性问题：版本升级可能会新增或移除某些算子，推理时缺少算子的实现。
    - 解决方法：使用和转换模型使用的转换工具版本相同的MindSpore Lite进行推理。通常情况下，MindSpore Lite推理兼容较低版本ms模型但是版本差异过大的情况下可能存在兼容性问题；同时，MindSpore Lite推理不保证向后兼容较高版本转换出的ms模型。

6. 日志报错信息：

    ```cpp
    [mindspore/lite/src/common/tensor_util.cc:**] CheckTensorsInvalid] The shape of tensor contains negative dimension, check the model and assign the input shape with method Resize().
    [mindspore/lite/src/lite_session.cc:**] RunGraph] CheckInputs failed.
    ```

    - 问题分析：ms模型的输入shape包含-1，即模型输入为动态shape，直接推理时由于shape无效导致推理失败。
    - 解决方法：MindSpore Lite在对包含动态shape输入的模型推理时要求指定合理的shape，使用benchmark工具可通过设置[inputShapes](https://mindspore.cn/lite/docs/zh-CN/master/use/benchmark_tool.html#id3) 参数指定，使用MindSpore Lite集成开发时可通过调用[Resize](https://mindspore.cn/lite/api/zh-CN/master/api_cpp/mindspore.html#resize) 方法设置。

7. 使用MindSpore Lite集成时对推理结果后处理后发现效果不理想，怀疑推理精度存在问题要如何定位？
    - 首先确认输入数据是否正确：在MindSpore Lite 1.3.0及之前版本ms模型的输入数据格式为NHWC，MindSpore Lite 1.5.0之后的版本支持[inputDataFormat](https://mindspore.cn/lite/docs/zh-CN/master/use/converter_tool.html) 参数设置输入数据格式为NHWC或NCHW，需要检查输入数据的格式确保和ms模型要求的输入格式一致；
    - 通过MindSpore Lite提供的基准测试工具[benchmark](https://mindspore.cn/lite/docs/zh-CN/master/use/benchmark_tool.html) 进行精度测试验证，日志如下则可能存在精度问题；否则MindSpore Lite推理精度正常，需要检查数据前/后处理过程是否有误。

      ```cpp
      Mean bias of all nodes/tensors is too big: **
      Compare output error -1
      Run MarkAccuracy error: -1
      ```

    - 若MindSpore Lite进行整网推理存在精度问题，可以通过benchmark工具的[Dump功能](https://mindspore.cn/lite/docs/zh-CN/master/use/benchmark_tool.html#dump) 保存算子层输出，和原框架推理结果进行对比进一步定位出现精度异常的算子。
    - 针对存在精度问题的算子，可以下载[MindSpore源码](https://gitee.com/mindspore/mindspore) 检查算子实现并构造相应单算子网络进行调试与问题定位；也可以在MindSpore社区[提ISSUE](https://gitee.com/mindspore/mindspore/issues) 给MindSpore Lite的开发人员处理。

8. MindSpore Lite使用fp32推理结果正确，但是fp16推理结果出现Nan或者Inf值怎么办？
    - 结果出现Nan或者Inf值一般为推理过程中出现数值溢出，可以查看模型结构，筛选可能出数值溢出的算子层，然后通过benchmark工具的[Dump功能](https://mindspore.cn/lite/docs/zh-CN/master/use/benchmark_tool.html#dump) 保存算子层输出确认出现数值溢出的算子。
    - MindSpore Lite 1.5.0之后版本提供混合精度推理能力，在整网推理优先使用fp16时支持设置某一层算子进行fp32推理，具体使用方法可参考官网文档[混合精度运行](https://mindspore.cn/lite/docs/zh-CN/master/use/runtime_cpp.html#id13) ，通过将溢出层设置为fp32避免在fp16推理时出现的整网推理精度问题。
