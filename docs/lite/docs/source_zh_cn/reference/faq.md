# 问题定位指南

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_zh_cn/reference/faq.md)

## 概述

在MindSpore Lite使用中遇到问题时，可首先查看日志，多数场景下的问题可以通过日志报错信息直接定位（通过设置环境变量[GLOG_v](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/env_var_list.html) 调整日志等级可以打印更多调试日志），这里简单介绍几种常见报错场景的问题定位与解决方法。

> 1. 因不同版本中日志行号可能存在差异，下述示例日志报错信息中的行号信息均用”**”表示；
> 2. 示例日志中只列出了通用信息，其他涉及具体场景的信息均用“****”表示。

## 模型转换失败

1. 模型路径错误或文件损坏，日志报错信息：

    ```cpp
    WARNING: Logging before InitGoogleLogging() is written to STDERR
    [WARNING] LITE(11979,7fbdc90a8ec0,converter_lite):2021-12-13-16:20:49.506.071 [mindspore/lite/tools/common/protobuf_utils.cc:94] ReadProtoFromBinaryFile] Parse ***.onnx failed.
    [ERROR] LITE(11979,7fbdc90a8ec0,converter_lite):2021-12-13-16:20:49.506.122 [mindspore/lite/build/tools/converter/parser/onnx/onnx_op_parser.cc:3079] InitOriginModel] Read onnx model file failed, model path: ./ml_audio_kit_vocals_resunet.onnx
    [ERROR] LITE(11979,7fbdc90a8ec0,converter_lite):2021-12-13-16:20:49.506.131 [mindspore/lite/build/tools/converter/parser/onnx/onnx_op_parser.cc:3026] Parse] init origin model failed.
    [ERROR] LITE(11979,7fbdc90a8ec0,converter_lite):2021-12-13-16:20:49.506.137 [mindspore/lite/tools/converter/converter.cc:64] BuildFuncGraph] Get funcGraph failed for fmk: ONNX
    [ERROR] LITE(11979,7fbdc90a8ec0,converter_lite):2021-12-13-16:20:49.506.143 [mindspore/lite/tools/converter/converter.cc:133] Convert] Parser/Import model return nullptr
    [ERROR] LITE(11979,7fbdc90a8ec0,converter_lite):2021-12-13-16:20:49.506.162 [mindspore/lite/tools/converter/converter.cc:209] RunConverter] CONVERT RESULT FAILED:-1 Common error code.
    CONVERT RESULT FAILED:-1 Common error code.
    ```

    - 问题分析：根据报错信息可以看出，模型导入直接报错，还没进入转换流程就退出。

    - 解决方法：这种报错首先排除模型转换时输入的命令中，模型路径是否正确，如果路径正确，那么需要排除模型是否破损，破损的文件无法解析，会直接退出。

2. 存在不支持的算子，日志报错信息：

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
    - 解决方法：对于不支持的算子可以尝试通过继承API接口[NodeParser](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#nodeparser) 自行添加parser并通过[NodeParserRegistry](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#nodeparserregistry) 进行Parser注册；或者在社区提[ISSUE](https://gitee.com/mindspore/mindspore/issues) 给MindSpore Lite开发人员处理。

3. 存在不支持的算子，日志报错信息：

    ```cpp
    [mindspore/lite/tools/converter/parser/caffe/caffe_model_parser.cc:**] ConvertLayers] parse node **** failed.
    ```

    - 问题分析：转换工具支持该算子转换，但是不支持该算子的某种特殊属性或参数导致模型转换失败（示例日志以caffe为例，其他框架日志信息相同）。
    - 解决方法：可以尝试通过继承API接口[NodeParser](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_converter.html#nodeparser) 添加自定义算子parser并通过[NodeParserRegistry](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore_registry.html#nodeparserregistry) 进行Parser注册；或者在社区提[ISSUE](https://gitee.com/mindspore/mindspore/issues) 给MindSpore Lite开发人员处理。

## 训练后量化转换失败

### 全量化转换失败

1. 针对动态Shape的模型，需要在[转换命令](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html#参数说明)上设置`--inputShape=<INPUTSHAPE>`，例如

    ```
    ./converter_lite --fmk=ModelType --modelFile=ModelFilePath --outputFile=ConvertedModelPath --configFile=/mindspore/lite/tools/converter/quantizer/config/full_quant.cfg --inputShape=intput_1:1,224,224,3;intput_2:1,48;
    ```

2. 针对多Batch的模型，需要无法直接使用数据预处理的功能，用户需要提前预处理矫正数据集，然后以`BIN`的形式设置校准数据集。

## 模型推理失败

### 图加载失败

1. 模型文件错误，日志报错信息：

    ```cpp
    [mindspore/lite/src/lite_model.cc:**] ConstructModel] The model buffer is invalid and fail to create graph.
    [mindspore/lite/src/lite_model.cc:**] ImportFromBuffer] construct model failed.
    ```

    - 问题分析：从ms模型文件读取的缓存内容无效，导致图加载失败。
    - 解决方法：确认推理使用的模型是直接通过转换工具转出的ms模型文件，若模型文件为经过传输或下载得到的，可以通过检查md5值进行校验以查看ms模型文件是否损坏。

2. 模型文件和推理包版本不兼容，日志报错信息：

    ```cpp
    [mindspore/lite/src/lite_model.cc:**] ConstructModel] Maybe this is a model transferred out using the conversion tool before 1.1.0.
    [mindspore/lite/src/lite_model.cc:**] ImportFromBuffer] construct model failed.
    ```

    - 问题分析：该ms模型文件所使用的转换工具版本较低，导致图加载失败。
    - 解决方法：请使用MindSpore Lite 1.1.0 以上的版本重新转出ms模型。

### CPU推理问题

#### 图编译失败

1. 模型文件和推理包版本不兼容，日志报错信息：

    ```cpp
    WARNING [mindspore/lite/src/lite_model.cc:**] ConstructModel] model version is MindSpore Lite 1.2.0, inference version is MindSpore Lite 1.5.0 not equal
    [mindspore/lite/src/litert/infer_manager.cc:**] KernelInferShape] Get infershape func failed! type: ****
    [mindspore/lite/src/scheduler.cc:**] ScheduleNodeToKernel] FindBackendKernel return nullptr, name: ****, type: ****
    [mindspore/lite/src/scheduler.cc:**] ScheduleSubGraphToKernels] schedule node return nullptr, name: ****, type: ****
    [mindspore/lite/src/scheduler.cc:**] ScheduleMainSubGraphToKernels] Schedule subgraph failed, index: 0
    [mindspore/lite/src/scheduler.cc:**] ScheduleGraphToKernels] ScheduleSubGraphToSubGraphKernel failed
    [mindspore/lite/src/scheduler.cc:**] Schedule] Schedule graph to kernels failed.
    [mindspore/lite/src/lite_session.cc:**] CompileGraph] Schedule kernels failed: -1.
    ```

    - 问题分析：推理使用的MindSpore Lite版本高于模型转换使用的转换工具版本，导致存在兼容性问题：版本升级可能会新增或移除某些算子，推理时缺少算子的实现。
    - 解决方法：使用和转换模型使用的转换工具版本相同的MindSpore Lite进行推理。通常情况下，MindSpore Lite推理兼容较低版本ms模型但是版本差异过大的情况下可能存在兼容性问题；同时，MindSpore Lite推理不保证向后兼容较高版本转换出的ms模型。

2. 模型输入为动态shape，日志报错信息：

    ```cpp
    [mindspore/lite/src/common/tensor_util.cc:**] CheckTensorsInvalid] The shape of tensor contains negative dimension, check the model and assign the input shape with method Resize().
    [mindspore/lite/src/lite_session.cc:**] RunGraph] CheckInputs failed.
    ```

    - 问题分析：ms模型的输入shape包含-1，即模型输入为动态shape，直接推理时由于shape无效导致推理失败。
    - 解决方法：MindSpore Lite在对包含动态shape输入的模型推理时要求指定合理的shape，使用benchmark工具时可通过设置[inputShapes](https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html#参数说明) 参数指定，使用MindSpore Lite集成开发时可通过调用[Resize](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#resize) 方法设置。

### OpenCL GPU 推理问题

#### 图编译失败

1. 模型文件和推理包版本不兼容，日志报错信息：

    ```cpp
    ERROR [mindspore/lite/src/lite_session.cc:1539] LoadModelByBuff] Please enable runtime convert.
    ERROR [mindspore/lite/src/lite_session.cc:1598] LoadModelAndCompileByPath] Read model file failed
    ERROR [mindspore/lite/src/cxx_api/model/model_impl.cc:93] Build] Init session failed
    ERROR [mindspore/lite/tools/benchmark/benchmark_unified_api.cc:845] RunBenchmark] ms_model_.Build failed while running
    ERROR [mindspore/lite/tools/benchmark/run_benchmark.cc:80] RunBenchmark] Run Benchmark Q888_CV_new_detect.pb.ms Failed : -1
    ms_model_.Build failed while running Run Benchmark Q888_CV_new_detect.pb.ms Failed : -1
    ```

    - 问题分析：没有使用模型转换工具对原始模型进行转换，或者推理使用的MindSpore Lite版本高于模型转换使用的转换工具版本，导致存在兼容性问题：版本升级可能会新增或移除某些算子，推理时缺少算子的实现。
    - 解决方法：使用和转换模型使用的转换工具版本相同的MindSpore Lite进行推理。通常情况下，MindSpore Lite推理兼容较低版本ms模型，但是版本差异过大的情况下可能存在兼容性问题；同时，MindSpore Lite推理不保证向后兼容较高版本转换出的ms模型。

#### 图执行失败

1. 模型输入为动态shape，日志报错信息：

    ```cpp
    WARNING [mindspore/lite/src/litert/kernel/opencl/kernel/arithmetic_self.cc:40] CheckSpecs]  only support dim = 4 or 2 but your dim = 3
    ERROR [mindspore/lite/src/litert/kernel/opencl/opencl_kernel.cc:222] ReSize] ReSize failed for check kernel specs!
    ERROR [mindspore/lite/src/inner_kernel.cc:81] Execute] run kernel PreProcess failed, name: Exp_1234
    ERROR [mindspore/lite/src/litert/gpu/opencl/opencl_executor.cc:70] RunOrTune] run kernel failed, name: Exp_1234
    ERROR [mindspore/lite/src/litert/kernel/opencl/opencl_subgraph.cc:574] Execute] Run opencl executor failed: -1
    ERROR [mindspore/lite/src/lite_mindrt.h:58] RunKernel] run kernel failed, name: GpuSubGraph4_8
    WARNING [mindspore/lite/src/litert/gpu/opencl/opencl_allocator.cc:475] MapBuffer] Host ptr no need map
    WARNING [mindspore/lite/src/litert/gpu/opencl/opencl_allocator.cc:525] UnmapBuffer] Host ptr do not mapped
    ```

    - 问题分析：ms模型的输入shape包含-1，即模型输入为动态shape，GPU推理时在图编译阶段会跳过和Shape相关的算子规格检查，默认GPU支持该算子，并在Predict阶段会再次进行算子规格检查，如果算子规格检查为不支持，则报错退出。
    - 解决方法：由于存在不支持的GPU算子，部分报错用户可根据提示修改模型中的算子类型或参数类型来进行规避，但大部分可能需要通过在MindSpore社区[提ISSUE](https://gitee.com/mindspore/mindspore/issues) 来通知开发人员进行代码修复和适配。

2. Map buffer类错误

    ```cpp
    WARNING [mindspore/lite/src/litert/gpu/opencl/opencl_allocator.cc:494] MapBuffer] Map buffer failed, can not found buffer or already mapped, dev_ptr=0x7244929ff0, host_ptr=0x722fbacd80
    ERROR [mindspore/lite/src/litert/kernel/arm/base/strided_slice.cc:179] FastRun] input_ptr_ must not be null!
    ERROR [mindspore/lite/src/inner_kernel.cc:88] Execute] run kernel failed, name: Slice_1147
    ERROR [mindspore/lite/src/sub_graph_kernel.cc:223] Execute] run kernel failed, name: Slice_1147
    ERROR [mindspore/lite/src/lite_mindrt.h:56] RunKernel] run kernel failed, name: CpuFP32SubGraph0_1
    ERROR [mindspore/lite/src/mindrt_executor.cc:193] Run] MindrtRun failed
    ERROR [mindspore/lite/src/lite_session.cc:709] RunGraph] RunGraph failed : -1
    ERROR [mindspore/lite/src/cxx_api/model/model_impl.cc:294] Predict] Run graph failed.
    ERROR [mindspore/lite/tools/benchmark/benchmark_unified_api.cc:721] MarkAccuracy] Inference error
    Inference error Run MarkAccuracy error: -1
    ```

    - 问题分析：推理阶段为了提升性能会忽略OpenCL算子执行结束后的Event检查，而OpenCL中Enqueue类函数会默认插入Event检查，如果有OpenCL算子执行出错，会在Map阶段返回错误。
    - 解决办法：由于OpenCL算子存在BUG，建议通过在MindSpore社区[提ISSUE](https://gitee.com/mindspore/mindspore/issues) 来通知开发人员进行代码修复和适配。

### TensorRT GPU 推理问题

#### 图编译失败

1. 模型输入为动态 shape，或者模型有 shape 算子，会有 Dimensions 相关日志报错信息：

    ```cpp
    ERROR [mindspore/lite/src/delegate/tensorrt/tensorrt_runtime.h:31] log] Parameter check failed at: optimizationProfile.cpp::setDimensions::119, condition: std::all_of(dims.d, dims.d + dims.nbDims, [](int x) { return x >= 0; })
    ERROR [mindspore/lite/src/delegate/tensorrt/tensorrt_subgraph.cc:219] ParseInputDimsProfile] setDimensions of kMIN failed for input
    ERROR [mindspore/lite/src/delegate/tensorrt/tensorrt_runtime.h:31] log] xxx: xxx size cannot have negative dimension, size = [-1]
    ```

    - 问题分析：TensorRT GPU 构图暂不支持有动态 shape 的模型，具体情况为模型的输入 shape 包含-1，或者模型中包含 shape 算子。
    - 解决方法：在使用 converter 将模型转换成ms时，需要在[转换命令](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html#参数说明)上设置`--inputShape=<INPUTSHAPE>`，指定输入 tensor 的 shape 信息。如需在推理时改变输入 shape，使用 benchmark 工具时可通过设置[inputShapes](https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html#参数说明) 参数指定，使用 MindSpore Lite 集成开发时可通过调用[Resize](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#resize) 方法设置。注意：[Resize](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#resize)输入的 shape 维度必须要小于等于 [Build](https://mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_cpp/mindspore.html#build)模型的维度。

#### 图执行失败

1. 离线 broadcast 类型算子，不支持 resize，日志现象会报错在某个算子的输入维度不匹配，例如：

    ```cpp
    ERROR [mindspore/lite/src/delegate/tensorrt/tensorrt_runtime.h:31] log] xxx: dimensions not compatible for xxx
    ERROR [mindspore/lite/src/delegate/tensorrt/tensorrt_runtime.h:31] log] shapeMachine.cpp (252) - Shape Error in operator(): broadcast with incompatible Dimensions
    ERROR [mindspore/lite/src/delegate/tensorrt/tensorrt_runtime.h:31] log] Instruction: CHECK_BROADCAST xx xx
    ERROR [mindspore/lite/src/delegate/tensorrt/tensorrt_subgraph.cc:500] Execute] TensorRT execute failed.
    ```

    - 问题分析：有算子在离线 converter 时，通过指定`--inputShape=<INPUTSHAPE>`自动做了离线 broadcast。例如 ones like，会依据输入的 shape 信息，将1 broadcast 成对应的常量tensor，此时再将输入 resize 成不同维度时，网络中对输入 tensor 维度敏感的算子（例如concat、matmul等）就会出现报错。
    - 解决方法：将此类算子替换为一个模型的输入，推理时通过内存拷贝的方式赋值，并在 resize 的时候指定对应的 shape 信息。

### NPU推理问题

#### 图编译失败

1. NPU图编译失败，通过工具抓取后台日志，并在日志中搜索“**MS_LITE**”关键字，得到报错提示如下：

    ```cpp
    MS_LITE : [mindspore/lite/src/delegate/npu/npu_subgraph.cc:**] BuildIRModel] Build IR model failed.
    MS_LITE : [mindspore/lite/src/delegate/npu/npu_subgraph.cc:**] Init] Build IR model failed.
    MS_LITE : [mindspore/lite/src/delegate/npu/npu_graph.cc:**] CreateNPUSubgraphKernel] NPU Subgraph Init failed.
    MS_LITE : [mindspore/lite/src/delegate/npu/npu_delegate.cc:**] Build] Create NPU Graph failed.
    ```

    - 问题分析：此报错为NPU在线构图失败。
    - 解决方法：由于构图系通过调用[HiAI DDK](https://developer.huawei.com/consumer/cn/doc/development/HiAI-Library/ddk-download-0000001053590180) 的接口完成，因此报错一般会首先出现在HiAI的错误日志中，部分报错用户可根据提示修改模型中的算子类型或参数类型来进行规避，但大部分可能需要通过在MindSpore社区[提ISSUE](https://gitee.com/mindspore/mindspore/issues) 来通知开发人员进行代码修复和适配。因此，我们下面仅给出较为常见的HiAI报错信息，以便您在社区提问时对问题有更清晰的描述，并加快问题定位的效率。

    （1）在日志中搜索“**E AI_FMK**”关键字，若在“MS_LITE”日志报错之前的位置处得到报错日志如下：

      ```cpp
      AI_FMK  : /conv_base_op_builder.cpp CheckShapeMatch(**)::"Input Channel ** does not match convolution weight channel ** * group **."
      AI_FMK  : /conv_base_op_builder.cpp Init(**)::"Shape of op **** does not match"
      AI_FMK  : /op_builder.cpp BuildOutputDesc(**)::""Init" failed. Node: ****."
      ```

    说明构图时多插或漏插了用于Tensor format转换的Transpose算子，导致卷积的输入Shape和权重的格式无法匹配。

    （2）在日志中搜索“**E AI_FMK**”关键字，若在“MS_LITE”日志报错之前的位置处得到报错日志如下：

      ```cpp
      AI_FMK  : /model_compatibility_check.cpp GetIRGraphCompatibilityCheckResult(**)::"Node **** type **** don't support!"
      AI_FMK  : /model_compatibility_check.cpp CheckIRGraphCompatibility(**)::"CompleteExecuteDeviceConfig CheckIRGraphCompatibility failed"
      ```

    说明HiAI ROM和当前MindSpore Lite使用的HiAI DDK版本存在算子兼容性问题，报错信息中提示的算子不支持。您可以尝试通过更新手机系统来升级HiAI ROM、替换当前不支持的算子来规避，或在开源社区进行反馈。

#### 图执行失败

1. NPU推理失败，通过工具抓取后台日志，并在日志中搜索“**MS_LITE**”关键字，得到报错提示如下：

      ```cpp
      MS_LITE : [mindspore/lite/src/delegate/npu/npu_executor.cc:**] Run] NPU Process failed. code is 1
      MS_LITE : [mindspore/lite/src/delegate/npu/npu_graph.cc:**] Execute] NPU Subgraph **** execute failed.
      MS_LITE : [mindspore/lite/src/lite_mindrt.h:**] RunKernel] run kernel failed, name: ****
      ```

    - 问题分析：此报错为NPU执行推理失败。
    - 解决方法：由于NPU模型的底层推理实际由HiAI完成，因此报错同样会首先出现在HiAI的错误日志中，我们下面仅给出较为常见的HiAI报错信息，以方便您定位。

    在日志中搜索“**E AI_FMK**”关键字，若在“**MS_LITE**”日志报错之前的位置处得到报错日志如下：

      ```cpp
      AI_FMK  : /common_memory_allocator.cpp Allocate(**)::"Call rt api failed, ret: ****"
      ```

    再搜索“**DEVMM**”关键字，若在上一条日志的前10行左右，显示报错日志如下：

      ```cpp
      /vendor/bin/hiaiserver: [DEVMM][E] DevmmJudgeAllocSize:** the alloc memory size exceeds the specification limit, already alloc total size = 0x3ff95000
      ```

    说明NPU的内存申请超出了限制。请确认模型文件是否比较大，或模型中存在shape较大的tensor。根据HiAI[官方要求](https://developer.huawei.com/consumer/cn/doc/development/hiai-References/modelbuildoptions-0000001139374903) ，单张NPU子图的大小不应超过200MB，数组的内存申请也不要超过NPU的显存大小，比如在本例的日志中，NPU可申请的显存上限为1GB。若仍需要跑NPU，请调整模型结构、将模型进行拆分或者调整tensor的shape大小。

## 模型推理精度问题

1. 使用MindSpore Lite集成时对推理结果后处理后发现效果不理想，怀疑推理精度存在问题要如何定位？
    - 首先确认输入数据是否正确：在MindSpore Lite 1.3.0及之前版本ms模型的输入数据格式为NHWC，MindSpore Lite 1.5.0之后的版本支持[inputDataFormat](https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/converter/converter_tool.html) 参数设置输入数据格式为NHWC或NCHW，需要检查输入数据的格式确保和ms模型要求的输入格式一致；
    - 通过MindSpore Lite提供的基准测试工具[benchmark](https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html) 进行精度测试验证，日志如下则可能存在精度问题；否则MindSpore Lite推理精度正常，需要检查数据前/后处理过程是否有误。

      ```cpp
      Mean bias of all nodes/tensors is too big: **
      Compare output error -1
      Run MarkAccuracy error: -1
      ```

    - 若MindSpore Lite进行整网推理存在精度问题，可以通过benchmark工具的[Dump功能](https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html#dump功能) 保存算子层输出，和原框架推理结果进行对比进一步定位出现精度异常的算子。
    - 针对存在精度问题的算子，可以下载[MindSpore源码](https://gitee.com/mindspore/mindspore) 检查算子实现并构造相应单算子网络进行调试与问题定位；也可以在MindSpore社区[提ISSUE](https://gitee.com/mindspore/mindspore/issues) 给MindSpore Lite的开发人员处理。

2. MindSpore Lite使用fp32推理结果正确，但是fp16推理结果出现NaN或者Inf值怎么办？
    - 结果出现NaN或者Inf值一般为推理过程中出现数值溢出，可以查看模型结构，筛选可能出数值溢出的算子层，然后通过benchmark工具的[Dump功能](https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html#dump功能) 保存算子层输出确认出现数值溢出的算子。
    - MindSpore Lite 1.5.0之后版本提供混合精度推理能力，在整网推理优先使用fp16时支持设置某一层算子进行fp32推理，具体使用方法可参考官网文档[混合精度运行](https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/infer/runtime_cpp.html#混合精度运行) ，通过将溢出层设置为fp32避免在fp16推理时出现的整网推理精度问题。

3. MindSpore Lite使用fp32和fp16推理结果同时出现NaN或者Inf值怎么办？
    - 问题分析：检查整个网络存在做除法操作的算子。在做推理时如果执行了除法操作，且除数是0时容易出现NaN值。比如下面的网络结构，如果该网络用于输入数据不做归一化的场景，输入数据在0-255范围，则会出现NaN值，原因在于不做归一化时，输入数据比较大，导致matmul的输出值会很大，导致Tanh激活函数输出等于1，最终导致Div算子除0。但如果网络输入数据做了归一化，Tanh激活函数不等于1，网络推理数据不存在NaN值。

      ![image-20211214191139062](./images/troubleshooting_Fp32_NAN.png)

    - 解决方法：如果是输入数据太大导致的，建议训练时把网络输入数据做归一化。如果输入数据归一化了还存在NaN值，这种需要通过benchmark工具的[Dump功能](https://mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/benchmark_tool.html#dump功能) 保存算子层输出确认出现数值溢出的算子，具体分析。

## 模型推理性能问题

1. 为何将设备指定为NPU后，实际推理性能和CPU并没有差别？
    - 若设备不支持NPU但在上下文中进行了指定，则模型并不会运行在NPU上，而是会自动切换到CPU来执行，此时的推理性能自然与CPU一致。可以通过工具（如adb logcat）抓取后台日志，并在日志中搜索“**MS_LITE**”关键字，以确认设备是否支持NPU。常见的提示信息及说明如下：

      ```cpp
      MS_LITE : [mindspore/lite/src/delegate/npu/npu_manager.cc:**] IsSupportNPU] The current devices NOT SUPPORT NPU.
      ```

    - 若日志中仅包含以上这一行提示，请检查您的设备是否为包含海思麒麟处理器的华为设备，否则不支持NPU。

      ```cpp
      MS_LITE : [mindspore/lite/src/delegate/npu/npu_manager.cc:**] IsKirinChip] Unsupported KirinChip ***.
      MS_LITE : [mindspore/lite/src/delegate/npu/npu_manager.cc:**] IsSupportNPU] The current devices NOT SUPPORT NPU.
      ```

    - 若日志中包含以上这两行提示，说明您的设备虽然使用的是麒麟芯片，但芯片型号不支持NPU。当前支持NPU的麒麟处理器芯片为：Kirin 810、Kirin 820、Kirin 985及其他高于此版本的型号。

      ```cpp
      MS_LITE : [mindspore/lite/src/delegate/npu/npu_manager.cc:**] CheckDDKVerGreatEqual] DDK Version 100.***.***.*** less than 100.320.011.019.
      MS_LITE : [mindspore/lite/src/delegate/npu/npu_manager.cc:**] IsSupportNPU] The current devices NOT SUPPORT NPU.
      ```

    - 若日志中包含以上这两行提示，说明您的设备虽然满足硬件要求，但系统的HiAI ROM版本不满足要求，同样无法运行NPU算子。当前MindSpore Lite要求HiAI ROM版本必须大于100.320.011.018。

      ```cpp
      MS_LITE : [mindspore/lite/src/delegate/npu/op/convolution_npu.cc:**] GetNPUConvOp] NPU does not support runtime inference shape.
      MS_LITE : [mindspore/lite/src/delegate/npu/op/npu_op.h:** GetNPUOp] NPU does not support runtime inference shape.
      ```

    - 若以上两条提示（或其中一条）在日志中出现多次，请确认您的模型输入是否为动态shape且在推理前对输入shape进行了指定，若是则不支持在NPU上运行，程序会自动切换到CPU执行。

2. 为何将设备指定为NPU后，实际推理性能比CPU还要差？
    - 绝大多数情况下，NPU的推理性能要大幅优于CPU，但在少数情况下会比CPU更劣：

    （1）检查模型中是否存在大量Pad或StridedSlice等算子，由于NPU中的数组格式与CPU有所不同，这类算子在NPU中运算时涉及数组的重排，因此相较CPU不存在任何优势，甚至劣于CPU。若确实需要在NPU上运行，建议尝试去除或替换此类算子。
    （2）通过工具（如adb logcat）抓取后台日志，搜索所有“**BuildIRModel build successfully**”关键字，发现相关日志出现了多次，说明模型在线构图时切分为了多张NPU子图，子图的切分一般都是由图中存在Transpose或/和当前不支持的NPU算子引起。目前我们支持最多20张子图的切分，子图数量越多，NPU的整体耗时增加越明显。建议比对MindSpore Lite当前支持的NPU[算子列表](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/reference/operator_list_lite.html)，在模型搭建时规避不支持的算子，或在MindSpore社区[提ISSUE](https://gitee.com/mindspore/mindspore/issues) 询问MindSpore Lite的开发人员。

## 使用Visual Studio相关问题

1. 使用静态库，运行时报Parameter的Creator函数找不到的错，日志报错信息：

    ```text
    ERROR [mindspore\lite\src\ops\populate\populate_register.h:**] GetParameterCreator] Unsupported parameter type in Create : **
    ERROR [mindspore\lite\src\scheduler.cc:**] InferNodeShape] parameter generator is nullptr.
    ERROR [mindspore\lite\src\scheduler.cc:**] InferSubGraphShape] InferShape failed, name: **, type: **
    ERROR [mindspore\lite\src\scheduler.cc:**] SchedulePreProcess] op infer shape failed.
    ERROR [mindspore\lite\src\lite_session.cc:**] CompileGraph] Schedule kernels failed: -500
    ```

    - 问题分析：链接静态库，默认不会导入静态库中的所有符号，Parameter的Creator函数是通过全局静态对象注册到单例对象去的。

    - 解决方法：链接 Visual Studio 编译出的静态库时，需要在“属性->链接器->命令行->其他选项”中，加/WHOLEARCHIVE:libmindspore-lite.lib。

2. 模型校验失败，日志报错信息：

    ```text
    ERROR [mindspore\lite\src\lite_model.cc:**] ModelVerify] Model does not have inputs.
    ERROR [mindspore\lite\src\lite_model.cc:**] ConstructModel] ModelVerify failed.
    ERROR [mindspore\lite\src\lite_model.cc:**] ImportFromBuffer] construct model failed.
    ```

    - 问题分析：读取的模型文件不完整的。

    - 解决方法：使用 Visual Studio 编译器时，读入 model 流必须加 std::ios::binary。

## 使用Xcode构建APP相关问题

1. 利用Xcode使用framework包构建APP，运行GetParameterCreator函数报错不支持某种parameter，日志如下：

    ```text
    ERROR [mindspore/lite/src/ops/populate/populate_register.h:46] GetParameterCreator] Unsupported parameter type in Create : **
    ERROR [mindspore/lite/src/scheduler.cc:208] InferNodeShape] parameter generator is nullptr.
    ERROR [mindspore/lite/src/scheduler.cc:266] InferSubGraphShape] InferShape failed, name: **, type: **
    ERROR [mindspore/lite/src/scheduler.cc:78] SchedulePreProcess] op infer shape failed.
    ERROR [mindspore/lite/src/lite_session.cc:508] CompileGraph] Schedule kernels failed: -500
    ```

    - 问题分析：链接framework中的静态库，不会导入静态库中的所有符号，Parameter的Creator函数是通过全局静态对象注册到单例对象去的。

    - 解决方法：使用framework包在Xcode中构建APP时，需要在“Build Settings->Linking->Other Linker Flags”中，添加“mindspore_lite.framework/mindspore_lite”路径。

2. 利用Xcode使用framework包构建APP时，报错如下：

    ```text
    Undefined symbol:
    mindspore::session::LiteSession::CreateSession(**);
    ```

    - 问题分析：没有找到对应的符号，在Xcode工程中没有正确的导入framework包。

    - 解决方法：使用framework包在Xcode中构建APP时，需要在“Build Settings->Search Paths->Framework Search Paths”中，添加“mindspore_lite.framework”路径。同时在“Build Settings->Search Paths->User Header Search Paths”中，添加“mindspore_lite.framework/Headers”路径。

## 其他问题

<font size=3>**Q 为何将设备指定为GPU/NPU后，并没有起作用？**</font>

A：Device的优先级取决于配置的先后顺序，请确保Context里面GPU/NPU的配置在CPU前面。

<br/>

<font size=3>**Q：MindSpore Lite支持的日志级别有几种？怎么设置日志级别？**</font>

A：目前支持DEBUG、INFO、WARNING、ERROR四种日志级别，用户可以通过设置环境变量GLOG_v为0~3选择打印的日志级别，0~3分别对应DEBUG、INFO、WARNING和ERROR，默认打印WARNING和ERROR级别的日志。例如设置GLOG_v为1即可打印INFO及以上级别的日志。

<br/>

<font size=3>**Q：NPU推理存在什么限制？**</font>

A：目前NPU仅支持在系统ROM版本EMUI>=11、芯片支持包括Kirin 9000、Kirin 9000E、Kirin 990、Kirin 985、Kirin 820、Kirin 810等，具体约束和芯片支持请查看：<https://developer.huawei.com/consumer/cn/doc/development/hiai-Guides/supported-platforms-0000001052830507#section94427279718>。

<br/>

<font size=3>**Q：为什么使用裁剪工具裁剪后的静态库在集成时存在编译失败情况？**</font>

A：目前裁剪工具仅支持CPU和GPU的库，具体使用请查看[使用裁剪工具降低库文件大小](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0rc1/tools/cropper_tool.html)文档。

<br/>

<font size=3>**Q：MindSpore Lite推理是否会耗尽手机全部内存?**</font>

A：MindSpore Lite内置内存池有最大容量限制，为3GB，如果模型较大，超过最大容量限制，运行将会异常退出。

<br/>

<font size=3>**Q：MindSpore Lite的离线模型MS文件如何进行可视化，看到网络结构？**</font>

A：模型可视化开源仓库`Netron`已经支持查看MindSpore Lite模型（MindSpore版本 >= r1.2），请到Netron官网下载安装包[Netron](https://github.com/lutzroeder/netron)。

<br/>

<font size=3>**Q：MindSpore有量化推理工具么？**</font>

A：[MindSpore Lite](https://www.mindspore.cn/lite)支持云侧量化感知训练的量化模型的推理，MindSpore Lite converter工具提供训练后量化以及权重量化功能，且功能在持续加强完善中。

<br/>

<font size=3>**Q：MindSpore有轻量的端侧推理引擎么？**</font>

A：MindSpore轻量化推理框架MindSpore Lite已于r0.7版本正式上线，欢迎试用并提出宝贵意见，概述、教程和文档等请参考[MindSpore Lite](https://www.mindspore.cn/lite)

<br/>

<font size=3>**Q：针对编译JAVA库时出现 `sun.security.validator.ValidatorException: PKIX path building failed: sun.security.provider.certpath.SunCertPathBuilderException: unable to find valid certification path to requested target` 问题时如何解决？**</font>

A：需要使用keytool工具将相关网站的安全证书导入java的cacerts证书库 `keytool -import -file "XX.cer" -keystore ${JAVA_HOME}/lib/security/cacerts" -storepass changeit`。

<br/>

<font size=3>**Q：MindSpore Lite云上推理版本在GPU环境执行推理，出现`libpython3.7m.so.1.0: cannot open shared object file`问题时如何解决？**</font>

A：`libpython3.7.m.so.1.0`是Python的so，默认在系统的lib目录下，如果没有，需要通过`export LD_LIBRARY_PATH=${USERPATH}/lib:{LD_LIBRARY_PATH}`进行配置，其中`${USERPATH}/lib`是`libpython3.7.m.so.1.0`的所在路径。
<br/>