# 离线构建自定义算子

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/lite/docs/source_zh_cn/advanced/third_party/converter_register.md)

## 概述

MindSpore Lite的[转换工具](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0/converter/converter_tool.html)除了基本的模型转换功能之外，还支持用户对模型进行自定义的优化与构建，生成用户自定义算子的模型。

我们提供了一套注册机制，允许用户基于转换工具进行能力扩展：包括节点解析扩展、模型解析扩展以及图优化扩展，用户可以根据自身的需要对模型实现自定义的解析与融合优化。

节点解析扩展：用户自定义模型中某一节点的解析过程，支持ONNX、CAFFE、TF、TFLITE。接口可参考[NodeParser](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_converter.html#nodeparser)、[NodeParserRegistry](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#nodeparserregistry)。
模型解析扩展：用户自定义模型的整个解析过程，支持ONNX、CAFFE、TF、TFLITE。接口可参考[ModelParser](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_converter.html#modelparser)、[ModelParserRegistry](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#modelparserregistry)。
图优化扩展：模型解析之后，将获得MindSpore定义的图结构，用户可基于此结构自定义图的优化过程。接口可参考[PassBase](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#passbase)、[PassPosition](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#passposition)、[PassRegistry](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#passregistry)。

> 节点解析扩展需要依赖flatbuffers和protobuf及三方框架的序列化文件，并且flatbuffers和protobuf需要与发布件采用的版本一致，序列化文件需保证兼容发布件采用的序列化文件。发布件中不提供flatbuffers、protobuf及序列化文件，用户需自行编译，并生成序列化文件。用户可以从[MindSpore仓](https://gitee.com/mindspore/mindspore/tree/v2.6.0)中获取[flatbuffers](https://gitee.com/mindspore/mindspore/blob/v2.6.0/cmake/external_libs/flatbuffers.cmake)、[probobuf](https://gitee.com/mindspore/mindspore/blob/v2.6.0/cmake/external_libs/protobuf.cmake)、[ONNX原型文件](https://gitee.com/mindspore/mindspore/tree/v2.6.0/third_party/proto/onnx)、[CAFFE原型文件](https://gitee.com/mindspore/mindspore/tree/v2.6.0/third_party/proto/caffe)、[TF原型文件](https://gitee.com/mindspore/mindspore/tree/v2.6.0/third_party/proto/tensorflow)和[TFLITE原型文件](https://gitee.com/mindspore/mindspore/blob/v2.6.0/mindspore/lite/tools/converter/parser/tflite/schema.fbs)。
>
> MindSpore Lite还提供了一系列的注册宏，以便于用户侧的扩展接入转换工具。注册宏包括节点解析注册[REG_NODE_PARSER](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#reg-node-parser)、模型解析注册[REG_MODEL_PARSER](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#reg-model-parser)、图优化注册[REG_PASS](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#reg-pass)、图优化调度注册[REG_SCHEDULED_PASS](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#reg-scheduled-pass)。

MindSpore Lite转换工具的扩展能力，目前仅支持Linux系统。

本章节将通过MindSpore Lite转换工具扩展功能的示例程序，涵盖节点扩展案例、优化扩展案例以及编译链接全流程，来使用户能够快速了解转换工具的扩展功能的使用。

> 鉴于模型解析扩展是模块化的扩展能力，本章对其不做详细介绍，但会提供一个简化的单元案例，以供用户参考。

本章节以[add.tflite](https://download.mindspore.cn/model_zoo/official/lite/quick_start/add.tflite)模型为例。该模型仅包含一个简单的Add算子，通过自定义的节点解析、图优化，将Add算子转化为[Custom算子](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0/advanced/third_party/register_kernel.html#custom算子)，最终输出Custom单算子模型。

相关代码放置在[mindspore/lite/examples/converter_extend](https://gitee.com/mindspore/mindspore/tree/v2.6.0/mindspore/lite/examples/converter_extend)路径下。

## 节点扩展

1. 自定义节点解析：用户需继承[NodeParser](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_converter.html#nodeparser)，继而根据不同的框架，选择不同的重载接口。

2. 节点解析注册：用户调用注册接口[REG_NODE_PARSER](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#reg-node-parser)，完成自定义的节点解析接入转换工具。

```c++
class AddParserTutorial : public NodeParser {  // 继承基类
 public:
  AddParserTutorial() = default;
  ~AddParserTutorial() = default;
  ops::PrimitiveC *Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,            // 重载接口
                         const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                         const std::unique_ptr<tflite::ModelT> &tflite_model) override;
};

REG_NODE_PARSER(kFmkTypeTflite, ADD, std::make_shared<AddParserTutorial>());     // 调用注册接口
```

示例代码请参考[node_parser](https://gitee.com/mindspore/mindspore/tree/v2.6.0/mindspore/lite/examples/converter_extend/node_parser)。

## 模型扩展

示例代码请参考MindSpore仓模型扩展的单元案例[ModelParserRegistryTest](https://gitee.com/mindspore/mindspore/blob/v2.6.0/mindspore/lite/test/ut/tools/converter/registry/model_parser_registry_test.cc)。

### 优化扩展

1. 自定义优化：用户需继承[PassBase](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#passbase)，重载Execute接口函数[Execute](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#execute)。

2. 优化注册：调用优化的注册接口[REG_PASS](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0/api_cpp/mindspore_registry.html#reg-pass)，完成自定义把用户自己实现的Pass类注册进MindSpore Lite里。

```c++
class PassTutorial : public registry::PassBase {  // 继承基类
 public:
  PassTutorial() : PassBase("PassTutorial") {}

  ~PassTutorial() = default;

  bool Execute(const api::FuncGraphPtr &func_graph) override;     // 重载接口

 private:
  AnfNodePtr CreateCustomOp(const api::FuncGraphPtr func_graph, const CNodePtr &cnode);
};

using mindspore::registry::POSITION_BEGIN;            // 选择调度位置
REG_PASS(PassTutorial, opt::PassTutorial)             // 注册扩展类
REG_SCHEDULED_PASS(POSITION_BEGIN, {"PassTutorial"})  // 注册调度逻辑
```

示例代码可参考[pass](https://gitee.com/mindspore/mindspore/tree/v2.6.0/mindspore/lite/examples/converter_extend/pass)。

> 在离线转换阶段，我们会对模型的每一个节点的输出张量进行推断，包括输出张量的Format、DataType以及Shape，因此，离线转换阶段，用户需提供自己实现的算子的推断过程，这里用户可以参考[算子Infershape扩展](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0/infer/runtime_cpp.html#扩展使用)说明，示例代码可参考[infer](https://gitee.com/mindspore/mindspore/tree/v2.6.0/mindspore/lite/examples/converter_extend/infer)。

## 示例演示

### 编译

- 环境要求

    - 系统环境：Linux x86_64，推荐使用Ubuntu 18.04.02LTS
    - 编译依赖：
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- 编译准备

  MindSpore Lite的发布件不会提供其他框架下的序列化文件，因此，用户需自行编译获得，请参考[概述](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0/advanced/third_party/converter_register.html#概述)。

  本示例采用的是tflite模型，用户需编译[flatbuffers](https://gitee.com/mindspore/mindspore/blob/v2.6.0/cmake/external_libs/flatbuffers.cmake)，从[MindSpore仓](https://gitee.com/mindspore/mindspore/tree/v2.6.0)中获取[TFLITE原型文件](https://gitee.com/mindspore/mindspore/blob/v2.6.0/mindspore/lite/tools/converter/parser/tflite/schema.fbs)，最终生成tflite的序列化文件。

  在`mindspore/lite/examples/converter_extend`目录下创建`schema`文件目录，继而将生成的序列化文件置于`schema`目录下。

- 编译构建

  在`mindspore/lite/examples/converter_extend`目录下执行[build.sh](https://gitee.com/mindspore/mindspore/blob/v2.6.0/mindspore/lite/examples/converter_extend/build.sh)，将自动下载MindSpore Lite发布件并编译Demo。

  ```bash
  bash build.sh
  ```

  > 若使用该build脚本下载MindSpore Lite发布件失败，请手动下载硬件平台为CPU、操作系统为Ubuntu-x64的MindSpore Lite发布件[mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/zh-CN/r2.6.0/use/downloads.html)，将解压后`tools/converter/lib`目录、`tools/converter/include`目录拷贝到`mindspore/lite/examples/converter_extend`目录下。
  >
  > 通过手动下载并且将文件放到指定位置后，需要再次执行`build.sh`脚本才能完成编译构建。

- 编译输出

  在`mindspore/lite/examples/converter_extend/build`目录下生成了`libconverter_extend_tutorial.so`的动态库。

### 执行程序

1. 拷贝动态库

   将生成的`libconverter_extend_tutorial.so`动态库文件拷贝到发布件的`tools/converter/lib`下。

2. 进入发布件的转换目录

   ```bash
   cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
   ```

3. 创建converter的配置文件（converter.cfg，详细可参考[扩展配置](#扩展配置)），文件内容如下：

   ```text
   [registry]
   plugin_path=libconverter_extend_tutorial.so      # 用户请配置动态库的正确路径
   ```

4. 将转换工具需要的动态链接库加入环境变量`LD_LIBRARY_PATH`

   ```bash
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PACKAGE_ROOT_PATH}/tools/converter/lib
   ```

5. 执行converter

   ```bash
   ./converter_lite --fmk=TFLITE --modelFile=add.tflite --configFile=converter.cfg --outputFile=add_extend
   ```

执行完后，将生成名为`add_extend.ms`的模型文件，文件路径由参数`outputFile`决定。

## 扩展配置

在转换阶段，为了能够加载扩展模块，用户需要配置扩展动态库路径。扩展相关的参数有`plugin_path`、`disable_fusion`、`fusion_blacklists`。参数的详细介绍如下所示：

| 参数              | 属性 | 功能描述             | 参数类型 | 默认值 | 取值范围            |
| ----------------- | ---- | -------------------- | -------- | ------ | ------------------- |
| plugin_path       | 可选 | 第三方库加载路径     | String   | -      | 如有多个请用`;`分隔 |
| disable_fusion    | 可选 | 是否关闭融合优化     | String   | off    | off、on             |
| fusion_blacklists | 可选 | 关闭指定融合算子名称 | String   | -      | 如有多个请用`,`分隔 |

发布件中已为用户生成好默认的配置文件（converter.cfg）。该配置文件内容如下：

```ini
[registry]
plugin_path=libconverter_extend_tutorial.so      # 用户请配置动态库的正确路径
```

如果用户需要关闭指定算子融合优化，关闭指定名单融合配置如下所示：

```ini
[registry]
# 当参数disable_fusion=off时，可通过配置fusion_blacklists关闭指定融合优化，当参数disable_fusion=on时，关闭所有融合优化，参数fusion_blacklists不生效。
disable_fusion=off
fusion_blacklists=ConvActivationFusion,MatMulActivationFusion
```

融合算子名单如下所示：

| 序号 | 融合算子名称                         |
| ---- | ------------------------------------ |
| 1    | AddConcatActivationFusion            |
| 2    | SqueezeFusion                        |
| 3    | TransposeFusion                      |
| 4    | ReshapeReshapeFusion                 |
| 5    | ConvBiasaddFusion                    |
| 6    | ConvBatchNormFusion                  |
| 7    | ConvScaleFusion                      |
| 8    | GroupNormFusion                      |
| 9    | TfNormFusion                         |
| 10   | OnnxLayerNormFusion                  |
| 11   | OnnxLayerNormFusion2                 |
| 12   | BatchMatMulFusion                    |
| 13   | BatchNormToScaleFusion               |
| 14   | SigmoidMulFusion                     |
| 15   | ActivationFusion                     |
| 16   | ConvActivationFusion                 |
| 17   | ConvTupleGetItemFusion               |
| 18   | ConvTupleActivationFusion            |
| 19   | TfliteLstmCellFusion                 |
| 20   | TfLstmCellFusion                     |
| 21   | TfBidirectionGruFusion               |
| 22   | TfGeLUFusion                         |
| 23   | OnnxGeLUFusion                       |
| 24   | TfliteRelPosMultiHeadAttentionFusion |
| 25   | GLUFusion                            |
| 26   | ConstFoldPass                        |
| 27   | AffineFusion                         |
| 28   | AffineActivationFusion               |
| 29   | ConvConvFusion                       |
| 30   | ConvPadFusion                        |
| 31   | MatMulAddFusion                      |
| 32   | MatMulMulFusion                      |
| 33   | TransposeMatMulFusion                |
| 34   | MulAddFusion                         |
| 35   | ScaleActivationFusion                |
| 36   | ScaleScaleFusion                     |
| 37   | FullConnectedFusion                  |
| 38   | FullconnectedAddFusion               |
| 39   | TensorDotFusion                      |
| 40   | MatMulActivationFusion               |

