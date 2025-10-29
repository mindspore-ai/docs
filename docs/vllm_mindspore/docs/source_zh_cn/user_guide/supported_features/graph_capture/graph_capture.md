# 整图捕获和回放

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_zh_cn/user_guide/supported_features/graph_capture/graph_capture.md)

vLLM-MindSpore插件支持通过对模型中的算子进行捕获，并在之后遇到相同输入shape时进行回放来提升算子下发性能，减少host侧瓶颈。在Ascend平台上，主要使用aclgraph来进行图的捕获和回放。

## 基本原理

aclgraph图捕获和回放可以理解为采用算子下发方式进行构图，主要分为以下两个阶段：

- **图捕获**：可以理解为构图阶段，需要传入固定shape的输入，运行时会记录下每个子图下发的算子和每个算子的输入输出地址，作为后续回放的关键数据。

- **图回放**：当之前捕获过的shape的输入传入时，可以通过一次下发执行捕获的图，减少要下发的host开销。

假设一个算子下发要用20us，一个网络包含100个算子，正常算子下发在host侧需要时间是20 * 100 = 2000us，而如果图被捕获后，则只需要一次整图下发，通常会比单算子下发慢一点，比如50us，也可以优化40倍的host时间。

aclgraph回放要求所有显存地址不变，因此采用aclgraph捕获时，整图的输入的显存地址需要预留下来，这就会导致每个shape的图捕获会需要额外消耗一定的显存保存对应shape的输入显存，因此通常无法对所有shape都进行捕获，而是对host性能敏感的小batch进行捕获和回放。在vLLM中，采用了pad机制来保证小batch都能够被捕获，比如最大能够捕获的batch为64，则会捕获[1,2,4,6,16,24,32,40,48,56,64]这些batch的shape，即大于8后，每8个batch捕获一个图，不是直接捕获的会pad到最近可回放的batch数上计算，以保证小batch的性能。

### 切图捕获

aclgraph不支持一些特定的算子和操作，这些算子是无法被aclgraph捕获的，比如流同步操作、分配显存操作、host侧计算操作、acl算子（算子执行过程会分配显存）等。 当前MindSpore的Attention算子（如FlashAttention和PagedAttention），需要在host侧进行tiling计算，导致这些Attention计算无法进入aclgraph图中，因此需要根据Attention算子，将模型切分为多个子图进行捕获。

vLLM原生会调用PyTorch的切图能力，在Python侧就将模型切分为多个子图进行捕获。但是MindSpore当前不支持在Python侧进行切图，而是在C++后端对图进行切分和捕获。根据Attention切分，一个N层的模型，通常会被切分成N+1个子图进行捕获，也就是会有N+1个子图和N个算子的下发开销，经过测试，虽然不如整图捕获和回放，一般host也不会成为性能瓶颈，能够比较有效地降低host开销。

## 开启整图捕获

vLLM原生支持在GPU平台上通过cudagraph进行图捕获和回放，vLLM-MindSpore插件复用vLLM原生的管理能力，只是将cudagraph替换为aclgraph，因此开启图捕获的配置与vLLM保持一致，通过编译级别来控制，启动命令中设置`-O 3`可以开启aclgraph图捕获：

```bash
vllm-mindspore serve -O 3 /path/to/Qwen2.5/model --trust-remote-code
```

当前，vLLM-MindSpore插件只会对纯decoder计算的模型进行整图捕获和回放，其他推理阶段不会使能这个能力。其中，“-O”选项，vLLM-MindSpore插件支持0、1、2、3，其中只有配置为3时会启用图捕获能力，其他配置效果相同，没有区别。

## 使用限制

### 捕获图限制

当前aclgraph每捕获一张图，需要消耗一个stream，而Ascend平台下，stream数量最大限制2000个，同时有些算子（如通信算子）也会申请stream，因此aclgraph能够捕获的最大数为2000左右。为了保证不会和其他特性冲突，考虑到安全因素，建议按照1800计算可捕获的子图数量，因此可以捕获的shape存在一定限制。以deepseek r1模型为例，一般63层，会被切分为64个子图，大概可以捕获30个不同的shape，通过计算，vLLM-MindSpore插件默认捕获最大batch_size为128，即捕获19个不同的shape，用户可以通过手动传递参数进行指定，如下设置捕获1、2、4、8这些batch的shape：

```bash
vllm-mindspore serve /path/to/Qwen2.5/model --compilation_config '{"level": "3", "capture_sizes": [1,2,4,8]}' --trust-remote-code
```

### 不支持MLA

当前vLLM-MindSpore插件支持PagedAttention类型的模型，对于deepseek这类使用MLA算子的模型，由于MindSpore的aclgraph适配存在问题，暂不支持，建议用户不要开启整图捕获特性。
