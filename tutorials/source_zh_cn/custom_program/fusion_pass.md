# 自定义融合

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_zh_cn/custom_program/fusion_pass.md)

## 概述

算子融合是通过将多个独立的算子组合成一个更大、更复杂的算子，从而减少运行时内存访问、提高计算效率。

具体表现为：

- 通过减少中间结果的存储和传输，有效地减少访存的开销。
- 通过合并多个算子，减少计算的次数，在NPU等并行计算设备上，有效地提高计算效率。

当前MindSpore有两种融合方式：

1. 设置jit_level=O1，使能图算融合。该功能会自动将复杂大算子展开为基础小算子，并根据指定的融合规则自动融合为融合算子，然后经过AKG自动生成融合算子的底层实现。
2. 通过融合pass融合，默认使能。该功能将模型中符合特定条件的多个连续小算子自动合并成一个融合算子。每一个融合算子对应一个融合Pass。MindSpore的IR图通过融合Pass之后实现算子融合替换的效果。MindSpore提供大量算子融合的优化Pass，这些融合算子是根据常见用户需求提取总结的，可以满足大部分用户的需求。

在实际网络调试过程中，用户可能希望手动控制算子融合的开关，例如：

- 调试网络时，用户根据自己的场景手动控制融合的开关，排除一些在该场景效果不佳的融合算子，或者使用更加激进的融合策略，提升网络计算的速度；
- 遇到精度问题时，用户希望通过关闭部分算子融合进行问题定位问题，确定网络精度对应的算子问题。

因此我们对于融合算子相关优化提供了对应接口，帮助用户自定义融合策略进行调试。

## 调试接口

当前算子融合相关优化Pass已经纳入图算优化控制点额范围。
我们提供了环境变量`MS_DEV_GRAPH_KERNEL_FLAGS`可以控制相关图算优化Pass的开关，包括：

### 指定优化等级

- **opt_level**：设置优化级别。默认值： `2` 。当opt_level的值大于0时，启动图算融合。可选值包括：
    - 0：关闭图算融合。
    - 1：启动算子的基本融合。
    - 2：包括级别1的所有优化，并打开更多的优化，如CSE优化算法、算术简化等。
    - 3：包括级别2的所有优化，并打开更多的优化，如SitchingFusion、ParallelFusion等。在某些场景下，该级别的优化激进且不稳定。使用此级别时要小心。

### 指定自动融合策略

- **enable_expand_ops**：将不在默认列表的算子强行展开，需有相应算子的expander实现。例如，通过设置 `--enable_expand_ops=Square` 可以让Square算子强行展开。默认展开的算子名单见附录1。
- **disable_expand_ops**：禁止对应算子展开。
- **enable_expand_ops_only**：仅允许对应算子展开。当设置该选项时，忽略以上两个选项。
- **enable_cluster_ops**：在默认融合算子名单的基础上，把对应算子加入参与融合的算子集合。例如，通过设置 `--enable_cluster_ops=MatMul` 可以让MatMul算子参与融合。当前默认融合的算子名单见附录2。
- **disable_cluster_ops**：禁止对应算子加入参与融合的算子集合。
- **enable_cluster_ops_only**：仅允许对应算子加入参与融合的算子集合。当设置该选项时，忽略以上两个选项。
- **disable_fusion_pattern**：禁止对应融合pattern参与融合。融合pattern名单见附录4。
- **enable_fusion_pattern_only**：仅允许对应融合pattern参与融合。当设置该选项时，忽略以上选项。

### 指定自动/手动融合pass是否使能

- **enable_pass**：默认关闭的pass可以通过该选项强制使能。
- **disable_pass**：默认使能的pass可以通过该选项强制关闭。

### 打开调试信息

- **dump_as_text**：将关键过程的详细信息生成文本文件保存到`graph_kernel_dump`目录里。默认值： `False` 。
- **enable_debug_mode**：在图算kernelmod launch前后插同步，并在launch失败时打印调试信息，仅支持GPU后端。默认值： `False` 。

> - 格式为`--key=value`，多个配置项以空格分隔，多个value以逗号分隔，例如`export MS_DEV_GRAPH_KERNEL_FLAGS="--enable_expand_ops=Square --enable_cluster_ops=MatMul,Add"`
> - 支持通过`--path=example.json`读取文件配置方式，json的键值对分别为以上key和value对应的字符串，例如`export MS_DEV_GRAPH_KERNEL_FLAGS="--path=example.json"`，example.json的内容为: { "enable_expand_ops" : "Square" }。

## 获得Pass名称

用户在调试时有两种获取对应Pass名称的方式，也可以通过本文附录3列表查询支持列表。

### 通过IR名字

如果用户Dump了相关IR，可以通过IR名字获取相关融合Pass名称。例如相关IR的名称为`hwopt ge_unify_mindir_pm_44_add_layer_norm_fusion_0559.ir`，我们可以在编号之间获得名字`add_layer_norm_fusion`。

### 通过INFO信息

在`[INFO]`信息中，我们提供了所有支持自定义开关的Pass列表。用户可以通过`export GLOG_v=1`来生成`[INFO]`信息。在`[INFO]`信息中，用户通过搜索`graph kernel pass`来获取该Pass列表。比如下面的信息示例中，`graph kernel pass：`后为所有可以自定义开关的Pass名称。

```shell
[INFO] PRE_ACT(631369,ffffb5450af0,python):2024-08-22-15:34:16.978.158 [mindspore/ccsrc/plugin/device/ascend/optimizer/backend_common_unify_mindir.cc:191] GetBackendFusionGroupPassManager] graph kernel passes: FlashAttentionFusionV1,FlashAttentionFusionV2,add_layer_norm_fusion,add_layer_norm_v3_fusion,add_layer_norm_ext_fusion,inference_swiglu_fusion,inference_matmul_split_fusion,shape_reshape,add_rms_norm_quant_fusion,rms_norm_quant_fusion,add_rms_norm_fusion,add_cast_rms_norm_cast_fusion,MatMulAllReduce,split_concat_fusion,matmul_elemwise_fusion,inference_qbmm_add_fusion,inference_qbmm_allreduce_add_fusion.
```

对于单个Pass，我们也可以通过日志信息确认它是否使能，比如：

- 已经使能Pass：下面信息表示`rms_norm_quant_fusion`已经使能，可以通过`disable_pass`关闭；

    ```shell
    [INFO] GRAPH_KERNEL(631369,ffffb5450af0,python):2024-08-22-15:34:17.640.739 [mindspore/ccsrc/backend/common/graph_kernel/core/graph_kernel_pass_manager.cc:84] RunPass] Run graph kernel pass fusion_group_10_rms_norm_quant_fusion in 74.64 us
    ```

- 被关闭Pass: 下面信息表示`transpose_matmul_fusion`被关闭，可以通过`enable_pass`打开；

    ```shell
    [INFO] GRAPH_KERNEL(631369,ffffb5450af0,python):2024-08-22-15:34:17.640.771 [mindspore/ccsrc/backend/common/graph_kernel/core/graph_kernel_pass_manager.cc:73] Run] graph kernel pass fusion_group_11_add_rms_norm_fusion is disabled.
    ```

## 附录1：相关后端默认expander列表

**注意**：该列表会随着相关框架更新定时更新。

| 算子名称 | Ascend | CPU | GPU |
|:-------:|:------:|:---:|:---:|
|Adam|Y|Y|N|
|AdamApplyOneWithDecayAssign|Y|N|N|
|Addcmul|Y|N|N|
|AddN|Y|Y|Y|
|BiasAdd|Y|Y|Y|
|BiasAddGrad|Y|Y|Y|
|FillV2|Y|N|N|
|GeLU|Y|Y|Y|
|Gelu|Y|Y|Y|
|FastGelu|Y|N|N|
|FastGeluGrad|Y|N|N|
|FastGeLU|Y|N|N|
|FastGeLUGrad|Y|N|N|
|SiLU|Y|N|N|
|SiLUGrad|Y|N|N|
|GeLUGrad|Y|Y|Y|
|RsqrtGrad|Y|N|N|
|SqrtGrad|Y|Y|Y|
|Square|Y|Y|Y|
|Tile|Y|Y|Y|
|ClipByNormNoDivSum|Y|N|N|
|FusedMulAdd|Y|N|N|
|Sigmoid|Y|N|Y|
|SigmoidGrad|Y|N|Y|
|SigmoidCrossEntropyWithLogits|Y|N|Y|
|SigmoidCrossEntropyWithLogitsGrad|Y|N|Y|
|SquaredDifference|Y|N|Y|
|TanhGrad|Y|Y|N|
|OnesLike|Y|Y|Y|
|ZerosLike|Y|N|N|
|ReduceMean|Y|N|Y|
|LogSoftmaxGrad|N|N|Y|
|ReLU|Y|Y|Y|
|ReluGrad|Y|N|Y|
|AssignAdd|Y|Y|Y|
|LambApplyOptimizerAssign|Y|N|N|
|LambApplyWeightAssign|Y|N|N|
|AdamApplyOneWithDecay|Y|N|N|
|ExpandDims|N|Y|Y|
|Squeeze|N|N|Y|
|SoftmaxGradExt|N|N|N|
|ApplyMomentum|N|N|N|
|LeakyReLUExt|Y|N|N|
|EluExt|Y|N|N|
|SoftplusExt|Y|N|N|
|SoftplusGradExt|Y|N|N|
|RepeatInterleaveInt|Y|N|N|
|HShrink|Y|N|N|
|HSigmoid|Y|N|N|
|HSwish|Y|N|N|
|BinaryCrossEntropy|Y|N|N|
|Erf|Y|N|N|
|Tanh|Y|N|N|
|Cosh|Y|N|N|
|Sinh|Y|N|N|
|ClampScalar|Y|N|N|
|DivMod|Y|N|N|
|BCEWithLogitsLoss|Y|N|N|
|AcoshExt|Y|N|N|
|AsinhExt|Y|N|N|
|MeanExt|Y|N|N|
|Erfc|N|N|Y|
|AdamWeightDecay|N|N|Y|
|BatchMatMul|N|N|Y|
|Dropout|N|N|Y|
|DropoutGrad|N|N|Y|
|MaximumGrad|N|Y|Y|
|MinimumGrad|N|Y|Y|
|LayerNorm|N|N|Y|
|LayerNormGrad|N|N|Y|
|LogSoftmax|N|N|Y|
|MatMul|N|N|Y|
|ArgMaxWithValue|N|N|Y|
|ArgMinWithValue|N|N|Y|
|Slice|N|N|Y|
|Softmax|N|N|Y|
|SoftmaxCrossEntropyWithLogits|N|N|Y|
|EqualCount|N|N|Y|
|SquareSumAll|N|N|Y|
|IdentityMath|N|N|Y|
|StandardNormal|N|N|Y|
|Softplus|N|Y|N|
|SoftplusGrad|N|Y|N|

## 附录2：相关后端默认cluster列表

**注意**：该列表会随着相关框架更新定时更新。

| 算子名称 | Ascend | CPU | GPU |
|:-------:|:------:|:---:|:---:|
|Abs|Y|Y|Y|
|Add|Y|Y|Y|
|BroadcastTo|Y|N|N|
|Cast|Y|Y|Y|
|Exp|Y|Y|Y|
|Log|Y|Y|Y|
|Maximum|Y|Y|Y|
|Minimum|Y|Y|Y|
|Mul|Y|Y|Y|
|Neg|Y|Y|Y|
|Pow|Y|Y|Y|
|Div|Y|N|Y|
|RealDiv|Y|Y|Y|
|Reciprocal|Y|Y|Y|
|Rsqrt|Y|Y|Y|
|Sqrt|Y|Y|Y|
|Sub|Y|Y|Y|
|Equal|Y|Y|Y|
|NotEqual|Y|N|Y|
|Greater|Y|N|Y|
|GreaterEqual|Y|N|Y|
|Less|Y|Y|Y|
|LessEqual|Y|Y|Y|
|LogicalAnd|Y|N|Y|
|LogicalOr|Y|N|Y|
|LogicalNot|Y|Y|Y|
|Select|Y|Y|Y|
|Assign|Y|N|Y|
|ReduceSum|Y|Y|Y|
|IsFinite|Y|N|Y|
|Reshape|N|Y|Y|
|Transpose|Y|Y|Y|
|Floor|Y|N|Y|
|Ceil|Y|N|N|
|Trunc|Y|N|Y|
|Round|N|Y|Y|
|Tanh|N|Y|Y|
|ACos|N|N|Y|
|Acosh|N|N|Y|
|ArgMax|N|N|N|
|Argmin|N|N|N|
|Asin|N|N|Y|
|Asinh|N|N|Y|
|Atan|N|N|Y|
|Atan2|N|N|Y|
|Cos|N|N|Y|
|Erf|N|N|Y|
|Expm1|N|N|Y|
|FloorDiv|N|N|Y|
|FloorMod|N|N|Y|
|IsInf|N|N|Y|
|IsNan|N|N|Y|
|Mod|N|Y|Y|
|ReduceMax|N|Y|Y|
|ReduceMin|N|N|Y|
|Sign|N|N|Y|
|Sin|N|N|Y|
|StridedSlice|N|N|Y|
|CumSum|N|N|Y|
|OneHot|N|N|Y|

## 附录3：相关后端使能Pass列表

**注意**：该列表会随着相关框架更新实时更新，但是仅供参考。具体使能Pass以上面两种方式为准。

| Pass 名称                           | 使能后端 |
|-------------------------------------|----------|
| FlashAttentionFusionV1              | Ascend   |
| FlashAttentionFusionV2              | Ascend   |
| add_layer_norm_fusion               | Ascend   |
| add_layer_norm_v3_fusion            | Ascend   |
| add_layer_norm_ext_fusion           | Ascend   |
| inference_swiglu_fusion             | Ascend   |
| inference_matmul_split_fusion       | Ascend   |
| shape_reshape                       | Ascend   |
| add_rms_norm_quant_fusion           | Ascend   |
| rms_norm_quant_fusion               | Ascend   |
| add_rms_norm_fusion                 | Ascend   |
| add_cast_rms_norm_cast_fusion       | Ascend   |
| MatMulAllReduce                     | Ascend   |
| split_concat_fusion                 | Ascend   |
| matmul_elemwise_fusion              | Ascend   |
| inference_qbmm_add_fusion           | Ascend   |
| inference_qbmm_allreduce_add_fusion | Ascend   |

## 附录4：相关后端使能融合pattern列表

| pattern 名称                        | 使能后端         |
|-------------------------------------|-----------------|
| elemwise_broadcast_fwd_depth        | Ascend GPU CPU  |
| elemwise_broadcast_fwd_width        | Ascend GPU CPU  |
| elemwise_broadcast_bwd_depth        | Ascend GPU CPU  |
| elemwise_broadcast_bwd_width        | Ascend GPU CPU  |
| reduce_fwd_depth                    | Ascend GPU CPU  |
| reduce_fwd_width                    | Ascend GPU CPU  |
| reshape                             | Ascend GPU CPU  |
| slice                               | Ascend          |
| elemany_addn                        | Ascend          |
| matmul_depth                        | Ascend          |
