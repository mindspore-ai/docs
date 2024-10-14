# 自定义融合Pass

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_zh_cn/model_train/custom_program/fusion_pass.md)

## 概述

算子融合是通过将多个独立的算子组合成一个更大、更复杂的算子，从而减少运行时内存访问、提高计算效率。可以减少中间结果的存储和传输，有效的减少访存的开销；另外，合并多个算子可以减少计算的次数，在NPU等并行计算设备上，可以有效提高计算效率。

当前MindSpore默认会自动进行融合算子优化，将模型中符合条件的多个连续小算子自动合并成一个融合算子。每一个融合算子对应一个融合Pass。MindSpore的IR图通过融合Pass之后实现算子融合替换的效果。MindSpore提供大量算子融合的优化Pass，这些融合算子是根据常见用户需求提取总结的，可以满足大部分用户的需求。

但是，在实际网络调试过程中，用户可能希望手动控制算子融合Pass的开关，例如：

- 调试网络时，用户根据自己的场景手动控制融合Pass的开关，排除一些在该场景效果不佳的融合算子，或者使用更加激进的融合策略，提升网络计算的速度；
- 遇到精度问题时，用户希望通过关闭部分算子融合进行问题定位问题，确定网络精度对应的算子问题。

因此我们对于融合算子相关优化Pass提供了相关接口，帮助用户开关融合Pass进行调试。

## 调试接口

当前算子融合相关优化Pass已经纳入图算优化控制点额范围。
`set_context`的相关设定中，我们提供了选项`graph_kernel_flags`可以控制相关图算优化Pass的开关，包括

- 关闭Pass：`set_context(graph_kernel_flags="--disable_pass=xxx")` 来配置，`xxx`为需要关闭Pass的名称。
- 打开Pass：`set_context(graph_kernel_flags="--enable_pass=xxx")` 来配置，`xxx`为需要打开Pass的名称。

## 获得Pass名称

用户在调试时有两种获取对应Pass名称的方式，也可以通过本文附录列表查询支持列表。

### 通过IR名字

如果用户Dump了相关IR，可以通过IR名字获取相关融合Pass名称。例如相关IR的名称为`hwopt ge_unify_mindir_pm_44_add_layer_norm_fusion_0559.ir`，我们可以在编号之间获得名字`add_layer_norm_fusion`。

### 通过INFO信息

在`[INFO]`信息中，我们提供了所有支持自定义开关的Pass列表。用户可以通过`export GLOG_v=1`来生成`[INFO]`信息。在`[INFO]`信息中，用户通过搜索`graph kernel pass`来获取的该Pass列表。比如下面的信息示例中，`graph kernel pass：`后为 所有可以自定义开关的Pass的名称。

```shell
[INFO] PRE_ACT(631369,ffffb5450af0,python):2024-08-22-15:34:16.978.158 [mindspore/ccsrc/plugin/device/ascend/optimizer/backend_common_unify_mindir.cc:191] GetBackendFusionGroupPassManager] graph kernel passes: FlashAttentionFusionV1,FlashAttentionFusionV2,add_layer_norm_fusion,add_layer_norm_v3_fusion,add_layer_norm_ext_fusion,inference_swiglu_fusion,inference_matmul_split_fusion,shape_reshape,shape_reshape_2,add_rms_norm_quant_fusion,rms_norm_quant_fusion,add_rms_norm_fusion,add_cast_rms_norm_cast_fusion,MatMulAllReduce,split_concat_fusion,matmul_elem_biasadd_fusion,matmul_elem_add_fusion,matmul_elem_relu_fusion,matmul_elem_gelu_fusion,inference_qbmm_add_fusion,inference_qbmm_allreduce_add_fusion.
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

## 附录：相关后端使能Pass列表

**注意**：该列表会随着相关框架更新实时更新，但是仅供参考，具体使能Pass以上面两种方式为准。

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
| shape_reshape_2                     | Ascend   |
| add_rms_norm_quant_fusion           | Ascend   |
| rms_norm_quant_fusion               | Ascend   |
| add_rms_norm_fusion                 | Ascend   |
| add_cast_rms_norm_cast_fusion       | Ascend   |
| MatMulAllReduce                     | Ascend   |
| split_concat_fusion                 | Ascend   |
| matmul_elem_biasadd_fusion          | Ascend   |
| matmul_elem_add_fusion              | Ascend   |
| matmul_elem_relu_fusion             | Ascend   |
| matmul_elem_gelu_fusion             | Ascend   |
| inference_qbmm_add_fusion           | Ascend   |
| inference_qbmm_allreduce_add_fusion | Ascend   |
