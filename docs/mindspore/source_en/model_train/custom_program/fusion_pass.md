# Custom Fusion Pass

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.1/docs/mindspore/source_en/model_train/custom_program/fusion_pass.md)

## Overview

Operator fusion combines multiple independent operators into a larger, more complex operator to reduce runtime memory accesses and improve computational efficiency. This approach minimizes the storage and transmission of intermediate results, effectively reducing memory access overhead. Additionally, fusing multiple operators reduces the number of computations, which can significantly enhance computational efficiency on parallel computing devices like NPUs.

MindSpore automatically fuses operators by default, combining multiple consecutive small operators following certain patterns into a fused operator. Each fused operator corresponds to a fusion pass. MindSpore runs fusion passes on MindIR to fuse and replace operators. MindSpore provides numerous optimization passes for operator fusion to satisfy the requirements of most users.

However, during actual network debugging, users may wish to manually control the switches for operator fusion passes. For example:

- When debugging a network, users can manually control the switches for fusion passes based on their scenarios, excluding fusion operators that perform poorly in that scenario or adopting more aggressive fusion strategies to improve network computation speed.
- When encountering accuracy issues, users can disable some operator fusions to locate the problem, determining the operator responsible for the network accuracy issue.

Therefore, MindSpore provides interfaces related to fusion operator optimization passes, allowing users to toggle fusion passes for debugging purposes.

## Debugging Interfaces

Currently, operator fusion-related optimization passes are included in the graph kernel optimization module. The `set_context` function provides the `graph_kernel_flags` option to control the switches for related graph optimization passes, including:

- Disable Pass: Use `set_context(graph_kernel_flags="--disable_pass=xxx")`, where `xxx` is the name of the pass to disable.
- Enable Pass: Use `set_context(graph_kernel_flags="--enable_pass=xxx")`, where `xxx` is the name of the pass to enable.

## Obtaining Pass Names

Users have two ways to obtain the corresponding pass names during debugging, or they can refer to the appendix list for supported passes.

### Through IR Names

If users have dumped the relevant IR, they can obtain the related fusion pass name from the IR name. For example, if the IR name is `hwopt ge_unify_mindir_pm_44_add_layer_norm_fusion_0559.ir`, the pass name `add_layer_norm_fusion` can be extracted from the ir name.

### Through INFO Messages

In `[INFO]` messages, we provide a list of all passes that support custom switches. Users can generate `[INFO]` messages by setting `export GLOG_v=1`. In the `[INFO]` messages, users can search for `graph kernel pass` to obtain the list of these passes. For example, in the following message, the names of all passes that can be customized are listed after `graph kernel pass:`.

```shell
[INFO] PRE_ACT(631369,ffffb5450af0,python):2024-08-22-15:34:16.978.158 [mindspore/ccsrc/plugin/device/ascend/optimizer/backend_common_unify_mindir.cc:191] GetBackendFusionGroupPassManager] graph kernel passes: FlashAttentionFusionV1,FlashAttentionFusionV2,add_layer_norm_fusion,add_layer_norm_v3_fusion,add_layer_norm_ext_fusion,inference_swiglu_fusion,inference_matmul_split_fusion,shape_reshape,shape_reshape_2,add_rms_norm_quant_fusion,rms_norm_quant_fusion,add_rms_norm_fusion,add_cast_rms_norm_cast_fusion,MatMulAllReduce,split_concat_fusion,matmul_elem_biasadd_fusion,matmul_elem_add_fusion,matmul_elem_relu_fusion,matmul_elem_gelu_fusion,inference_qbmm_add_fusion,inference_qbmm_allreduce_add_fusion.
```

For individual passes, users can also confirm whether they are enabled through log messages. For example:

- Enabled Pass: The following message indicates that `rms_norm_quant_fusion` is enabled and can be disabled using `disable_pass`.

    ```shell
    [INFO] GRAPH_KERNEL(631369,ffffb5450af0,python):2024-08-22-15:34:17.640.739 [mindspore/ccsrc/backend/common/graph_kernel/core/graph_kernel_pass_manager.cc:84] RunPass] Run graph kernel pass fusion_group_10_rms_norm_quant_fusion in 74.64 us
    ```

- Disabled Pass: The following message indicates that `transpose_matmul_fusion` is disabled and can be enabled using `enable_pass`.

    ```shell
    [INFO] GRAPH_KERNEL(631369,ffffb5450af0,python):2024-08-22-15:34:17.640.771 [mindspore/ccsrc/backend/common/graph_kernel/core/graph_kernel_pass_manager.cc:73] Run] graph kernel pass fusion_group_11_add_rms_norm_fusion is disabled.
    ```

## Appendix: List of Enabled Passes for Relevant Backends

**Note**: This list is provided for reference only and subject to change. The actual enabled passes should be determined using the methods described above.

| Pass Names                          | Backend  |
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
