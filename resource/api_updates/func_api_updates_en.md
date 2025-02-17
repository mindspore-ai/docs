# mindspore.ops API Interface Change

Compared with the previous version 2.4.10, the added, deleted and supported platforms change information of `mindspore.ops` operators in MindSpore, is shown in the following table.

|API|Status|Description|Support Platform|Class
|:----|:----|:----|:----|:----
[mindspore.ops.swiglu](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.swiglu.html#mindspore.ops.swiglu)|New|Computes SwiGLU (Swish-Gated Linear Unit activation function) of input tensor.|r2.5.0: Ascend|Activation Functions
[mindspore.ops.reverse](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.reverse.html#mindspore.ops.reverse)|Changed|r2.4.10: mindspore.ops.reverse() will be deprecated in the future. => r2.5.0: This interface will be deprecated in the future, and use mindspore.ops.flip() instead.|r2.4.10: Ascend/GPU/CPU => r2.5.0: Deprecated |Array Operation
[mindspore.ops.polar](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.polar.html#mindspore.ops.polar)|Changed|Converts polar coordinates to Cartesian coordinates.|r2.4.10: GPU/CPU => r2.5.0: Ascend/GPU/CPU|Element-wise Operations
[mindspore.ops.rotary_position_embedding](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.rotary_position_embedding.html#mindspore.ops.rotary_position_embedding)|New|Implements the Rotary Position Embedding algorithm.|r2.5.0: Ascend|Image Functions
[mindspore.ops.rotated_iou](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.rotated_iou.html#mindspore.ops.rotated_iou)|New|Calculate the overlap area between rotated rectangles.|r2.5.0: Ascend|Image Functions
[mindspore.ops.all_gather_matmul](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.all_gather_matmul.html#mindspore.ops.all_gather_matmul)|New|In the TP segmentation scenario, allgather and matmul are fused, and communication and computational pipelines are parallelized within the fusion operator.|r2.5.0: Ascend|MC2 Functions
[mindspore.ops.matmul_reduce_scatter](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.matmul_reduce_scatter.html#mindspore.ops.matmul_reduce_scatter)|New|In the TP segmentation scenario, matmul and reducescatter are fused, and communication and computational pipelines are parallelized within the fusion operator.|r2.5.0: Ascend|MC2 Functions
[mindspore.ops.flash_attention_score](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.flash_attention_score.html#mindspore.ops.flash_attention_score)|New|Implement self-attention calculations in training scenarios.|r2.5.0: Ascend|Neural Network
[mindspore.ops.incre_flash_attention](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.incre_flash_attention.html#mindspore.ops.incre_flash_attention)|New|The interface for incremental inference.|r2.5.0: Ascend|Neural Network
[mindspore.ops.prompt_flash_attention](https://mindspore.cn/docs/en/r2.5.0/api_python/ops/mindspore.ops.prompt_flash_attention.html#mindspore.ops.prompt_flash_attention)|New|The interface for fully inference.|r2.5.0: Ascend|Neural Network
