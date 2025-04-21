# Sharding Techniques

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/parallel/split_technique.md)

## Overview

For a new model using `Sharding Propagation` to configure the parallelization strategy, the key issue is to configure which operator's slicing strategy will yield better performance. Since the goal of strategy propagation is to minimize the cost of tensor rearranging rather than minimizing the end-to-end iteration time, it is important to configure the appropriate cut strategy for the "key operators". However, there is no explicit rule governing which operators must be configured with a sharding strategy. Nevertheless, based on our experience in training large models, there are some principles that can be used to guide new users in configuring parallel strategies. Here, we list 4 empirical principles.

### Configuring Operators Involving Weights

The sharding strategy for parameter weights is very important, especially for large models, as the memory consumption caused by parameter weights accounts for a large portion of the total memory consumption for model training. Therefore, operators involving weights usually need to explicitly configure the sharding strategy. In the two examples below, the Gather and MatMul operators involving weights are configured with sharding strategy, while the other operators are not. These correspond the data-parallel VocabEmbedding layer and hybrid-parallel FeedForward Layer in [MindFormers](https://gitee.com/mindspore/mindformers/blob/master/mindformers/modules/transformer/transformer.py), respectively.

![sp_case1](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_en/parallel/images/sp_case1.png "Configuring Operators Involving Weights")

### Configuring Dimension-changing/Axis-changing Operators

The operators of deep learning frameworks can be broadly categorized into two types: operators that are semantically simple and dimension-preserving and operators that change the dimension of the input tensor. For dimension-preserving operators, the strategy propagation algorithm can propagate the sharding strategy more easily. However, for dimension-changing operators, explicitly configuring the sharding strategy is the only way to better express the user initial thoughts and avoid the strategy propagation algorithm from deriving the sharding strategy that is not expected by the user. Common dimension-changing and axis-changing operators are: [ReduceMean](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.ReduceMean.html), [ReduceSum](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.ReduceSum.html), [Transpose](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.Transpose.html), [StridedSlice](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.StridedSlice.html), [MatMul](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.MatMul.html), and [BatchMatMul](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.BatchMatMul.html). In the example below, ReduceMean and MatMul are dimension-changing operators that are configured with sharding strategy.

![sp_case2](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_en/parallel/images/sp_case2.png "Configuring Dimension-changing Operators")

### Configuring Boundary Operators that Change in Parallel Strategy

For ResNet-like models, different parts of the model have different preferred parallel: the first half uses data parallel, and the second half uses model parallel for optimal iterative performance. For Llama-like large models, when vocab_size is too large, model parallel slicing may be chosen for memory considerations; when sequence_length is too large, the strategy of sequence parallelism may also be chosen. The above strategies belong to those carefully configured by the user based on the model and hardware information.Sharding Propagation is a plain algorithm to find the least cost of rearrangement, and it does not find the carefully configured strategies automatically, so for the operator strategies carefully tuned by the user, it is necessary to configure them exclusively. In the example below, the first MatMul is configured with a strategy for data parallel, which will propagate the strategy for data parallel forward to the first half of the model, while the second MatMul is configured with a strategy for model parallel, which will propagate the strategy for model parallel backward to the second half of the model.

![sp_case3](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/tutorials/source_en/parallel/images/sp_case3.png "Configuring Boundary Operators that Change in Parallel Method")

### Configuring Fusion Operators

Fusion large operators, such as [FlashAttentionScore](https://www.mindspore.cn/lite/api/en/br_base/generate/classmindspore_ops_FlashAttentionScore.html#exhale-class-classmindspore-ops-flashattentionscore), [rms_norm](https://www.mindspore.cn/docs/en/br_base/api_python/ops/mindspore.ops.rms_norm.html), are also operators that require the user to manually configure the strategy. The input and output logic of the fusion operator is relatively complex, and the propagated strategy without reordering is not necessarily the strategy expected by the user. These operators also require explicit configuration of the operator-level strategy.

Users working with strategy propagation need to have some understanding not only of its propagation algorithm itself, but also of the parallelism of the model to be trained. If there exists a certain operator whose parallelization strategy determined by the strategy propagation algorithm does not meet the user's expectations, that can always be solved by configuring an additional operator parallelization strategy. In practice, for a new model, it does take several attempts to obtain an overall parallel configuration with better performance.

## Configuring Code Samples

Taking the encapsulated class [RowParallelLinear](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/experimental/graph/tensor_parallel/layers.py) in MindFormers as an example:

<table>
<tr>
<td valign='top'>

```diff
# If use semi-automatic, you need to call the shard method to configure the strategy for all operators
class RowParallelLinear(nn.Cell):
    def shard(self, config: TransformerConfig) -> None:
        dp = config.data_parallel
        tp = config.tensor_parallel
        cp = config.context_parallel
        if self.transpose_b:
            weight_strategy = (tp, 1)
        else:
            weight_strategy = (1, tp)
            matmul_in_strategy = ((dp * cp, 1), weight_strategy)
            self.matmul.shard(in_strategy=matmul_in_strategy)
+      if not self.skip_bias_add:
+          dd_in_strategy = ((dp * cp, tp), (tp,))
+          self.add.shard(in_strategy=add_in_strategy)
```

</td>
<td valign='top'>

```diff
# Instead, using strategy propagation, only the strategy of one of the MatMul operators needs to be configured, and there is no need to configure the Add operator:
class RowParallelLinear(nn.Cell):
    def shard(self, config: TransformerConfig) -> None:
        dp = config.data_parallel
        tp = config.tensor_parallel
        cp = config.context_parallel
        if self.transpose_b:
            weight_strategy = (tp, 1)
        else:
            weight_strategy = (1, tp)
            matmul_in_strategy = ((dp * cp, 1), weight_strategy)
            self.matmul.shard(in_strategy=matmul_in_strategy)
```

</td>
</tr>
</table>

The other example is [CoreAttention](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/experimental/graph/transformer/transformer.py). Configure it as above:
<table>
<tr>
<td valign='top'>

```diff
# Semi-automatic configuration is as follows:
class CoreAttention(nn.Cell):
    def shard(self, config: TransformerConfig):
        dp = config.data_parallel
        tp = config.tensor_parallel
        cp = config.context_parallel
+       dropout_strategy = (dp, tp, cp, 1)
+       self.dropout.shard(strategy=dropout_strategy)
        self.bmm_qkv.shard(((dp, tp, cp, 1), (dp, tp, 1, 1)))
+       self.mul.shard(((dp, tp, cp, 1), ()))
        self.bmm_qk.shard(((dp, tp, cp, 1), (dp, tp, 1, 1)))
        self.merge_head_transpose.shard(((dp, tp, cp, 1),))
```

</td>
<td valign='top'>

```diff
# The strategy propagation configuration code is as follows, and only the Matmul and Transpose operators need to be configured:
class CoreAttention(nn.Cell):
    def shard(self, config: TransformerConfig):
        dp = config.data_parallel
        tp = config.tensor_parallel
        cp = config.context_parallel
        self.bmm_qkv.shard(((dp, tp, cp, 1), (dp, tp, 1, 1)))
        self.bmm_qk.shard(((dp, tp, cp, 1), (dp, tp, 1, 1)))
        self.merge_head_transpose.shard(((dp, tp, cp, 1),))
```

</td>
</tr>
</table>

Check the example of [FlashAttention](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/modules/flash_attention.py):
<table>
<tr>
<td valign='top'>

```diff
# Semi-automatic configuration is as follows:
class FlashAttention(Cell):
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        cp_ds = parallel_config.get_ulysses_cp_num()
        fa_strategies = self._generate_flash_attention_strategy(
            dp, mp, cp, cp_ds)
        self.flash_attention.shard(fa_strategies)
+       if self.use_alibi_mask:
+           self.alibi_rescale_mul.shard(((dp, mp, cp, 1), (1,)))
+       return self
```

</td>
<td valign='top'>

```diff
# The strategy propagation configuration code is as follows, which requires the FlashAttentionScore operator to be configured and does not require the Mul operator to be configured:
class FlashAttention(Cell):
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        cp = parallel_config.context_parallel
        cp_ds = parallel_config.get_ulysses_cp_num()
        fa_strategies = self._generate_flash_attention_strategy(
            dp, mp, cp, cp_ds)
        self.flash_attention.shard(fa_strategies)
        return self
```

</td>
</tr>
</table>

If classes that are open source and already paired with a strategy in MindFormers are used directly, the external network does not need to configure the shard strategy for the operator again, e.g., [LlamaForCausalLM](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/llama.py).
<table>
<tr>
<td valign='top'>

```diff
# Semi-automatic configuration is as follows:
class LlamaForCausalLM(LlamaPretrainedModel):
    def shard(self, config: TransformerConfig):
+       dp = config.data_parallel
+       slice_in_strategy = ((dp, 1),)
+       self.slice.shard(in_strategy=slice_in_strategy)
+       not_equal_in_strategy = ((dp, 1), ())
+       self.not_equal.shard(in_strategy=not_equal_in_strategy)
+       mul_in_strategy = ((dp, 1), (dp, 1))
+       self.mul.shard(in_strategy=mul_in_strategy)
+       return self
```

</td>
<td valign='top'>

```diff
# No other operator strategies need to be configured to use strategy propagation
class LlamaForCausalLM(LlamaPretrainedModel):
    def shard(self, config: TransformerConfig):
+       pass
```

</td>
</tr>
</table>

**When the user cannot confirm whether the operator needs to be configured with a strategy, it can be left unconfigured and the algorithm will propagate to find the optimal strategy, but it may not be able to obtain the best parallel results. If the user can confirm what strategy needs to be configured for the operator, it can be configured to help the algorithm obtain the desired results.**