# Function Differences with tf.distribute.Strategy

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/DistributedTrain.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## tf.distribute.Strategy

```python
strategy = tf.distribute.MirroredStrategy()
per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
losses =  strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
```

For more information, see [tf.distribute.Strategy](https://tensorflow.google.cn/versions/r2.6/api_docs/python/tf/distribute/Strategy).

## mindspore.set_auto_parallel_context

```python
import mindspore as ms
ms.set_auto_parallel_context(device_num=8)
ms.set_auto_parallel_context(global_rank=0)
ms.set_auto_parallel_context(gradients_mean=True)
ms.set_auto_parallel_context(gradient_fp32_sync=False)
ms.set_auto_parallel_context(parallel_mode="auto_parallel")
ms.set_auto_parallel_context(search_mode="dynamic_programming")
ms.set_auto_parallel_context(auto_parallel_search_mode="dynamic_programming")
ms.set_auto_parallel_context(parameter_broadcast=False)
ms.set_auto_parallel_context(strategy_ckpt_load_file="./strategy_stage1.ckpt")
ms.set_auto_parallel_context(strategy_ckpt_save_file="./strategy_stage1.ckpt")
ms.set_auto_parallel_context(dataset_strategy=((1, 8), (1, 8)))
ms.set_auto_parallel_context(enable_parallel_optimizer=False)
ms.set_auto_parallel_context(enable_alltoall=False)
ms.set_auto_parallel_context(all_reduce_fusion_config=[8, 160])
ms.set_auto_parallel_context(pipeline_stages=2)
parallel_config = {"gradient_accumulation_shard": True, "parallel_optimizer_threshold": 24}
ms.set_auto_parallel_context(parallel_optimizer_config=parallel_config, enable_parallel_optimizer=True)
config = {"allreduce": {"mode": "size", "config": 32}, "allgather": {"mode": "size", "config": 32}}
ms.set_auto_parallel_context(comm_fusion=config)
```

For more information, see [mindspore.set_auto_parallel_context](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_auto_parallel_context.html#mindspore.set_auto_parallel_context).

## Differences

Tensorflow: Data parallel training is performed through tf.distribute.Strategy, and different strategies specify different data initialization, synchronization.

MindSpore: Specify the value of the configurations in auto-parallel context by passing `**kwargs` to this interface. The rest of the network script is consistent with the single-card network script.
