# Function Differences with tf.distribute.Strategy

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/tensorflow_diff/DistributedTrain.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## tf.distribute.Strategy

```python
strategy = tf.distribute.MirroredStrategy()
per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
losses =  strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
```

For more information, see [tf.distribute.Strategy](http://www.tensorflow.org/versions/r1.15/api_docs/python/tf/distribute/Strategy).

## mindspore.context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)

```python
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init
context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
# Run net, the same with stand alone training
```

For more information, see [context.set_auto_parallel_context(ParallelMode=ParallelMode.DataParallel)](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/mindspore.context.html#mindspore.context.set_auto_parallel_context).

## Differences

Tensorflow: Data parallel training is performed through tf.distribute.Strategy, and different strategies specify different data initialization, synchronization.

MindSpore：Specify the data parallel mode through the ParallelMode parameter in 'context.set_auto_parallel_context', and specify the gradient synchronization strategy through the gradients_mean parameter.
The rest of the network script is consistent with the single-card network script.
