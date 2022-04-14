# Function Differences with tf.distribute.Strategy

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/tensorflow_diff/DistributedTrain.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## tf.distribute.Strategy

```python
strategy = tf.distribute.MirroredStrategy()
per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
losses =  strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
```

For more information, see [tf.distribute.Strategy](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/distribute/Strategy).

## mindspore.context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)

```python
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init
context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
# Run net, the same with stand alone training
```

For more information, see [context.set_auto_parallel_context(ParallelMode=ParallelMode.DataParallel)](https://mindspore.cn/docs/en/r1.7/api_python/mindspore.context.html#mindspore.context.set_auto_parallel_context).

## Differences

Tensorflow: Data parallel training is performed through tf.distribute.Strategy, and different strategies specify different data initialization, synchronization.

MindSpore: Specify the data parallel mode through the ParallelMode parameter in 'context.set_auto_parallel_context', and specify the gradient synchronization strategy through the gradients_mean parameter.
The rest of the network script is consistent with the single-card network script.
