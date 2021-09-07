# Function Differences with tf.distribute.Strategy

## tf.distribute.Strategy

```python
strategy = tf.distribute.MirroredStrategy()
per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
losses =  strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
```

## mindspore.context.set_auto_parallel_context(ParallelMode=ParallelMode.DataParallel, gradients_mean=True)

```python
from mindspore import context
from mindspore.communication import init
context.set_auto_parallel_context(ParallelMode=ParallelMode.DataParallel, gradients_mean=True)
init()
# Run net, the same with stand alone training
```

## Differences

Tensorflow: Data parallel training is performed through tf.distribute.Strategy, and different strategies specify different data initialization, synchronization.

MindSpore：Specify the data parallel mode through the ParallelMode parameter in 'context.set_auto_parallel_context', and specify the gradient synchronization strategy through the gradients_mean parameter.
The rest of the network script is consistent with the single-card network script.
