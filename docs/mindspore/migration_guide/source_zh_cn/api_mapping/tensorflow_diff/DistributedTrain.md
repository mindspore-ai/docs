# 比较与tf.distribute.Strategy的功能差异

## tf.distribute.Strategy

```python
strategy = tf.distribute.MirroredStrategy()
per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
losses =  strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                          axis=None)

```

## mindspore.context.set_auto_parallel_context(ParallelMode=ParallelMode.DataParallel, gradients_mean=True)

```python
from mindspore import context
from mindspore.communication import init
context.set_auto_parallel_context(ParallelMode=ParallelMode.DataParallel, gradients_mean=True)
init()
# Run net, the same with stand alone training
```

## 使用方式

Tensorflow: 通过tf.distribute.Strategy执行数据并行训练，不同策略指定不同的数据初始化、同步等策略。

MindSpore：通过context.set_auto_parallel_context中的ParallelMode参数的指定数据并行模式，通过gradients_mean参数指定梯度同步策略，
其余网络脚本部分与单卡网络脚本保持一致。
