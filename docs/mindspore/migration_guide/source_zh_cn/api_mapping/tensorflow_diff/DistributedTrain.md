# 比较与tf.distribute.Strategy的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/tensorflow_diff/DistributedTrain.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## tf.distribute.Strategy

```python
strategy = tf.distribute.MirroredStrategy()
per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
losses =  strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                          axis=None)

```

更多内容详见[tf.distribute.Strategy](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/distribute/Strategy)。

## mindspore.context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)

```python
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication import init
context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
init()
# Run net, the same with stand alone training
```

更多内容详见[context.set_auto_parallel_context(ParallelMode=ParallelMode.DataParallel)](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.context.html#mindspore.context.set_auto_parallel_context)。

## 使用方式

TensorFlow：通过tf.distribute.Strategy执行数据并行训练，不同策略指定不同的数据初始化、同步等策略。

MindSpore：通过context.set_auto_parallel_context中的ParallelMode参数的指定数据并行模式，通过gradients_mean参数指定梯度同步策略，
其余网络脚本部分与单卡网络脚本保持一致。
