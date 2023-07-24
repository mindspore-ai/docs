# 分布式推理

`Linux` `Ascend` `推理应用` `初级` `中级` `高级`

[![查看源文件](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_zh_cn/distributed_inference.md)

分布式推理是指推理阶段采用多卡进行推理。如果训练时采用数据并行或者模型参数是合并保存，那么推理方式与上述一致，只需要注意每卡加载同样的checkpoint文件进行推理。

## 分布式推理流程

本篇教程主要介绍在多卡训练过程中，每张卡上保存模型的切片，在推理阶段采用多卡形式，按照推理策略重新加载模型进行推理的过程。针对超大规模神经网络模型的参数个数过多，模型无法完全加载至单卡中进行推理的问题，可利用多卡进行分布式推理。

> 分布式推理样例代码：
>
> <https://gitee.com/mindspore/docs/tree/r1.3/docs/sample_code/distributed_inference>

分布式推理流程如下：

1. 执行训练，生成checkpoint文件和模型参数切分策略文件。

    > - 分布式训练教程和样例代码可参考链接：<https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/distributed_training_ascend.html>.
    > - 在分布式推理场景中，训练阶段的`CheckpointConfig`接口的`integrated_save`参数需设定为`False`，表示每卡仅保存模型切片而不是全量模型。
    > - `set_auto_parallel_context`接口的`parallel_mode`参数需设定为`auto_parallel`或者`semi_auto_parallel`，并行模式为自动并行或者半自动并行。
    > - 此外还需指定`strategy_ckpt_save_file`参数，即生成的策略文件的地址。

2. 设置context，根据推理数据推导出推理策略。

    ```python
    context.set_auto_parallel_context(full_batch=True, parallel_mode='semi_auto_parallel', strategy_ckpt_load_file='./train_strategy.ckpt')
    network = Net()
    model = Model(network)
    predict_data = create_predict_data()
    predict_strategy = model.infer_predict_layout(predict_data)
    ```

    其中，

    - `full_batch`：是否全量导入数据集，为`True`时表明全量导入，每卡的数据相同，该场景中必须设置为`True`。
    - `parallel_mode`：并行模式，该场景中必须设置为自动并行或者半自动并行模式。
    - `strategy_ckpt_load_file`：训练阶段生成的策略文件的文件地址，分布式推理场景中该参数必须设置。
    - `create_predict_data`：用户需自定义的接口，返回推理数据。与训练阶段不同的是，分布式推理场景中返回类型必须为`Tensor`。
    - `infer_predict_layout`：根据推理数据生成推理策略。

3. 导入checkpoint文件，根据推理策略加载相应的模型切片至每张卡中。

    ```python
    ckpt_file_list = create_ckpt_file_list()
    load_distributed_checkpoint(network, ckpt_file_list, predict_strategy)
    ```

    其中，

    - `create_ckpt_file_list`：用户需自定义的接口，返回按rank id排序的CheckPoint文件名列表。
    - `load_distributed_checkpoint`：对模型切片进行合并，再根据推理策略进行切分，加载至网络中。

    > `load_distributed_checkpoint`接口支持predict_strategy为`None`，此时为单卡推理，其过程与分布式推理有所不同，详细用法请参考链接：
    > <https://www.mindspore.cn/docs/api/zh-CN/r1.3/api_python/mindspore.html#mindspore.load_distributed_checkpoint>.

4. 进行推理，得到推理结果。

    ```python
    model.predict(predict_data)
    ```

## 分布式场景导出MindIR文件

针对超大规模神经网络模型参数过多的情况，MindIR格式的模型无法完全加载至单卡中进行推理的问题，可采用多卡进行分布式推理，此时在进行推理任务前就需要导出多个MindIR文件。
针对多卡训练，分布式推理的情况，需要分布式导出MindIR文件，具体方法如下：

首先，需要准备checkpoint文件和训练策略文件。

checkpoint文件在训练过程中产生。checkpoint具体用法可参考: [checkpoint用法](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/save_model.html#checkpoint)。

训练策略文件，需要在训练时通过设置context生成，context配置项如下：
`context.set_auto_parallel_context(strategy_ckpt_save_file='train_strategy.ckpt')`

这样在训练后，就会在设置的目录下产生名为`train_strategy.ckpt`的训练策略文件。

由于导出MindIR文件前，一般需要加载checkpoint文件，而加载分布式训练的checkpoint文件，需要结合训练策略和推理策略，所以还需生成推理策略文件。
产生推理策略的代码如下：
`predict_strategy = model.infer_predict_layout(predict_data)`

然后，使用加载分布式checkpoint的方法，把之前训练好的参数，加载到网络中。
代码如下：
`load_distributed_checkpoint(model, ckpt_file_list, predict_strategy)`

`load_distributed_checkpoint`的具体用法可参考：[分布式推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/multi_platform_inference_ascend_910.html#id1)。

最后，就可以导出在分布式推理场景下的MindIR文件。

核心代码如下：

```python
# 在context中配置训练过程中产生的策略文件
context.set_auto_parallel_context(strategy_ckpt_load_file='train_strategy.ckpt')
# 定义网络结构
network = Net()
model = Model(network)
# 得到推理策略文件
predict_strategy = model.infer_predict_layout(predict_data)
# 创建checkpoint list
ckpt_file_list = create_ckpt_file_list()
# 加载分布式参数
load_distributed_checkpoint(model, ckpt_file_list, predict_strategy)
# 导出分布式MindIR文件
export(net, Tensor(input), file_name='net', file_format='MINDIR')
```

多卡训练、单卡推理的情况，导出MindIR的用法与单机相同，加载checkpoint用法可参考：[分布式推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/multi_platform_inference_ascend_910.html#ascend-910-ai)。

> 分布式场景导出MindIR文件样例代码：
>
> <https://gitee.com/mindspore/docs/tree/r1.3/docs/sample_code/distributed_export>
