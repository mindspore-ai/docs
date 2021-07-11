# 分布式场景导出MindIR文件

针对超大规模神经网络模型参数过多的情况，MindIR格式的模型无法完全加载至单卡中进行推理的问题，可采用多卡进行分布式推理，此时在进行推理任务前就需要导出多个MindIR文件。
针对多卡训练，分布式推理的情况，需要分布式导出MindIR文件，具体方法如下：

首先，需要准备checkpoint文件和训练策略文件。

checkpoint文件在训练过程中产生。checkpoint具体用法可参考: [checkpoint用法](https://www.mindspore.cn/tutorial/training/zh-CN/r1.3/use/save_model.html#checkpoint)。

训练策略文件，需要在训练时通过设置context生成，context配置项如下：
`context.set_auto_parallel_context(strategy_ckpt_save_file='train_strategy.ckpt')`

这样在训练后，就会在设置的目录下产生名为`train_strategy.ckpt`的训练策略文件。

由于导出MindIR文件前，一般需要加载checkpoint文件，而加载分布式训练的checkpoint文件，需要结合训练策略和推理策略，所以还需生成推理策略文件。
产生推理策略的代码如下：
`predict_strategy = model.infer_predict_layout(predict_data)`

然后，使用加载分布式checkpoint的方法，把之前训练好的参数，加载到网络中。
代码如下：
`load_distributed_checkpoint(model, ckpt_file_list, predict_strategy)`

`load_distributed_checkpoint`的具体用法可参考：[分布式推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.3/multi_platform_inference_ascend_910.html#id1)。

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

多卡训练、单卡推理的情况，导出MindIR的用法与单机相同，加载checkpoint用法可参考：[分布式推理](https://mindspore.cn/tutorial/inference/zh-CN/r1.3/multi_platform_inference_ascend_910.html#ascend-910-ai)。

> 分布式场景导出MindIR文件样例代码：
>
> <https://gitee.com/mindspore/docs/tree/r1.3/tutorials/tutorial_code/distributed_export>
