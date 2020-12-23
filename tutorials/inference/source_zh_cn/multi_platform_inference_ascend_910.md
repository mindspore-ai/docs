# Ascend 910 AI处理器上推理

`Linux` `Ascend` `推理应用` `初级` `中级` `高级`

<!-- TOC -->

- [Ascend 910 AI处理器上推理](#ascend-910-ai处理器上推理)
    - [使用checkpoint格式文件单卡推理](#使用checkpoint格式文件单卡推理)
    - [分布式推理](#分布式推理)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/inference/source_zh_cn/multi_platform_inference_ascend_910.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 使用checkpoint格式文件单卡推理

1. 使用`model.eval`接口来进行模型验证。

   1.1 模型已保存在本地  

   首先构建模型，然后使用`mindspore.train.serialization`模块的`load_checkpoint`和`load_param_into_net`从本地加载模型与参数，传入验证数据集后即可进行模型推理，验证数据集的处理方式与训练数据集相同。

    ```python
    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    dataset = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size,
                             1)
    acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    ```

    其中，  
    `model.eval`为模型验证接口，对应接口说明：<https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.html#mindspore.Model.eval>。
    > 推理样例代码：<https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/lenet/eval.py>。

   1.2 使用MindSpore Hub从华为云加载模型

   首先构建模型，然后使用`mindspore_hub.load`从云端加载模型参数，传入验证数据集后即可进行推理，验证数据集的处理方式与训练数据集相同。

    ```python
    model_uid = "mindspore/ascend/0.7/googlenet_v1_cifar10"  # using GoogleNet as an example.
    network = mindspore_hub.load(model_uid, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    dataset = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size,
                             1)
    acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    ```

    其中，  
    `mindspore_hub.load`为加载模型参数接口，对应接口说明：<https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore_hub/mindspore_hub.html#module-mindspore_hub>。

2. 使用`model.predict`接口来进行推理操作。

   ```python
   model.predict(input_data)
   ```

   其中，  
   `model.predict`为推理接口，对应接口说明：<https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.html#mindspore.Model.predict>。

## 分布式推理

分布式推理是指推理阶段采用多卡进行推理。如果训练时采用数据并行或者模型参数是合并保存，那么推理方式与上述一致，只需要注意每卡加载同样的checkpoint文件进行推理。

本篇教程主要介绍在多卡训练过程中，每张卡上保存模型的切片，在推理阶段采用多卡形式，按照推理策略重新加载模型进行推理的过程。针对超大规模神经网络模型的参数个数过多，模型无法完全加载至单卡中进行推理的问题，可利用多卡进行分布式推理。

> 分布式推理样例代码：
>
> <https://gitee.com/mindspore/docs/tree/master/tutorials/tutorial_code/distributed_inference>

分布式推理流程如下：

1. 执行训练，生成checkpoint文件和模型参数切分策略文件。

    > - 分布式训练教程和样例代码可参考链接：<https://www.mindspore.cn/tutorial/training/zh-CN/master/advanced_use/distributed_training_ascend.html>.
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
    > <https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/mindspore.html#mindspore.load_distributed_checkpoint>.

4. 进行推理，得到推理结果。

    ```python
    model.predict(predict_data)
    ```
