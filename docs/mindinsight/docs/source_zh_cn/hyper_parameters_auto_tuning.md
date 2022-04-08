# 使用mindoptimizer进行超参调优

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/docs/source_zh_cn/hyper_parameters_auto_tuning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

## 概述

机器学习领域一般有两类参数，一类是模型内部参数，依靠训练数据来对模型参数进行调参，还有一类则是模型外部的设置参数，需要人工配置，这类参数被称为“超参数”。不同的超参数会对模型效果有不小的影响，因此超参在训练任务中的重要性较高。传统的方式都需要人工去调试和配置，这种方式消耗时间和精力。MindInsight调参功能可以用于搜索超参，基于用户给的调参配置信息，可以自动搜索参数并且执行模型训练。

MindInsight提供的`mindoptimizer`调参命令可以根据用户配置，从训练日志中提取以往训练记录，再对以往训练记录进行分析，推荐超参，最后自动执行训练脚本。用户在使用时需要按照yaml格式来配置超参的范围等信息，再参考本教程替换训练脚本中的超参，旨在将自动推荐的超参同步到训练脚本里面。当前仅支持高斯过程调参方法，其他方法敬请期待。

## 安装

此工具为MindInsight的子模块，安装MindInsight后，即可使用MindInsight调参命令，安装MindInsight请参考该[安装文档](https://gitee.com/mindspore/mindinsight/blob/master/README_CN.md#)。

## 用法

MindInsight提供调参命令，命令行（Command-line interface, CLI）的使用方式如下：

```text
usage: mindoptimizer [-h] [--version] [--config CONFIG]
                     [--iter ITER]

optional arguments:
  -h, --help             Shows the help message and exits.
  --version              Shows the program version and exits.
  --config CONFIG        Specifies the configuration file for parameter tuning.
                         The file format is yaml.
  --iter ITER            Specifies the times of automatic training.
                         Automatically recommended parameters are used every time
                         before the training is performed.
                         The default value of ITER is 1.
```

## 配置文件规则说明

调参配置文件的格式是yaml，需配置运行命令、训练日志根目录、调参方法、优化目标和超参数信息。其中超参数需要配置取值范围，类型和来源等。MindInsight会根据配置的超参数和优化目标从训练日志中取训练记录，如学习率和正确率，可以供推荐算法分析它们之间的关系，更好地推荐超参数。

1. 配置运行命令

    通过`command`来配置运行命令，如`command: python train.py`。在调参程序推荐出超参数后，运行命令会被直接执行。

2. 配置训练日志根目录

    `summary_base_dir`是训练日志根目录，它用于训练记录的提取，这样可以更好地推荐超参。同时，建议用户在训练脚本中加`SummaryColletor`来收集训练信息，可查看[Summmary收集教程](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/summary_record.html)。调参命令会根据配置的`summary_base_dir`来生成子目录路径，可配置在`SummaryColletor`记录该次训练记录。自动执行训练后，会在训练日志根目录的子目录记录当次训练信息，产生的训练信息可以作为训练记录来推荐下一次需要的超参。配置`summary_base_dir`如`summary_base_dir: /home/summaries`。

3. 配置调参方法

    通过`name`配置调参方法的名字，通过`args`字段来配置这个调参方法的参数。

    当前采用的算法是高斯过程回归器（Gaussian process regressor, GP），这个算法可配置采集方法(Acquisition Function)，可选，范围是[`ucb`, `pi`,`ei`]，默认值为`ucb`。

    - Upper confidence bound (UCB)
    - Probability of improvement (PI)
    - Expected improvement (EI)

    示例：

    ```yaml
    tuner:
        name: gp
        args:
            method: ucb
    ```

4. 配置调参目标

    用户可以选择loss或者自定义的评估指标作为调参的目标。

    配置说明：

    - group：可选，取值包括`system_defined`和`metric`，默认`system_defined`。使用`group`来配置优化目标所在的组，如loss是系统自定义收集字段，则是`system_defined`组；而其他在`Model()`中使用的评估指标，如`model = Model(net, loss_fn=loss, optimizer=None, metrics={'Accuracy'})`，`Accuracy`属于评估指标（metrics），因此组别是`metric`。
    - goal：可选，取值包括`minimize`、`maximize`，默认`minimize`。使用`goal`来表示该目标的优化方向，如正确率越高越好，即`goal`需要配置为`maximize`。

    配置loss：

    ```yaml
    target:
        name:loss
    ```

    配置评估指标中的Accuracy:

    ```yaml
    target:
        group: metric
        name: Accuracy
        goal: maximize
    ```

5. 配置超参信息

    超参的配置字段：`bounds`、`choice`、`type`和`source`。这里配置的超参字段，会用于训练记录的提取和超参推荐。其中，`bounds`、`choice`和`type`会影响超参推荐，`bounds`配置了参数的上下界，`choice`表示推荐值从中选取，`type`则是配置了该参数的类型。

    目前系统自定义收集的可调字段包括`learning_rate`、`batch_size`和`epoch`。其余参数都为用户自定义参数，可配置为`user_defined`，将在训练时被自动收集在训练日志中。

    - bounds: 列表，元素个数为2，第一个数为下界值min，第二个数为上界值max。范围是[min, max)，生成随机数方法是`numpy.random.uniform()`。
    - choice：列表，个数不限，参数取值从这个列表中的元素中选取。
    - type：必填，取值为`int`或`float`。
    - source：可选，取值为`system_defined`或`user_defined`。如果是自动收集的字段，默认为`system_defined`；否则，默认为`user_defined`。

    > `bounds`和`choice`有且仅有一个，必填。如果配置了`choice`，仅会从`choice`的列表中选取值；如果同时配置了`choice`和`type`，则`type`不生效。

## 使用示例

若用户要优化`learning_rate`、`batch_size`和`momentum`这几个超参数，且优化目标是`Accuracy`，则应按照如下示例配置yaml文件。

1. 配置config.yaml

    ```yaml
    command: sh /home/example/run_alexnet_ascend.sh
    summary_base_dir: /home/summaries
    tuner:
        name: gp
    target:
        group: metric
        name: Accuracy
        goal: maximize
    parameters:
        learning_rate:
            bounds: [0.00001, 0.001]
            type: float
        batch_size:
            choice: [32, 64, 128, 256]
            type: int
        momentum:
            source: user_defined
            choice: [0.8, 0.9]
            type: float
    ```

    > `momentum`和系统定义的变量不存在重名问题，可不设置source这个字段。

    **yaml配置同名字段会选取最后一个，请避免以下使用方式。**

    ```yaml
    parameters:
        learning_rate:
            bounds: [0.0005, 0.001]
            type: float
        learning_rate:
            source: user_defined
            bounds: [0.00002, 0.0001]
            type: float
    ```

2. 在训练脚本实例化`HyperConfig`对象

    (1) 用户需要实例化`HyperConfig`，并使用`HyperConfig`实例的参数变量作为训练脚本中对应参数的取值。  
    (2) 加上`SummaryCollector`来收集训练信息，包括超参和评估指标值等。

    如[Model Zoo](https://www.mindspore.cn/docs/zh-CN/master/note/network_list_ms.html)中的训练脚本：

    ```python
    ds_train = create_dataset_cifar10(args.data_path, batch_size)
    lr = Tensor(get_lr_cifar10(0, cfg.learning_rate, cfg.epoch_size, step_per_epoch))
    opt = nn.Momentum(network.trainable_params(), lr, cfg.momentum)

    model.train(cfg.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()]
    ```

    修改后：

    ```python
    from mindinsight.optimizer import HyperConfig
    config = HyperConfig()
    params = config.params

    # Replace batch_size with params.batch_size.
    ds_train = create_dataset_cifar10(args.data_path, params.batch_size)
    # Replace cfg.learning_rate with params.learning_rate.
    lr = Tensor(get_lr_cifar10(0, params.learning_rate, cfg.epoch_size, step_per_epoch))
    # Replace cfg.momentum with params.momentum.
    opt = nn.Momentum(network.trainable_params(), lr, params.momentum)

    # Instantiate SummaryCollector and add it to callback to automatically collect training information.
    summary_cb = SummaryCollector(config.summary_dir)
    model.train(cfg.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor(), summary_cb]
    ```

3. 运行

    在进行自动调参前请确保训练脚本可以正确执行。

    ```bash
    mindoptimizer --config ./config.yaml --iter 10
    ```

    > 将执行训练的命令填写在配置文件中，在能够成功运行该训练命令的目录下运行mindoptimizer程序。

4. 可视化

    基于config.yaml里面配置的summary_base_dir来启动MindInsight，启动方法可以查看[MindInsight启动命令](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/mindinsight_commands.html#启动服务)。

## 注意事项

1. 训练脚本由用户编写和维护，本工具不会自动修改训练脚本，如果训练脚本本身有错误，则使用本工具支持训练脚本时也会出错；
2. 本工具不对运行过程中的打印信息进行处理或修改；
3. 本工具需要确保调参过程可信，参数配置错误或脚本执行错误都会终止调参过程，用户可根据相应的提示来进行问题定位。
