# Use Mindoptimizer to Tune Hyperparameters

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindinsight/docs/source_en/hyper_parameters_auto_tuning.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

There are two kinds of parameters in machine learning. One is the model internal parameters, relying on training data and algorithms to tune the model parameters. And the other is the model external setting parameters, they need to be manually configured, such parameters are called hyperparameters. Because different hyperparameters impact the performance of model, hyperparameters are highly important in training tasks. Traditional methods require manual analysis of hyperparameters, manual debugging, and configuration, which consumes time and effort. MindInsight parameter tuning command can be used for automatic parameter tuning. Based on the parameter tuning configuration information provided by users, parameters can be automatically configured and model training can be performed.

MindInsight provides `mindoptimizer`. This tuning command can extract past training summaries from the training log according to the user configuration, analyze past training records and recommend hyperpameters, and finally automate training scripts. When using it, users need to configure information such as the scope of hyperparameters in yaml format. And then users need to replace the hyperparameters in the training script according to the tutorial, with the aim of synchronizing the auto-recommended hyperparameters into the training script. Currently, only the Gauss process tuning method is supported, and other methods are gradually supported.

## Installation

This tool is a submodule of MindInsight. After MindInsight is installed, you can use the MindInsight parameter tuning command. For details about how to install MindInsight, see the [installation guide](https://gitee.com/mindspore/mindinsight/blob/master/README.md#).

## Usage

MindInsight provides parameters tuning command. The command-line interface (CLI) provides the following commands:

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

## Configuration File Rules

The file format of the configuration file is yaml, which requires configurations of running command, the root directory of training summaries, tuning method, optimization objectives, and hyperparameters information. It is necessary to configure the information about hyperparameters including the range of values, types and sources, etc. MindInsight extracts training records from the training summary based on configured hyperparameters and optimization objectives, such as learning and accuracy. They can be used by recommended algorithms to analyze their relationships and better recommend hyperparameters.

1. Configure the running command

    Use `command` to configure running command, such as `command: python train.py`. The running command is executed directly after the tuning program recommends hyperparameters.

2. Configure the root directory of training summaries

    The `summary_base_dir` is the root directory of training summaries. It is also used for the extraction of training records, which makes hyperparameters better recommended. At the same time, it is recommended that users add `SummaryColletor` in their training scripts to collect training information, you can view the [summary collection tutorial](https://www.mindspore.cn/mindinsight/docs/en/master/summary_record.html). The tuning command generates a subdirectory path based on the configured `summary_base_dir`, which can be configured to record the training record at `SummaryColletor`. Therefore, after training, the training information is recorded in the subdirecte of the root directory of training summaries, and the training information can be used as a training record to recommend the next required hyperparameters. Configure the `summary_base_dir`, such as `/home/summaries`.

3. Configure the parameter tuning method

    Use `name` to specify the name of an acquisition function, and `args` tp specify parameters of the acquisition function.

    The current algorithm is Gaussian process regressor(GP). The acquisition functhon of GP is optional, and its range is in [`ucb`, `pi`,`ei`]. The default value is `ucb`.

    - Upper confidence bound (UCB)
    - Probability of improvement (PI)
    - Expected improvement (EI)

    For example:

    ```yaml
    tuner:
        name: gp
        args:
            method: ucb
    ```

4. Configure the parameter tuning target

    You can select loss or self-defined metrics as the target.

    Configuration description:
    - group: This parameter is optional. The value can be `system_defined` or `metric`. The default value is `system_defined`. Use `group` to configure the group in which the optimization target is located, such as the system custom collection field, which is the `system_defined` group. However, other evaluation metrics used in `Model()`, such as `model = Model(net, loss_fn=loss, optimizer=None, metrics={'Accuracy'})`. `Accuracy` belongs to the metrics, so the group is `metric`.
    - goal: This parameter is optional. The value can be `minimize` or `maximize`. The default value is `minimize`. Use `goal` to indicate the optimization direction of the target. For example, if `Accuracy` is higher, the performance of model is better, so the `goal` needs to be configured as 'maximize'.

    Config `loss`:

    ```yaml
    target:
        name:loss
    ```

    Config `Accuracy` in metrics:

    ```yaml
    target:
        group: metric
        name: Accuracy
        goal: maximize
    ```

5. Configure hyperparameters bounds, choice, type, and source

    Configuration fields for hyperparameters consist of `bounds`, `choices`, `type`, and `source`. The fields of hyperparameters configured here are used for extraction of training summaries and recommendation of hyperparameters. In addition, `bounds`, `choice`, and `type` affect the recommendation of hyperparameters. `bounds` are configured as the upper and lower boundaries of the hyperparameters, `choice` indicates that which recommended values are selected, and `type` is the type of parameter configured.

    The tunable fields currently collected by the system customization include `learning_rate`, `batch_size` and `epoch`. Other parameters are user-defined parameters will be automatically collected in the training summary during training if the source is configured as `user_defined`.

    - bounds: a list. The number of elements is 2. The first number is the lower bound min, and the second number is the upper bound max. The value range is [min, max). The method for generating a random number is `numpy.random.uniform()`.
    - choice: a list. The number of values is not limited. Values are selected from the elements in the list.
    - type: This parameter is mandatory and should be set to `int` or `float`.
    - source: This parameter is optional. The value should be `system_defined` or `user_defined`. If the name of parameter exists in system-defined field, the default value is `system_defined`, otherwise, the default value is `user_defined`.

    > You need to choose either `bounds` or `choice`. If you have configured `choice`, values are selected from the configured list only, and if you have configured both `choice` and `type`, `type` does not take effect.

## Usage Examples

If you want to optimize the `learning_rate`, `batch_size`, and `momentum`, and the optimization objective is `Accuracy`, configure the YAML file as follows:

1. Configure config.yaml

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

    > The name of `momentum` is not the same as that of the variable defined by the system. Therefore, you do not need to set the source field.

    **If the fields with the same name exist in the YAML file, the last one will be selected. Do not use the following method.**

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

2. Instantiate the `HyperConfig` object in the training script

    (1) After instantiating `HyperConfig`, use the parameter variables of the `HyperConfig` instance as the values of the corresponding parameters in the training script.  
    (2) Please add `SummaryCollector` to collect lineage information, including hyperparameters and evaluation metrics.

    For example, the training script in [Model Zoo](https://www.mindspore.cn/docs/note/en/master/network_list_ms.html) is as follows:

    ```python
    ds_train = create_dataset_cifar10(args.data_path, batch_size)
    lr = Tensor(get_lr_cifar10(0, cfg.learning_rate, cfg.epoch_size, step_per_epoch))
    opt = nn.Momentum(network.trainable_params(), lr, cfg.momentum)

    model.train(cfg.epoch_size, ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()]
    ```

    After the modification:

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

3. Execution

    Please make sure that the training script can be executed correctly before performing automatic tuning.

    ```bash
    mindoptimizer --config ./config.yaml --iter 10
    ```

    > Please fill in the training command to execute the training in the configuration file, and run the automatic tuning program in the directory where the training command can be successfully run.

4. Visualization

    Enable MindInsight based on summary_base_dir configured in config.yaml. For details about the visualization method, see the [MindInsight start tutorial](https://www.mindspore.cn/mindinsight/docs/en/master/mindinsight_commands.html#start-the-service).

## Notices

1. The training script is written and maintained by users. This tool does not automatically modify the training script. If the training script is incorrect, an error occurs when this tool is used to support the training script.
2. This tool does not process or modify the printed information during the running process.
3. Ensure that the parameter tuning process is trustworthy. If a parameter configuration error or script execution error occurs, the parameter tuning process will be terminated. You can locate the fault based on the displayed information.
