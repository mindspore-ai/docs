# Distributed Inference

<a href="https://gitee.com/mindspore/docs/blob/r1.10/tutorials/experts/source_en/parallel/distributed_inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.10/resource/_static/logo_source_en.png"></a>

## Overview

Distributed inference means use multiple devices for prediction. If data parallel or integrated save is used in training, the method of distributed inference is same with the [Single-card inference](https://www.mindspore.cn/tutorials/experts/en/r1.10/infer/inference.html#model-eval-model-validation). It is noted that each device should load one same checkpoint file.

This tutorial would focus on the process that the model slices are saved on each device in the distributed training process, and the model is reloaded according to the predication strategy in the inference stage. In view of the problem that there are too many parameters in the super large scale neural network model, the model can not be fully loaded into a single device for inference, so multiple devices can be used for distributed inference.

## Operational Practices

### Sample Code Description

>Download [Distributed inference](https://gitee.com/mindspore/docs/tree/r1.10/docs/sample_code/distributed_inference) sample code.

## Process of Distributed Inference

1. Execute training, and generate the checkpoint file and the model strategy file.

    > - The distributed training tutorial and sample code can be referred to [Distributed Parallel Training Example (Ascend)](https://www.mindspore.cn/tutorials/experts/en/r1.10/parallel/train_ascend.html).
    > - In the distributed Inference scenario, during the training phase, the `integrated_save` of `CheckpointConfig` interface should be set to `False`, which means that each device only saves the slice of model instead of the full model.
    > - `parallel_mode` of `set_auto_parallel_context` interface should be set to `auto_parallel` or `semi_auto_parallel`. The parallel mode is either auto parallel or semi-automatic parallelism.
    > - In addition, you need to specify `strategy_ckpt_save_file` to indicate the path of the strategy file.
    > - If pipeline distributed inference is used, then the pipeline parallel training also must be used. And the `device_num` and `pipeline_stages` used for pipeline training and inference must be the same.  While applying pipeline inference, `micro_size` is 1 and there is no need to call `PipelineCell`. Please refer to [Pipeline Parallel](https://www.mindspore.cn/tutorials/experts/en/r1.10/parallel/pipeline_parallel.html).

2. Set context and infer inference strategy according to the inference data.

    ```python
    set_auto_parallel_context(full_batch=True, parallel_mode='semi_auto_parallel', strategy_ckpt_load_file='./train_strategy.ckpt')
    network = Net()
    model = Model(network)
    predict_data = create_predict_data()
    predict_strategy = model.infer_predict_layout(predict_data)
    ```

    In the inference data:

    - `full_batch`: whether to load the dataset in full or not. When `True`, it indicates full load, and data of each device is the same. It must be set to `True` in this scenario.
    - `parallel_mode`: parallel mode, it must be `auto_parallel` or `semi_auto_parallel`.
    - `strategy_ckpt_load_file`: file path of the strategy generated in the training phase, which must be set in the distributed inference scenario.
    - `create_predict_data`: user-defined interface that returns predication data whose type is `Tensor`.
    - `infer_predict_layout`: generates predication strategy based on predication data.

3. Load checkpoint files, and load the corresponding model slice into each device based on the inference strategy.

    ```python
    ckpt_file_list = create_ckpt_file_list()
    load_distributed_checkpoint(network, ckpt_file_list, predict_strategy)
    ```

    In the preceding information:

    - `create_ckpt_file_list`：user-defined interface that returns a list of checkpoint file path in order of rank id.
    - `load_distributed_checkpoint`：merges model slices, then splits it according to the predication strategy, and loads it into the network.

    > For pipeline inference, each `stage` only needs to load the checkpoint file of self_stage.
    >
    > The `load_distributed_checkpoint` interface supports that predict_strategy is `None`, which is single device inference, and the process is different from distributed inference. The detailed usage can be referred to the [reference link](https://www.mindspore.cn/docs/en/r1.10/api_python/mindspore/mindspore.load_distributed_checkpoint.html#mindspore.load_distributed_checkpoint).

4. Execute inference and get the inference result.

    ```python
    model.predict(predict_data)
    ```

## Exporting MindIR Files on the Distributed Scenarios

When the super-large-scale neural network model has too many parameters, the MindIR format model cannot be completely loaded into a single card for inference. Multi-card should be used for distributed inference. In this case, multiple MindIR files need to be exported before the inference task. The specific methods are as follows:

First, you need to prepare checkpoint files and training strategy files.

The checkpoint file is generated during the training process. For the detailed use of checkpoint, refer to [Use CheckPoint](https://www.mindspore.cn/tutorials/en/r1.10/beginner/save_load.html#saving-and-loading-the-model)

The training strategy file needs to be generated by setting the context during training. The context configuration items are as follows:

`set_auto_parallel_context(strategy_ckpt_save_file='train_strategy.ckpt')`

In this way, after training, a training strategy file named `train_strategy.ckpt` will be generated in the set directory.

Before exporting the MindIR file, it is necessary to load the checkpoint file, the distributed training checkpoint file, the training strategy and the reasoning strategy need to be combined, so the reasoning strategy file can be generated.
The code to generate the inference strategy is as follows:

`predict_strategy = model.infer_predict_layout(predict_data)`

Then, use the method of loading distributed checkpoints to load the previously trained parameters to the network.
Code show as below:

`load_distributed_checkpoint(model, ckpt_file_list, predict_strategy)`

Finally, you can export the MindIR file in the distributed inference scenario.

The core code is as follows:

```python
# Configure the strategy file generated during the training process in the context
set_auto_parallel_context(strategy_ckpt_load_file='train_strategy.ckpt')
# Define network structure
network = Net()
model = Model(network)
# Get the reasoning strategy file
predict_strategy = model.infer_predict_layout(predict_data)
# Create checkpoint list
ckpt_file_list = create_ckpt_file_list()
# Load distributed parameters
load_distributed_checkpoint(model, ckpt_file_list, predict_strategy)
# Export distributed MindIR file
export(net, Tensor(input), file_name='net', file_format='MINDIR')
```

In the case of multi-card training and single-card inference, the usage of exporting MindIR is the same as that of single host.

> Distributed scenario exports MindIR file sample code: [distributed_export](https://gitee.com/mindspore/docs/tree/r1.10/docs/sample_code/distributed_export)
