# Distributed Inference With Multi Devices

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/distributed_inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

Distributed inference means use multiple devices for prediction. If data parallel or integrated save is used in training, the method of distributed inference is same with the above description. It is noted that each device should load one same checkpoint file.

This tutorial would focus on the process that the model slices are saved on each device in the distributed training process, and the model is reloaded according to the predication strategy in the inference stage. In view of the problem that there are too many parameters in the super large scale neural network model, the model can not be fully loaded into a single device for inference, so multiple devices can be used for distributed inference.

> Distributed inference sample code:
>
> <https://gitee.com/mindspore/docs/tree/r1.3/docs/sample_code/distributed_inference>

The process of distributed inference is as follows:

1. Execute training, generate the checkpoint file and the model strategy file.

    > - The distributed training tutorial and sample code can be referred to the link: <https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.3/distributed_training_ascend.html>.
    > - In the distributed Inference scenario, during the training phase, the `integrated_save` of `CheckpointConfig` interface should be set to `False`, which means that each device only saves the slice of model instead of the full model.
    > - `parallel_mode` of `set_auto_parallel_context` interface should be set to `auto_parallel` or `semi_auto_parallel`.
    > - In addition, you need to specify `strategy_ckpt_save_file` to indicate the path of the strategy file.

2. Set context and infer predication strategy according to the predication data.

    ```python
    context.set_auto_parallel_context(full_batch=True, parallel_mode='semi_auto_parallel', strategy_ckpt_load_file='./train_strategy.ckpt')
    network = Net()
    model = Model(network)
    predict_data = create_predict_data()
    predict_strategy = model.infer_predict_layout(predict_data)
    ```

    In the preceding information:

    - `full_batch`: whether to load the dataset in full or not. When `True`, it indicates full load, and data of each device is the same. It must be set to `True` in this scenario.
    - `parallel_mode`: parallel mode, it must be `auto_parallel` or `semi_auto_parallel`.
    - `strategy_ckpt_load_file`: file path of the strategy generated in the training phase, which must be set in the distributed inference scenario.
    - `create_predict_data`: user-defined interface that returns predication data whose type is `Tensor`.
    - `infer_predict_layout`: generates predication strategy based on predication data.

3. Load checkpoint files, and load the corresponding model slice into each device based on the predication strategy.

    ```python
    ckpt_file_list = create_ckpt_file_list()
    load_distributed_checkpoint(network, ckpt_file_list, predict_strategy)
    ```

    In the preceding information:

    - `create_ckpt_file_list`：user-defined interface that returns a list of checkpoint file path in order of rank id.
    - `load_distributed_checkpoint`：merges model slices, then splits it according to the predication strategy, and loads it into the network.

    > The `load_distributed_checkpoint` interface supports that predict_strategy is `None`, which is single device inference, and the process is different from distributed inference. The detailed usage can be referred to the link:
    > <https://www.mindspore.cn/docs/api/zh-CN/r1.3/api_python/mindspore.html#mindspore.load_distributed_checkpoint>.

4. Execute inference.

    ```python
    model.predict(predict_data)
    ```
