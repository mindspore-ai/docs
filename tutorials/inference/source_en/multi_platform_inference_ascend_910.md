# Inference on the Ascend 910 AI processor

`Linux` `Ascend` `Inference Application` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Inference on the Ascend 910 AI processor](#inference-on-the-ascend-910-ai-processor)
    - [Inference Using a Checkpoint File with Single Device](#inference-using-a-checkpoint-file-with-single-device)
    - [Distributed Inference with Multiple Devices](#Distributed-inference-with-multiple-devices)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.1/tutorials/inference/source_en/multi_platform_inference_ascend_910.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Inference Using a Checkpoint File with Single Device

1. Use the `model.eval` interface for model validation.

   1.1 Local Storage

     When the pre-trained models are saved in local, the steps of performing inference on validation dataset are as follows: firstly creating a model, then loading the model and parameters using `load_checkpoint` and `load_param_into_net` in `mindspore.train.serialization` module, and finally performing inference on the validation dataset once being created. The method of processing the validation dataset is the same as that of the training dataset.

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

    In the preceding information:  
    `model.eval` is an API for model validation. For details about the API, see <https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/mindspore.html#mindspore.Model.eval>.
    > Inference sample code: <https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/lenet/eval.py>.

    1.2 Remote Storage

    When the pre-trained models are saved remotely, the steps of performing inference on the validation dataset are as follows: firstly determining which model to be used, then loading the model and parameters using `mindspore_hub.load`, and finally performing inference on the validation dataset once being created. The method of processing the validation dataset is the same as that of the training dataset.

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

    In the preceding information:

    `mindpsore_hub.load` is an API for loading model parameters. Please check the details in <https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore_hub/mindspore_hub.html#module-mindspore_hub>.

2. Use the `model.predict` API to perform inference.

   ```python
   model.predict(input_data)
   ```

   In the preceding information:  
   `model.predict` is an API for inference. For details about the API, see <https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/mindspore.html#mindspore.Model.predict>.

## Distributed Inference With Multi Devices

Distributed inference means use multiple devices for prediction. If data parallel or integrated save is used in training, the method of distributed inference is same with the above description. It is noted that each device should load one same checkpoint file.

This tutorial would focus on the process that the model slices are saved on each device in the distributed training process, and the model is reloaded according to the predication strategy in the inference stage. In view of the problem that there are too many parameters in the super large scale neural network model, the model can not be fully loaded into a single device for inference, so multiple devices can be used for distributed inference.

> Distributed inference sample code:
>
> <https://gitee.com/mindspore/docs/tree/r1.1/tutorials/tutorial_code/distributed_inference>

The process of distributed inference is as follows:

1. Execute training, generate the checkpoint file and the model strategy file.

    > - The distributed training tutorial and sample code can be referred to the link: <https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/distributed_training_ascend.html>.
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
    > <https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.load_distributed_checkpoint>.

4. Execute inference.

    ```python
    model.predict(predict_data)
    ```
