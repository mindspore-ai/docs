# Inference on the Ascend 910 AI processor

`Linux` `Ascend` `Inference Application` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/inference/source_en/multi_platform_inference_ascend_910.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Inference Using a Checkpoint File

1. Use the `model.eval` interface for model validation. 

   1.1 Local Storage

     When the pre-trained models are saved in local, the steps of performing inference on validation dataset are as follows: firstly creating a model, then loading model and parameters using `load_checkpoint` and `load_param_into_net` in `mindspore.train.serialization` module, and finally performing inference on validation dataset once created. The processing method of the validation dataset is the same as that of the training dataset.

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
    `model.eval` is an API for model validation. For details about the API, see <https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.html#mindspore.Model.eval>.
    > Inference sample code: <https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/lenet/eval.py>.

    1.2 Remote Storage
    
    When the pre-trained models are saved remotely, the steps of performing inference on validation dataset are as follows: firstly determine which model to be used, then loading model and parameters using `mindspore_hub.load`, and finally performing inference on validation dataset once created. The processing method of the validation dataset is the same as that of the training dataset.

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
        
    `mindpsore_hub.load` is an API for loading model parameters. PLease check the details in <https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore_hub/mindspore_hub.html#module-mindspore_hub>.

2. Use the `model.predict` API to perform inference.
   ```python
   model.predict(input_data)
   ```
   In the preceding information:  
   `model.predict` is an API for inference. For details about the API, see <https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.html#mindspore.Model.predict>.
