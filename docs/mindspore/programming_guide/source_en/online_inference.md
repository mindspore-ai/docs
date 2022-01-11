# Online Inference with Checkpoint

`Ascend` `Inference Application`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_en/online_inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png"></a>

## Use the `model.eval` interface for model validation

### Local Storage

When the pre-trained models are saved in local, the steps of performing inference on validation dataset are as follows: firstly creating a model, then loading the model and parameters using `load_checkpoint` and `load_param_into_net` in `mindspore` module, and finally performing inference on the validation dataset once being created. The method of processing the validation dataset is the same as that of the training dataset.

```python
network = LeNet5(cfg.num_classes)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
model = Model(network, net_loss, metrics={"Accuracy": Accuracy()})

print("============== Starting Testing ==============")
param_dict = load_checkpoint(args.ckpt_path)
load_param_into_net(network, param_dict)
dataset = create_dataset(os.path.join(args.data_path, "test"),
                            cfg.batch_size,)
acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
print("============== {} ==============".format(acc))
```

In the preceding information:  
`model.eval` is an API for model validation. For details about the API, see <https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore.html#mindspore.Model.eval>.
> Inference sample code: <https://gitee.com/mindspore/models/blob/r1.6/official/cv/lenet/eval.py>.

### Remote Storage

When the pre-trained models are saved remotely, the steps of performing inference on the validation dataset are as follows: firstly determining which model to be used, then loading the model and parameters using `mindspore_hub.load`, and finally performing inference on the validation dataset once being created. The method of processing the validation dataset is the same as that of the training dataset.

```python
model_uid = "mindspore/ascend/0.7/googlenet_v1_cifar10"  # using GoogleNet as an example.
network = mindspore_hub.load(model_uid, num_classes=10)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
model = Model(network, net_loss, metrics={"Accuracy": Accuracy()})

print("============== Starting Testing ==============")
dataset = create_dataset(os.path.join(args.data_path, "test"),
                            cfg.batch_size,)
acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
print("============== {} ==============".format(acc))
```

In the preceding information:

`mindspore_hub.load` is an API for loading model parameters. Please check the details in <https://www.mindspore.cn/hub/docs/en/r1.6/hub.html#module-mindspore_hub>.

## Use the `model.predict` API to perform inference

   ```python
   model.predict(input_data)
   ```

   In the preceding information:  
   `model.predict` is an API for inference. For details about the API, see <https://www.mindspore.cn/docs/api/en/r1.6/api_python/mindspore.html#mindspore.Model.predict>.
