# 在线推理

`Ascend` `推理应用`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/online_inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 使用`model.eval`接口进行模型验证

### 模型已保存在本地

首先构建模型，然后使用`mindspore`模块的`load_checkpoint`和`load_param_into_net`从本地加载模型与参数，传入验证数据集后即可进行模型推理，验证数据集的处理方式与训练数据集相同。

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

其中，  
`model.eval`为模型验证接口，对应接口说明：<https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.Model.eval>。

> 推理样例代码：<https://gitee.com/mindspore/models/blob/master/official/cv/lenet/eval.py>。

### 使用MindSpore Hub从华为云加载模型

首先构建模型，然后使用`mindspore_hub.load`从云端加载模型参数，传入验证数据集后即可进行推理，验证数据集的处理方式与训练数据集相同。

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

其中，  
`mindspore_hub.load`为加载模型参数接口，对应接口说明：<https://www.mindspore.cn/hub/docs/zh-CN/master/index.html#module-mindspore_hub>。

## 使用`model.predict`接口进行推理操作

```python
model.predict(input_data)
```

其中，  
`model.predict`为推理接口，对应接口说明：<https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.Model.predict>。
