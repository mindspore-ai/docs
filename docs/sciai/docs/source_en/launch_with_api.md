# Launching Model with API

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/sciai/docs/source_en/launch_with_api.md)&nbsp;&nbsp;

MindSpore SciAI provides users with a high order interface `AutoModel`, with which the supported model in the model library can be instantiated with one line code.

User can launch training and evaluation process with `AutoModel`.

## Obtaining the Network with AutoModel

User can use the function `AutoModel.from_pretrained` to get the network models, which are supported in SciAI.

Here we use the model Conservative Physics-Informed Neural Networks (CPINNs) as example. For the codes of CPINNs model, please refer to the [link](https://gitee.com/mindspore/mindscience/tree/master/SciAI/sciai/model/cpinns).

The fundamental idea about this model can be found in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782520302127).

```python
from sciai.model import AutoModel

# obtain the cpinns model.
model = AutoModel.from_pretrained("cpinns")
```

## Training and Fine-tuning with AutoModel

User can use the function `AutoModel.train` to train the neural networks, and before training, user can use `AutoModel.update_config` to configure the training parameters or finetune the model by loading the `.ckpt` file.

The acceptable arguments for API `AutoModel.update_config` depends on the model instantiated.

```python
from sciai.model import AutoModel

# obtain the cpinns model.
model = AutoModel.from_pretrained("cpinns")
# (optional) load the ckpt file and initialize the model based on the loaded parameters.
model.update_config(load_ckpt=True, load_ckpt_path="./checkpoints/your_file.ckpt", epochs=500)
# train the network with default configuration,
# the figures, data and logs generated will be saved in your execution path.
model.train()
```

## Evaluating with AutoModel

User can evaluate the trained networks with function `AutoModel.evaluate`.

This function will load the `.ckpt` files provided in SciAI by default. Alternatively, user can load their own `.ckpt` file with interface `AutoModel.update_config`.

```python
from sciai.model import AutoModel

# obtain the cpinns model
model = AutoModel.from_pretrained("cpinns")
# (optional) load the ckpt file provided by user
model.update_config(load_ckpt=True, load_ckpt_path="./checkpoints/your_file.ckpt")
# evaluate the model
model.evaluate()
```
