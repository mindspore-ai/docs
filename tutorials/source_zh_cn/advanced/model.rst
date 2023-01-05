.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_notebook.png
    :target: https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/tutorials/zh_cn/advanced/model/mindspore_model.ipynb
.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_download_code.png
    :target: https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/master/tutorials/zh_cn/advanced/model/mindspore_model.py
.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced/model/model.ipynb

高阶封装：Model
===============

.. toctree::
  :maxdepth: 1
  :hidden:

  model/callback
  model/metric

通常情况下，定义训练和评估网络并直接运行，已经可以满足基本需求。

一方面，\ ``Model``\ 可以在一定程度上简化代码。例如：无需手动遍历数据集；在不需要自定义\ ``nn.TrainOneStepCell``\ 的场景下，可以借助\ ``Model``\ 自动构建训练网络；可以使用\ ``Model``\ 的\ ``eval``\ 接口进行模型评估，直接输出评估结果，无需手动调用评价指标的\ ``clear``\ 、\ ``update``\ 、\ ``eval``\ 函数等。

另一方面，\ ``Model``\ 提供了很多高阶功能，如数据下沉、混合精度等，在不借助\ ``Model``\ 的情况下，使用这些功能需要花费较多的时间仿照\ ``Model``\ 进行自定义。

本文档首先对MindSpore的Model进行基本介绍，然后重点讲解如何使用\ ``Model``\ 进行模型训练、评估和推理。

.. figure:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_zh_cn/advanced/model/images/model.png
   :alt: model


.. code:: python

    import mindspore
    from mindspore import nn
    from mindspore import ops
    from mindspore.dataset import vision, transforms
    from mindspore.dataset import MnistDataset
    from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor


Model基本介绍
-------------

`Model <https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model>`__\ 是MindSpore提供的高阶API，可以进行模型训练、评估和推理。其接口的常用参数如下：

-  ``network``\ ：用于训练或推理的神经网络。
-  ``loss_fn``\ ：所使用的损失函数。
-  ``optimizer``\ ：所使用的优化器。
-  ``metrics``\ ：用于模型评估的评价函数。
-  ``eval_network``\ ：模型评估所使用的网络，未定义情况下，\ ``Model``\ 会使用\ ``network``\ 和\ ``loss_fn``\ 进行封装。

``Model``\ 提供了以下接口用于模型训练、评估和推理：

-  ``fit``\ ：边训练边评估模型。
-  ``train``\ ：用于在训练集上进行模型训练。
-  ``eval``\ ：用于在验证集上进行模型评估。
-  ``predict``\ ：用于对输入的一组数据进行推理，输出预测结果。

使用Model接口
~~~~~~~~~~~~~

对于简单场景的神经网络，可以在定义\ ``Model``\ 时指定前向网络\ ``network``\ 、损失函数\ ``loss_fn``\ 、优化器\ ``optimizer``\ 和评价函数\ ``metrics``\ 。

下载并处理数据集
----------------

.. code:: python

    # Download data from open datasets
    from download import download
    
    url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
          "notebook/datasets/MNIST_Data.zip"
    path = download(url, "./", kind="zip")
    
    
    def datapipe(path, batch_size):
        image_transforms = [
            vision.Rescale(1.0 / 255.0, 0),
            vision.Normalize(mean=(0.1307,), std=(0.3081,)),
            vision.HWC2CHW()
        ]
        label_transform = transforms.TypeCast(mindspore.int32)
        
        dataset = MnistDataset(path)
        dataset = dataset.map(image_transforms, 'image')
        dataset = dataset.map(label_transform, 'label')
        dataset = dataset.batch(batch_size)
        return dataset
    
    train_dataset = datapipe('MNIST_Data/train', 64)
    test_dataset = datapipe('MNIST_Data/test', 64)


创建模型
--------

.. code:: python

    # Define model
    class Network(nn.Cell):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.dense_relu_sequential = nn.SequentialCell(
                nn.Dense(28*28, 512),
                nn.ReLU(),
                nn.Dense(512, 512),
                nn.ReLU(),
                nn.Dense(512, 10)
            )
    
        def construct(self, x):
            x = self.flatten(x)
            logits = self.dense_relu_sequential(x)
            return logits
    
    model = Network()

定义损失函数和优化器
--------------------

要训练神经网络模型，需要定义损失函数和优化器函数。

-  损失函数这里使用交叉熵损失函数\ ``CrossEntropyLoss``\ 。
-  优化器这里使用\ ``SGD``\ 。

.. code:: python

    # Instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.SGD(model.trainable_params(), 1e-2)

训练及保存模型
--------------

在开始训练之前，MindSpore需要提前声明网络模型在训练过程中是否需要保存中间过程和结果，因此使用\ ``ModelCheckpoint``\ 接口用于保存网络模型和参数，以便进行后续的Fine-tuning（微调）操作。

.. code:: python

    steps_per_epoch = train_dataset.get_dataset_size()
    config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)
    
    ckpt_callback = ModelCheckpoint(prefix="mnist", directory="./checkpoint", config=config)
    loss_callback = LossMonitor(steps_per_epoch)

通过MindSpore提供的\ ``model.fit``\ 接口可以方便地进行网络的训练与评估，\ ``LossMonitor``\ 可以监控训练过程中\ ``loss``\ 值的变化。

.. code:: python

    trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    trainer.fit(10, train_dataset, test_dataset, callbacks=[ckpt_callback, loss_callback])


.. code:: none

    epoch: 1 step: 938, loss is 0.602992594242096
    Eval result: epoch 1, metrics: {'accuracy': 0.8435}
    epoch: 2 step: 938, loss is 0.2797124981880188
    Eval result: epoch 2, metrics: {'accuracy': 0.9003}
    epoch: 3 step: 938, loss is 0.32015785574913025
    Eval result: epoch 3, metrics: {'accuracy': 0.9179}
    epoch: 4 step: 938, loss is 0.17153620719909668
    Eval result: epoch 4, metrics: {'accuracy': 0.9308}
    epoch: 5 step: 938, loss is 0.18772485852241516
    Eval result: epoch 5, metrics: {'accuracy': 0.9382}
    epoch: 6 step: 938, loss is 0.45641791820526123
    Eval result: epoch 6, metrics: {'accuracy': 0.946}
    epoch: 7 step: 938, loss is 0.11519066989421844
    Eval result: epoch 7, metrics: {'accuracy': 0.9506}
    epoch: 8 step: 938, loss is 0.43486487865448
    Eval result: epoch 8, metrics: {'accuracy': 0.9555}
    epoch: 9 step: 938, loss is 0.1941455900669098
    Eval result: epoch 9, metrics: {'accuracy': 0.9588}
    epoch: 10 step: 938, loss is 0.13441434502601624
    Eval result: epoch 10, metrics: {'accuracy': 0.9632}


训练过程中会打印loss值，loss值会波动，但总体来说loss值会逐步减小，精度逐步提高。每个人运行的loss值有一定随机性，不一定完全相同。

通过模型运行测试数据集得到的结果，验证模型的泛化能力：

1. 使用\ ``model.eval``\ 接口读入测试数据集。
2. 使用保存后的模型参数进行推理。

.. code:: python

    acc = trainer.eval(test_dataset)
    acc

.. code:: none

    {'accuracy': 0.9632}



可以在打印信息中看出模型精度数据，示例中精度数据达到95%以上，模型质量良好。随着网络迭代次数增加，模型精度会进一步提高。
