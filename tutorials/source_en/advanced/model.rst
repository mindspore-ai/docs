.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/model.rst

Advanced Encapsulation: Model
==============================

.. toctree::
  :maxdepth: 1
  :hidden:

  model/callback
  model/metric

Generally, defining a training and evaluation network and running it directly can meet basic requirements.

On the one hand, ``Model`` can simplify code to some extent. For example, you do not need to manually traverse datasets.
If you do not need to customize ``nn.TrainOneStepCell``, you can use ``Model`` to automatically build a training network.
You can use the ``eval`` API of ``Model`` to evaluate the model and directly output the evaluation result.
You do not need to manually invoke the ``clear``, ``update``, and ``eval`` functions of evaluation metrics.

On the other hand, ``Model`` provides many high-level functions, such as data offloading and mixed precision.
Without the help of ``Model``, it takes a long time to customize these functions by referring to ``Model``.

The following describes MindSpore models and how to use ``Model`` for model training, evaluation, and inference.

.. figure:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/tutorials/source_en/advanced/model/images/model.png
   :alt: model

.. code:: python 

    import mindspore
    from mindspore import nn
    from mindspore.dataset import vision, transforms
    from mindspore.dataset import MnistDataset
    from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor

Introduction to Model
---------------------

`Model <https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Model.html#mindspore.train.Model>`__
is a high-level API provided by MindSpore for model training, evaluation, and inference. The common parameters of the API are as follows:

-  ``network``: neural network used for training or inference.
-  ``loss_fn``: used loss function.
-  ``optimizer``: used optimizer.
-  ``metrics``: evaluation function used for model evaluation.
-  ``eval_network``: network used for model evaluation. If the network is not defined, ``Model`` uses ``network`` and ``loss_fn`` for encapsulation.

``Model`` provides the following APIs for model training, evaluation, and inference:

- ``fit``: Evaluate the model while training.
-  ``train``: used for model training on the training set.
-  ``eval``: used to evaluate the model on the evaluation set.
-  ``predict``: performs inference on a group of input data and outputs the prediction result.

Using the Model API
~~~~~~~~~~~~~~~~~~~

For a neural network in a simple scenario, you can specify the feedforward network ``network``, loss function ``loss_fn``, optimizer ``optimizer``,
and evaluation function ``metrics`` when defining ``Model``.

Download and Process Dataset
----------------------------

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

Define Model
------------

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

Define loss function and optimizer
----------------------------------

To train neural network model, loss function and optimizer function need to be defined.

-  The loss function here uses ``CrossEntropy Loss`` .

-  The optimizer uses ``SGD`` here.

.. code:: python

    # Instantiate loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.SGD(model.trainable_params(), 1e-2)

Train and Save Model
--------------------

Before starting the training, MindSpore needs to state in advance whether the network model needs to save the intermediate process and results
during the training process. Therefore, ``ModelCheckpoint`` is used to save the network model and parameters for subsequent fine tuning.

.. code:: python

    steps_per_epoch = train_dataset.get_dataset_size()
    config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch)
    
    ckpt_callback = ModelCheckpoint(prefix="mnist", directory="./checkpoint", config=config)
    loss_callback = LossMonitor(steps_per_epoch)

The ``model.fit`` interface provided by MindSpore makes it easy to train and evaluate the network, and ``LossMonitor`` can monitor the changes of ``loss`` values during the training process.

.. code:: python

    trainer = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'accuracy'})
    
    trainer.fit(10, train_dataset, test_dataset, callbacks=[ckpt_callback, loss_callback])


.. raw:: html

    <div class="highlight"><pre>
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
    </pre></div>

During training, the loss value will be printed, and the loss value will fluctuate, but in general, the loss value will gradually decrease and
the accuracy will gradually improve. The loss values run by each person are random and not necessarily identical.

The results obtained by running the test dataset of the model verify the generalization ability of the model:

1. Use ``model.eval`` to read in the test dataset.
2. Use the saved model parameters for reasoning.

.. code:: python

    acc = trainer.eval(test_dataset)
    acc

.. raw:: html

    <div class="highlight"><pre>
    {'accuracy': 0.9632}
    </pre></div>

The model accuracy data can be seen from the print information. In the example, the accuracy data reaches more than 95%, and the model quality
is good. As the number of network iterations increases, the accuracy of the model will be further improved.
