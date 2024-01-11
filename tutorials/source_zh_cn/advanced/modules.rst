.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_modelarts.svg
    :target: https://authoring-modelarts-cnnorth4.huaweicloud.com/console/lab?share-url-b64=aHR0cHM6Ly9taW5kc3BvcmUtd2Vic2l0ZS5vYnMuY24tbm9ydGgtNC5teWh1YXdlaWNsb3VkLmNvbS9ub3RlYm9vay9yMi4yL3R1dG9yaWFscy96aF9jbi9hZHZhbmNlZC9tb2R1bGVzL21pbmRzcG9yZV9jZWxsLmlweW5i&imageid=4c43b3ad-9df7-4b83-a096-c775dc4ba243
.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_notebook.svg
    :target: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/r2.2/tutorials/zh_cn/advanced/modules/mindspore_cell.ipynb
.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_download_code.svg
    :target: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/r2.2/tutorials/zh_cn/advanced/modules/mindspore_cell.py
.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.2/tutorials/source_zh_cn/advanced/modules/cell.ipynb
    :alt: 查看源文件

模型模块自定义
==============

.. toctree::
  :maxdepth: 1
  :hidden:
  
  modules/layer
  modules/initializer
  modules/loss
  modules/optimizer

基础用法示例
------------

神经网络模型由各种层(Layer)构成，MindSpore提供构造神经网络层的基础单元Cell，基于Cell进行神经网络封装。下面使用Cell构造经典模型AlexNet。

.. figure:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/tutorials/source_zh_cn/advanced/modules/images/AlexNet.ppm
   :alt: alextnet

如图所示，AlexNet由5个卷积层与3个全连接层串联构成，我们使用\ ``mindspore.nn``\ 提供的神经网络层接口进行构造。

.. code:: python

    from mindspore import nn

下面的代码展示了如何使用\ ``nn.Cell``\ 快速构造AlexNet。其中：

-  顶层神经网络继承\ ``nn.Cell``\ ，为嵌套结构；
-  每个神经网络层都是\ ``nn.Cell``\ 的子类；
-  ``nn.SequentialCell``\ 可以在定义顺序结构的模型时进行简化。

.. code:: python

    class AlexNet(nn.Cell):
        def __init__(self, num_classes=1000, dropout=0.5):
            super().__init__()
            self.features = nn.SequentialCell(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, pad_mode='pad', padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, pad_mode='pad', padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, pad_mode='pad', padding=1),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, pad_mode='pad', padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, pad_mode='pad', padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.classifier = nn.SequentialCell(
                nn.Dropout(p=dropout),
                nn.Dense(256 * 6 * 6, 4096),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Dense(4096, 4096),
                nn.ReLU(),
                nn.Dense(4096, num_classes),
            )
    
        def construct(self, x):
            x = self.features(x)
            x = x.view(x.shape[0], 256 * 6 * 6)
            x = self.classifier(x)
            return x

在定义模型的过程中，\ ``construct``\ 方法内可使用Python语法进行模型结构的任意构造，如条件、循环等控制流语句。但在进行即时编译(Just
In
Time)时，需通过编译器进行语法解析，此时存在语法限制，具体参考：\ `静态图语法支持 <https://www.mindspore.cn/docs/zh-CN/r2.2/note/static_graph_syntax_support.html>`__\ 。

完成模型构造后，我们构造一个单样本数据，将其送入实例化的AlexNet中，即可求得正向结果。

.. code:: python

    import numpy as np
    import mindspore
    from mindspore import Tensor
    
    x = Tensor(np.random.randn(1, 3, 224, 224), mindspore.float32)

.. code:: python

    network = AlexNet()
    logits = network(x)
    print(logits.shape)


.. raw:: html

    <div class="highlight"><pre>
    (1, 1000)
    </pre></div>


更多使用场景
------------

除基础的网络结构构造外，我们分别对神经网络层(Layer)、损失函数(Loss)和优化器(Optimizer)，神经网络层需要的参数(Parameter)及其初始化方法(Initializer)的构造等场景进行详细介绍。

-  `Cell与参数 <https://www.mindspore.cn/tutorials/zh-CN/r2.2/advanced/modules/layer.html>`__
-  `参数初始化 <https://www.mindspore.cn/tutorials/zh-CN/r2.2/advanced/modules/initializer.html>`__
-  `损失函数 <https://www.mindspore.cn/tutorials/zh-CN/r2.2/advanced/modules/loss.html>`__
-  `优化器 <https://www.mindspore.cn/tutorials/zh-CN/r2.2/advanced/modules/optimizer.html>`__
