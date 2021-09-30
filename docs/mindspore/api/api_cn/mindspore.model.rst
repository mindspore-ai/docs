mindspore.Model
================

.. py:class:: mindspore.Model(network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", acc_level="O0", **kwargs)

   训练模型及测试的高阶API接口， `Model` 会根据用户传入的参数封装可训练或推理的实例。

   **参数** ：

      - **network** (`Cell`) – 一个训练或测试的神经网络模型。
      - **loss_fn** (`Cell`) - 损失函数，如果损失函数是None， `network` 需要包含损失函数逻辑以及梯度计算，如果有并行计算逻辑也需加入。默认值：None。
      - **optimizer** (`Cell`) - 更新网络权重的优化器。默认值：None。
      - **metrics** (`Union[dict, set]`) - 在训练和测试时的模型评价指标。例如: {‘accuracy’, ‘recall’}。默认值：None。
      - **eval_network** (`Cell`) -  指定用于评估的模型。如果没有定义， *network* 和 *loss_fn* 将会被封装成 *eval_network* 。默认值：None。
      - **eval_indexes** (`list`) -  在定义 *eval_network* 时，如果 *eval_indexes* 为None， *eval_network* 的所有输出将传给 *metrics* 中，否则 *eval_indexes* 必须包含三个元素，为损失值、预测值和标签在输出中的位置。损失值将传给损失评价函数，而预测值和标签在输出中的位置传给其他评价函数。默认值：None。
      - **amp_level** (`str`) - 在 *mindspore.amp.build_train_network* 中的可选参数 *level* ， *level* 为混合精度的等级，该参数支持 [“O0”, “O2”, “O3”, “auto”]。默认值：“O0”。
         
         - O0: 无变化。
         - O2: 将网络训练精度转为float16，batchnorm保持在float32精度进行，同时使用动态loss scale策略。
         - O3: 将网络训练精度转为float16，同时设置属性keep_batchnorm_fp32等于False。
         - auto: 在不同处理器上会将 *amp_level* 设置为专家推荐的 *level* ，如在GPU上设为02，在Ascend上设为03。但这并不总是符合实际要求，建议在不同网络模型上用户要根据情况自定义设置 *amp_level* 。
      在GPU上建议使用O2，在Ascend上建议使用O3。关于 *amp_level* 详见 *mindpore.amp.build_train_network* 。

      - **boost_level** (str) – mindspore.boost中的参数级别选项，用于提升模式训练的级别。支持[“O0”、“O1”、“O2”]。 默认值：“O0”。

         - O0：无变化。
         - O1：开启boost模式，性能提升20%左右，精度与原精度相同。
         - O2：开启boost模式，性能提升30%左右，精度下降不到3%。

   **样例** :

      .. code-block::

              >>> from mindspore import Model, nn
              >>> class Net(nn.Cell):
              >>>     def __init__(self, num_class=10, num_channel=1):
              >>>         super(Net, self).__init__()
              >>>         self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
              >>>         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
              >>>         self.fc1 = nn.Dense(16*5*5, 120, weight_init='ones')
              >>>         self.fc2 = nn.Dense(120, 84, weight_init='ones')
              >>>         self.fc3 = nn.Dense(84, num_class, weight_init='ones')
              >>>         self.relu = nn.ReLU()
              >>>         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
              >>>         self.flatten = nn.Flatten()
              >>>
              >>>     def construct(self, x):
              >>>         x = self.max_pool2d(self.relu(self.conv1(x)))
              >>>         x = self.max_pool2d(self.relu(self.conv2(x)))
              >>>         x = self.flatten(x)
              >>>         x = self.relu(self.fc1(x))
              >>>         x = self.relu(self.fc2(x))
              >>>         x = self.fc3(x)
              >>>         return x
              >>> net = Net()
              >>> loss = nn.SoftmaxCrossEntropyWithLogits()
              >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
              >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
              >>> # 如何构建数据集，请参考官方网站上关于【数据集】的章节。
              >>> dataset = create_custom_dataset()
              >>> model.train(2, dataset)

   .. py:method:: build(train_dataset=None, valid_dataset=None, sink_size=-1)

      用数据下沉模式建立计算图和数据图。

      .. warning:: 这是一个实验性的原型，可能会被改变和/或删除。

      .. note:: 预构建计算图目前仅支持*GRAPH_MODE*和*Ascend*，如果已经使用了该接口去构建计算图，那么‘model.train’会直接执行计算图。仅支持数据下沉模式。

      **参数** ：

         - **train_dataset** (`Dataset`) – 一个训练集迭代器。如果定义了 *train_dataset* ，将会初始化训练集的计算图。默认值：None。
         - **valid_dataset** (`Dataset`) - 一个验证集迭代器。如果定义了 *valid_dataset* ，将会初始化验证集的计算图，并且 *Model* 中的 *metrics* 不可设置为None。默认值：None。
         - **sink_size** (`int`) - 控制每次数据下沉的数据量，默认值：-1。


      **样例** :

         .. code-block::

               >>> from mindspore import Model, nn, FixedLossScaleManager
               >>>
               >>> # 如何构建数据集，请参考官方网站上关于【数据集】的章节。
               >>> dataset = create_custom_dataset()
               >>> net = Net()
               >>> loss = nn.SoftmaxCrossEntropyWithLogits()
               >>> loss_scale_manager = FixedLossScaleManager()
               >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
               >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
               >>> model.build(dataset)
               >>> model.train(2, dataset)

   .. py:method:: eval(valid_dataset, callbacks=None, dataset_sink_mode=True)

      模型评估接口，其迭代过程由Python前端控制。

      配置项是PYNATIVE_MODE或CPU时，模型评价流程使用的是数据不下沉（non-sink）模式。

      .. note:: 
      
         如果dataset_sink_mode配置为True，数据将被送到处理器中。如果处理器是Ascend，数据特征将被逐一传输，每次数据传输的限制是256M。
         
         如果dataset_sink_mode配置为True，调用epoch_end方法时会执行Callback类的step_end方法。

      **参数** ：

         - **valid_dataset** (`Dataset`) – 评估模型的数据集。
         - **callbacks** (`Optional[list(Callback)]`) - 训练过程中必须被执行的回调对象或者包含回调对象的列表。默认值：None。
         - **dataset_sink_mode** (`bool`) - 决定是否以数据集下沉模式进行训练。默认值：True。
   
      **返回** ：

         Dict，返回测试模式下模型的损失值和评估值。

      **样例** :

         .. code-block::

               >>> from mindspore import Model, nn

               >>> # 如何构建数据集，请参考官方网站上关于【数据集】的章节。
               >>> dataset = create_custom_dataset()
               >>> net = Net()
               >>> loss = nn.SoftmaxCrossEntropyWithLogits()
               >>> model = Model(net, loss_fn=loss, optimizer=None, metrics={'acc'})
               >>> acc = model.eval(dataset, dataset_sink_mode=False)

   .. py:property:: eval_network

      获得该模型的评价网络。

   .. py:method:: infer_predict_layout(*predict_data)

      在自动或半自动并行模式下为预测网络生成参数布局，数据可以是单个或多个张量。

      .. note:: 同一批次数据应放在一个张量中。

      **参数** ：

         - **predict_data** (`Tensor`) – 单个或多个张量的预测数据
   
      **返回** ：

         Dict，用于加载分布式checkpoint的参数布局字典。

      **抛出异常** :

         - **RuntimeError** – 如果 *get_context* 不是图模式（GRAPH_MODE）。

      **样例** :

         .. code-block::

                  >>> # 该例子需要在多设备上运行。请参考mindpore.cn上的教程 > 分布式训练。
                  >>> import numpy as np
                  >>> import mindspore as ms
                  >>> from mindspore import Model, context, Tensor
                  >>> from mindspore.context import ParallelMode
                  >>> from mindspore.communication import init
                  >>> 
                  >>> context.set_context(mode=context.GRAPH_MODE)
                  >>> init()
                  >>> context.set_auto_parallel_context(full_batch=True, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
                  >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), ms.float32)
                  >>> model = Model(Net())
                  >>> model.infer_predict_layout(input_data)

   .. py:method:: infer_train_layout(train_dataset, dataset_sink_mode=True, sink_size=-1)

      在自动或半自动并行模式下为训练网络生成参数布局，当前只有数据下沉模式可支持使用。

      .. warning:: 这是一个实验性的原型，可能会被改变和/或删除。

      .. note:: 这是一个预编译函数。参数必须与model.train()函数相同。

      **参数** ：

         - **train_dataset** (`Dataset`) – 一个训练数据集迭代器。如果没有损失函数（ *loss_fn* ），返回一个包含多个数据的元组（data1, data2, data3, ...）并传递给网络。否则，返回一个元组（data, label），数据和标签将被分别传递给网络和损失函数。
         - **dataset_sink_mode** (`bool`) – 决定是否以数据集下沉模式进行训练。默认值：True。配置项是pynative模式或CPU时，训练模型流程使用的是数据不下沉（non-sink）模式。默认值：True。
         - **sink_size** (`int`) – 控制每次数据下沉的数据量，如果sink_size=-1，则每一次epoch下沉完整数据集。如果sink_size>0，则每一次epoch下沉数据量为sink_size的数据集。如果dataset_sink_mode为False，则设置sink_size为无效。默认值：-1。
   

      **返回** ：

         Dict，用于加载分布式checkpoint的参数布局字典。

      **样例** :

         .. code-block::

                  >>> # 该例子需要在多设备上运行。请参考mindpore.cn上的教程 > 分布式训练。
                  >>> import numpy as np
                  >>> import mindspore as ms
                  >>> from mindspore import Model, context, Tensor, nn, FixedLossScaleManager
                  >>> from mindspore.context import ParallelMode
                  >>> from mindspore.communication import init
                  >>> 
                  >>> context.set_context(mode=context.GRAPH_MODE)
                  >>> init()
                  >>> context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
                  >>> 
                  >>> # 如何构建数据集，请参考官方网站上关于【数据集】的章节。
                  >>> dataset = create_custom_dataset()
                  >>> net = Net()
                  >>> loss = nn.SoftmaxCrossEntropyWithLogits()
                  >>> loss_scale_manager = FixedLossScaleManager()
                  >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
                  >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
                  >>> layout_dict = model.infer_train_layout(dataset)

   .. py:method:: predict(*predict_data)

      输入样本得到预测结果。数据可以是单个张量，包含张量的列表，或者是包含张量的元组。

      .. note:: 这是一个预编译函数。参数应与model.predict()函数相同。

      **参数** ：

         - **predict_data** (`Tensor`) – 预测样本，可以是布尔值、数值型、浮点型、字符串、None、张量，或者存储这些类型的元组、列表和字典。


      **返回** ：

         返回预测结果，类型是张量或数组。
         
      **样例** :

         .. code-block::

                  >>> import mindspore as ms
                  >>> from mindspore import Model, Tensor
                  >>> 
                  >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), ms.float32)
                  >>> model = Model(Net())
                  >>> result = model.predict(input_data)

   .. py:property:: predict_network

      获得该模型的预测网络。

   .. py:method:: train(epoch, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=-1)

      模型训练接口，其迭代过程由Python前端控制。

      配置项是PYNATIVE_MODE或CPU时，模型训练流程使用的是数据不下沉（non-sink）模式。

      .. note:: 
      
      如果dataset_sink_mode配置为True，数据将被送到处理器中。如果处理器是Ascend，数据特征将被逐一传输，每次数据传输的限制是256M。

      如果dataset_sink_mode配置为True，调用epoch_end方法时会执行Callback类的step_end方法。
      
      如果sink_size > 0，每次epoch可以无限次遍历数据集，直到遍历数据量等于sink_size为止。然后下次epoch是从上一次遍历的最后位置继续开始遍历。该接口会构建并执行计算图，如果'model.build'已经执行过，那么它会直接执行计算图而不构建。

      **参数** ：

         - **epoch** (`int`) – 一般来说，一次epoch是完整数据集进行迭代训练的总次数。当dataset_sink_mode设置为true且sink_size>0时，则被一次epoch中数据集在sink_size遍历所需的步数所替代。
         - **train_dataset** (`Dataset`) – 一个训练数据集迭代器。如果没有损失函数，返回一个包含多个数据的元组（data1, data2, data3, ...）并传递给网络。否则，返回一个元组（data, label），数据和标签将被分别传递给网络和损失函数。
         - **callbacks** (`Optional[list[Callback], Callback]`) – 训练过程中必须被执行的回调对象或者包含回调对象的列表。默认值：None。
         - **dataset_sink_mode** (`bool`) – 决定是否以数据集下沉模式进行训练。默认值：True。配置项是pynative模式或CPU时，训练模型流程使用的是数据不下沉（non-sink）模式。默认值：True。
         - **sink_size** (`int`) – 控制每次数据下沉的数据量，如果sink_size=-1，则每一次epoch下沉完整数据集。如果sink_size>0，则每一次epoch下沉数据量为sink_size的数据集。如果dataset_sink_mode为False，则设置sink_size为无效。默认值：-1。

      **样例** :

         .. code-block::

                  >>> from mindspore import Model, nn, FixedLossScaleManager
                  >>>
                  >>> # 如何构建数据集，请参考官方网站上关于【数据集】的章节。
                  >>> dataset = create_custom_dataset()
                  >>> net = Net()
                  >>> loss = nn.SoftmaxCrossEntropyWithLogits()
                  >>> loss_scale_manager = FixedLossScaleManager()
                  >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
                  >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
                  >>> model.train(2, dataset)

   .. py:property:: train_network

      获得该模型的训练网络。






