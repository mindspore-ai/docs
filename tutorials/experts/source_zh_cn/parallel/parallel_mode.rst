并行模式
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/parallel_mode.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  data_parallel
  semi_auto_parallel
  auto_parallel
  manual_parallel
  parameter_server_training

----------------------

根据使用方式的不同，MindSpore可以采取下述的几种并行模式：

- `数据并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/data_parallel.html>`_：数据并行模式下，模型被复制到多个计算节点上，每个节点使用不同的数据子集来训练模型。每个节点独立计算梯度，并将其传递给其他节点以进行模型参数更新。数据并行适用于拥有大量数据的情况，可以加快训练速度。
- `半自动并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/semi_auto_parallel.html>`_：半自动并行是介于自动并行和手动并行之间的模式。在半自动并行模式下，用户需要指定一些并行训练的细节，如某些算子的数据切分策略或者某些参数的切分策略，框架会根据配置的策略管理并行任务的分配和同步。
- `自动并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/auto_parallel.html>`_：自动并行是指框架自动处理分布式训练，无需用户手动干预。在自动并行模式下，框架通过特定的算法找到时间较短的并行策略，根据策略将模型和数据自动分配到不同的计算节点，并自动处理梯度的传递和参数的同步。这样，用户只需要定义模型和数据，框架会自动完成并行训练过程。
- `手动并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/manual_parallel.html>`_：手动并行是最灵活但也最复杂的分布式训练模式。在手动并行模式下，用户需要自己编写代码来控制模型和数据的分布，以及梯度的传递和参数的同步。这种模式适用于高级用户，可以根据特定需求进行精细的控制。
- `参数服务器 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/parameter_server_training.html>`_：参数服务器模式是一种特殊的分布式训练模式，在这种模式下，计算节点被分为两类：参数服务器和工作节点。参数服务器维护模型的参数，而工作节点负责计算梯度。工作节点计算梯度后，将其发送给参数服务器，参数服务器更新模型参数，并将更新后的参数广播给所有工作节点。这种模式在模型参数较大时比较适用，但需要更多的通信开销。
