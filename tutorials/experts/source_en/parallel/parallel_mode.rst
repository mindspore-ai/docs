Parallel Mode
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/parallel/parallel_mode.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  data_parallel
  semi_auto_parallel
  auto_parallel
  manual_parallel
  parameter_server_training

----------------------

Depending on different usages, MindSpore can take several parallel modes as described below:

- `Data parallel <https://www.mindspore.cn/tutorials/experts/en/master/parallel/data_parallel.html>`_: In the data parallel mode, the model is replicated to multiple compute nodes, each of which uses a different data subset to train the model. Each node computes the gradient independently and passes it to other nodes for model parameter updates. Data parallel is suitable for situations where a large amount of data is available and can speed up training.
- `Semi-Automatic Parallel Mode <https://www.mindspore.cn/tutorials/experts/en/master/parallel/semi_auto_parallel.html>`_: Semi-automatic parallel is a mode between automatic parallel and manual parallel. In semi-automatic parallel mode, the users need to specify some details of parallel training, such as the data slicing strategy for certain operators or the slicing strategy for certain parameters, and the framework manages the allocation and synchronization of parallel tasks according to the configured strategy.
- `Automatic Parallel Mode <https://www.mindspore.cn/tutorials/experts/en/master/parallel/auto_parallel.html>`_: Automatic parallel means that the framework automatically handles distributed training without manual user intervention. In the auto-parallel mode, the framework finds a parallel strategy with shorter time through a specific algorithm, automatically assigns the model and data to different computing nodes according to the strategy, and automatically handles the passing of gradients and the synchronization of parameters. In this way, the users only need to define the model and data, and the framework will automatically complete the parallel training process.
- `Manual Parallel Mode <https://www.mindspore.cn/tutorials/experts/en/master/parallel/manual_parallel.html>`_: Manual parallel is the most flexible but also the most complex distributed training mode. In manual parallel mode, users need to write their own code to control the distribution of models and data, as well as the passing of gradients and synchronization of parameters. This mode is suitable for advanced users and allows for fine-grained control based on specific needs.
- `Parameter Server <https://www.mindspore.cn/tutorials/experts/en/master/parallel/parameter_server_training.html>`_: The parameter server mode is a special distributed training model in which computational nodes are divided into two categories: parameter servers and worker nodes. The parameter server maintains the parameters of the model, while the worker nodes are responsible for computing the gradient. After calculating the gradient, the worker nodes send it to the parameter server, which updates the model parameters and broadcasts the updated parameters to all worker nodes. This model is more suitable when the model parameters are large, but requires more communication overhead.
