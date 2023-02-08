MindSpore Graph Learning文档
==============================

MindSpore Graph Learning是一款图学习套件，支持以点为中心编程实现图神经网络和高效的训练推理。

得益于MindSpore的图算融合能力，MindSpore Graph Learning能够针对图模型特有的执行模式进行编译优化，帮助开发者缩短训练时间。MindSpore Graph Learning还创新性地提出以点为中心编程范式，提供更原生的图神经网络表达方式，并内置覆盖大部分应用场景的模型，使开发者能够轻松搭建图神经网络。

.. image:: ./images/graphlearning_cn.png
  :width: 700px

设计特点
---------

1. 以点为中心的编程范式

   图神经网络模型需要在给定的图结构上做信息的传递和聚合，整图计算无法直观表达这些操作。MindSpore Graph Learning提供以点为中心的编程范式，更符合图学习算法逻辑和Python语言风格，可以直接进行公式到代码的翻译，减少算法设计和实现间的差距。

2. 高效加速图模型

   MindSpore Graph Learning结合MindSpore的图算融合和自动算子编译技术（AKG）特性，自动识别图神经网络任务中特有的执行pattern进行融合和kernel level优化，能够覆盖现有框架中已有的算子和新组合算子的融合优化，获得相比现有流行框架平均3到4倍的性能提升。

训练流程
---------

MindSpore Graph Learning为用户提供了丰富的数据读入、图操作和网络结构模块接口，用户使用MindSpore Graph Learning实现训练图神经网络只需要以下几步：

1. 定义网络结构，用户可以直接调用mindspore_gl.nn提供的接口，也可以参考这里的实现自定义图学习模块。

2. 定义loss函数。

3. 构造数据集，mindspore_gl.dataset提供了一些研究用的公开数据集的读入和构造。

4. 网络训练和验证。

特性介绍
---------

MindSpore Graph Learning提供了以点为中心的GNN网络编程范式，内置将以点为中心的计算表达翻译为图数据的计算操作的代码解析函数，为了方便用户调试解析过程将打印出用户输入代码与计算代码的翻译对比图。

如下图基于以点为中心编程模型实现经典GAT网络。用户定义一个函数以节点`v`作为入参，在函数内用户通过`v.innbs()`获取邻居节点列表，遍历每个邻居节点`u`，获取节点特征，计算邻居节点与中心节点的特征交互得到邻边上的权重列表，然后将邻边权重与邻居节点进行加权平均，返回更新的中心节点特征。

.. image:: ./images/gat_example.PNG
  :width: 700px

未来路标
---------

MindSpore Graph Learning初始版本包含以点为中心的编程范式，并内置提供了典型图模型的实现，以及在小数据集上单机训练的案例和性能评测。初始版本暂时不包含大数据集上的性能评测和分布式训练，也不支持对接图数据库。MindSpore Graph Learning后续版本将包含这些内容，敬请期待。

使用MindSpore Graph Learning的典型场景
---------------------------------------

.. toctree::
   :maxdepth: 1
   :caption: 安装部署

   mindspore_graphlearning_install

.. toctree::
   :maxdepth: 1
   :caption: 使用指南

   full_training_of_GCN
   batched_graph_training_GIN
   spatio_temporal_graph_training_STGCN
   single_host_distributed_Graphsage

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindspore_gl.dataloader
   mindspore_gl.dataset
   mindspore_gl.graph
   mindspore_gl.nn
   mindspore_gl.sampling
   mindspore_gl.utils

.. toctree::
   :maxdepth: 1
   :caption: 参考文档

   faq

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
