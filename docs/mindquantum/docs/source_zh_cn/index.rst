MindQuantum文档
=========================

MindQuantum是基于昇思MindSpore开源深度学习框架和HiQ量子计算云平台开发的通用量子计算框架，支持多种量子神经网络的训练和推理。得益于华为HiQ团队的量子计算模拟器和昇思MindSpore高性能自动微分能力，MindQuantum能够高效处理量子机器学习、量子化学模拟和量子优化等问题，为广大的科研人员、老师和学生提供快速设计和验证量子机器学习算法的高效平台。

.. raw:: html

   <img src="https://gitee.com/mindspore/docs/raw/master/docs/mindquantum/docs/source_zh_cn/images/mindquantum_cn.png" width="700px" alt="" >

使用MindQuantum的典型场景
------------------------------

1. `量子机器学习 <https://www.mindspore.cn/mindquantum/docs/zh-CN/master/qnn_for_nlp.html>`_

   将量子神经网络加入训练过程，提高收敛精度。

2. `量子化学模拟 <https://www.mindspore.cn/mindquantum/docs/zh-CN/master/vqe_for_quantum_chemistry.html>`_

   使用量子变分求解器，求解分子体系基态能量。

3. `量子组合优化 <https://www.mindspore.cn/mindquantum/docs/zh-CN/master/quantum_approximate_optimization_algorithm.html>`_

   利用QAOA算法来解决最大割问题。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   mindquantum_install


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 基础使用指南

   parameterized_quantum_circuit
   initial_experience_of_quantum_neural_network
   get_gradient_of_PQC_with_mindquantum

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 变分量子算法

   classification_of_iris_by_qnn
   quantum_approximate_optimization_algorithm
   qnn_for_nlp
   vqe_for_quantum_chemistry

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 通用量子算法

   quantum_phase_estimation
   grover_search_algorithm
   shor_algorithm

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindquantum.core
   mindquantum.simulator
   mindquantum.framework
   mindquantum.algorithm
   mindquantum.io
   mindquantum.engine
   mindquantum.utils
