MindQuantum文档
=========================

MindQuantum是基于MindSpore的新一代量子计算领域套件，支持多种量子神经网络的训练和推理。

MindQuantum聚焦于NISQ阶段的算法实现与落地。结合HiQ高性能量子计算模拟器和昇思MindSpore并行自动微分能力，MindQuantum有着极简的开发模式和极致的性能体验，能够高效处理量子机器学习、量子化学模拟和量子组合优化等问题，为广大科研人员、老师和学生提供快速设计和验证量子算法的高效平台，让量子计算触手可及。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindquantum/docs/source_zh_cn/images/mindquantum_cn.png" width="700px" alt="" >

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
   quantum_simulator
   initial_experience_of_quantum_neural_network
   get_gradient_of_PQC_with_mindquantum
   advanced_operations_of_quantum_circuit
   quantum_measurement
   noise
   bloch_sphere

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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE