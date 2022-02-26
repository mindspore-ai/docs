MindQuantum Documents
======================

MindQuantum is a general-purpose quantum computing library designed to train and infer multiple quantum neural networks. Developed by MindSpore and HiQ, it leverages the quantum computing simulator developed by HiQ and high-performance automatic differentiation of MindSpore, ensuring MindQuantum can efficiently solve problems in quantum machine learning, chemistry simulation, and optimization. It provides a platform for researchers, teachers, and students to quickly design and verify quantum machine learning algorithms.

.. raw:: html

   <img src="https://gitee.com/mindspore/docs/raw/master/docs/mindquantum/docs/source_en/images/mindquantum_en.png" width="700px" alt="" >

Typical Application Scenarios
------------------------------

1. `Quantum Machine Learning <https://www.mindspore.cn/mindquantum/docs/en/master/qnn_for_nlp.html>`_

   Add the quantum neural network to the training process to improve the convergence accuracy.

2. `Quantum Chemical Simulation <https://www.mindspore.cn/mindquantum/docs/en/master/vqe_for_quantum_chemistry.html>`_

   Use VQE to solve the ground state energy of molecular system.

3. `Quantum Combinatorial Optimization <https://www.mindspore.cn/mindquantum/docs/en/master/quantum_approximate_optimization_algorithm.html>`_

   Use QAOA to solve the Max-Cut problem.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation

   mindquantum_install


.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guide

   parameterized_quantum_circuit
   get_gradient_of_PQC_with_mindquantum
   classification_of_iris_by_qnn
   quantum_approximate_optimization_algorithm
   qnn_for_nlp
   vqe_for_quantum_chemistry
   shor_algorithm

.. toctree::
   :maxdepth: 1
   :caption: API References

   mindquantum.core
   mindquantum.simulator
   mindquantum.framework
   mindquantum.algorithm
   mindquantum.io
   mindquantum.engine
   mindquantum.utils
