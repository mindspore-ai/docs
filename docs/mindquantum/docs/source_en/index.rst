MindSpore Quantum Documents
=============================

MindSpore Quantum is a new-generation quantum computing suite based on MindSpore, supporting training and inference of multiple quantum neural networks.

MindSpore Quantum focuses on the implementation of NISQ algorithms. It combines the HiQ high-performance quantum computing simulator with the parallel automatic differentiation capability of MindSpore. MindSpore Quantum is easy-to-use with ultra-high performance. It can efficiently handle problems like quantum machine learning, quantum chemistry simulation, and quantum optimization. MindSpore Quantum provides an efficient platform for researchers, teachers and students to quickly design and verify quantum algorithms, making quantum computing at your fingertips.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindquantum/docs/source_en/images/mindquantum_en.png" width="700px" alt="" >

Code repository address: <https://gitee.com/mindspore/mindquantum>

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
   :caption: Variational Quantum Algorithm

   classification_of_iris_by_qnn
   quantum_approximate_optimization_algorithm
   qnn_for_nlp
   vqe_for_quantum_chemistry
   equivalence_checking_of_PQC

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: General Quantum Algorithm

   quantum_phase_estimation
   grover_search_algorithm
   shor_algorithm

.. toctree::
   :maxdepth: 1
   :caption: API References

   overview
   mindquantum.core
   mindquantum.simulator
   mindquantum.framework
   mindquantum.algorithm
   mindquantum.device
   mindquantum.io
   mindquantum.engine
   mindquantum.utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
