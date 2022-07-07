mindquantum.core.circuit
========================

量子线路模块，通过有序的组织各种量子门，我们可以轻松的搭建出符合要求的量子线路，包括参数化量子线路。本模块还包含各种预设的量子线路以及对量子线路进行高效操作的模块。

Quantum Circuit
---------------

.. include:: mindquantum.core.circuit.Circuit.rst

.. include:: mindquantum.core.circuit.SwapParts.rst

.. include:: mindquantum.core.circuit.U3.rst

.. include:: mindquantum.core.circuit.UN.rst

.. include:: mindquantum.core.circuit.add_prefix.rst

.. include:: mindquantum.core.circuit.apply.rst

.. include:: mindquantum.core.circuit.change_param_name.rst

.. include:: mindquantum.core.circuit.controlled.rst

.. include:: mindquantum.core.circuit.dagger.rst

.. include:: mindquantum.core.circuit.decompose_single_term_time_evolution.rst

.. include:: mindquantum.core.circuit.pauli_word_to_circuits.rst

.. include:: mindquantum.core.circuit.shift.rst

.. automodule:: mindquantum.core.circuit
    :exclude-members: C, D, A, AP, CPN
    :members:

functional
----------

如下的操作符是对应量子线路操作符的简写。

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - high level circuit operators
   * - mindquantum.core.circuit.C
     - :class:`mindquantum.core.circuit.controlled`
   * - mindquantum.core.circuit.D
     - :class:`mindquantum.core.circuit.dagger`
   * - mindquantum.core.circuit.A
     - :class:`mindquantum.core.circuit.apply`
   * - mindquantum.core.circuit.AP
     - :class:`mindquantum.core.circuit.add_prefix`
   * - mindquantum.core.circuit.CPN
     - :class:`mindquantum.core.circuit.change_param_name`
