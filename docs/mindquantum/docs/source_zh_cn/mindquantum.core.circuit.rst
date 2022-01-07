mindquantum.core.circuit
========================

Quantum Circuit
---------------

.. automodule:: mindquantum.core.circuit
    :exclude-members: C, D, A, AP, CPN
    :members:

functional
----------

The functional operators are shortcut of some pre-instantiated quantum circuit operators.

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
