mindquantum
===========

.. automodule:: mindquantum

mindquantum.gate
----------------

.. automodule:: mindquantum.gate
    :members:

functional
----------

The functional gates are the pre-instantiated quantum gates, which can be used directly as an instance of quantum gate.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - gates
   * - mindquantum.gate.CNOT
     - :class:`mindquantum.gate.CNOTGate`
   * - mindquantum.gate.I
     - :class:`mindquantum.gate.IGate`
   * - mindquantum.gate.H
     - :class:`mindquantum.gate.HGate`
   * - mindquantum.gate.S
     - :class:`mindquantum.gate.PhaseShift` (numpy.pi/2)
   * - mindquantum.gate.SWAP
     - :class:`mindquantum.gate.SWAPGate`
   * - mindquantum.gate.X
     - :class:`mindquantum.gate.XGate`   
   * - mindquantum.gate.Y
     - :class:`mindquantum.gate.YGate`
   * - mindquantum.gate.Z
     - :class:`mindquantum.gate.ZGate`

mindquantum.circuit
-------------------

.. automodule:: mindquantum.circuit
    :exclude-members: C, D, A, AP, CPN
    :members:

functional
----------

The functional operators are shortcut of some pre-instantiated quantum circuit operators.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - high level operators
   * - mindquantum.circuit.C
     - :class:`mindquantum.circuit.controlled`
   * - mindquantum.circuit.D
     - :class:`mindquantum.circuit.dagger`
   * - mindquantum.circuit.A
     - :class:`mindquantum.circuit.apply`
   * - mindquantum.circuit.AP
     - :class:`mindquantum.circuit.add_prefix`
   * - mindquantum.circuit.CPN
     - :class:`mindquantum.circuit.change_param_name`

mindquantum.engine
------------------

.. automodule:: mindquantum.engine
    :members:
