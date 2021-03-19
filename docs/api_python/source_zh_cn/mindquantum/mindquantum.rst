mindquantum
===========

.. automodule:: mindquantum

mindquantum.circuit
-------------------

.. automodule:: mindquantum.circuit
    :members:


mindquantum.engine
------------------

.. automodule:: mindquantum.engine
    :members:

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

mindquantum.nn
--------------

.. automodule:: mindquantum.nn
    :exclude-members:  PQC, MindQuantumLayer, Evolution
    :members:

Operators
^^^^^^^^^

.. msplatformautosummary::
    :toctree: nn
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.nn.Evolution
    mindquantum.nn.MindQuantumLayer
    mindquantum.nn.PQC

mindquantum.parameterresolver
-----------------------------

.. automodule:: mindquantum.parameterresolver
    :members:

mindquantum.utils
-----------------

.. automodule:: mindquantum.utils
    :members:
