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
   * - mindspore.gate.CNOT
     - :class:`mindspore.gate.CNOTGate`
   * - mindspore.gate.I
     - :class:`mindspore.gate.IGate`
   * - mindspore.gate.H
     - :class:`mindspore.gate.HGate`
   * - mindspore.gate.S
     - :class:`mindspore.gate.PhaseShift` (numpy.pi/2)
   * - mindspore.gate.SWAP
     - :class:`mindspore.gate.SWAPGate`
   * - mindspore.gate.X
     - :class:`mindspore.gate.XGate`   
   * - mindspore.gate.Y
     - :class:`mindspore.gate.YGate`
   * - mindspore.gate.Z
     - :class:`mindspore.gate.ZGate`

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
