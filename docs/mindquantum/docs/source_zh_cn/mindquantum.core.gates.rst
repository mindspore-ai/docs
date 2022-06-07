mindquantum.core.gates
======================

量子门模块，提供不同的量子门。

Quantum Gates
-------------

.. include:: mindquantum.core.gates.BasicGate.rst

.. include:: mindquantum.core.gates.BitFlipChannel.rst

.. include:: mindquantum.core.gates.BitPhaseFlipChannel.rst

.. include:: mindquantum.core.gates.CNOTGate.rst

.. include:: mindquantum.core.gates.DepolarizingChannel.rst

.. include:: mindquantum.core.gates.GlobalPhase.rst

.. include:: mindquantum.core.gates.HGate.rst

.. include:: mindquantum.core.gates.IGate.rst

.. include:: mindquantum.core.gates.ISWAPGate.rst

.. include:: mindquantum.core.gates.Measure.rst

.. include:: mindquantum.core.gates.MeasureResult.rst

.. include:: mindquantum.core.gates.ParameterGate.rst

.. include:: mindquantum.core.gates.PauliChannel.rst

.. include:: mindquantum.core.gates.PhaseFlipChannel.rst

.. include:: mindquantum.core.gates.PhaseShift.rst

.. include:: mindquantum.core.gates.Power.rst

.. include:: mindquantum.core.gates.RX.rst

.. include:: mindquantum.core.gates.RY.rst

.. include:: mindquantum.core.gates.RZ.rst

.. include:: mindquantum.core.gates.SGate.rst

.. include:: mindquantum.core.gates.SWAPGate.rst

.. include:: mindquantum.core.gates.TGate.rst

.. include:: mindquantum.core.gates.UnivMathGate.rst

.. include:: mindquantum.core.gates.XGate.rst

.. include:: mindquantum.core.gates.XX.rst

.. include:: mindquantum.core.gates.YGate.rst

.. include:: mindquantum.core.gates.YY.rst

.. include:: mindquantum.core.gates.ZGate.rst

.. include:: mindquantum.core.gates.ZZ.rst

.. include:: mindquantum.core.gates.gene_univ_parameterized_gate.rst

.. automodule:: mindquantum.core.gates
    :members:

functional
----------

The functional gates are the pre-instantiated quantum gates, which can be used directly as an instance of quantum gate.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - gates
   * - mindquantum.core.gates.CNOT
     - :class:`mindquantum.core.gates.CNOTGate`
   * - mindquantum.core.gates.I
     - :class:`mindquantum.core.gates.IGate`
   * - mindquantum.core.gates.H
     - :class:`mindquantum.core.gates.HGate`
   * - mindquantum.core.gates.S
     - :class:`mindquantum.core.gates.PhaseShift` (numpy.pi/2)
   * - mindquantum.core.gates.SWAP
     - :class:`mindquantum.core.gates.SWAPGate`
   * - mindquantum.core.gates.X
     - :class:`mindquantum.core.gates.XGate`   
   * - mindquantum.core.gates.Y
     - :class:`mindquantum.core.gates.YGate`
   * - mindquantum.core.gates.Z
     - :class:`mindquantum.core.gates.ZGate`
