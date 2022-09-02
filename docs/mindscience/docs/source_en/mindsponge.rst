MindSPONGE APIs
=======================

.. warning::
    These are experimental prototypes that are subject to change and/or deletion.

mindsponge.callback
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.callback.WriteH5MD
    mindsponge.callback.RunInfo

mindsponge.cell
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.cell.Attention
    mindsponge.cell.GlobalAttention
    mindsponge.cell.MSARowAttentionWithPairBias
    mindsponge.cell.MSAColumnAttention
    mindsponge.cell.MSAColumnGlobalAttention
    mindsponge.cell.TriangleAttention
    mindsponge.cell.TriangleMultiplication
    mindsponge.cell.OuterProductMean
    mindsponge.cell.InvariantPointAttention
    mindsponge.cell.Transition

mindsponge.colvar
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.colvar.Colvar
    mindsponge.colvar.Distance
    mindsponge.colvar.Angle
    mindsponge.colvar.Torsion
    mindsponge.colvar.Atom
    mindsponge.colvar.Position
    mindsponge.colvar.AtomDistances
    mindsponge.colvar.AtomAngles
    mindsponge.colvar.AtomTorsions
    mindsponge.colvar.BondedColvar
    mindsponge.colvar.BondedDistances
    mindsponge.colvar.BondedTorsions
    mindsponge.colvar.BondedAngles
    mindsponge.colvar.IndexColvar
    mindsponge.colvar.IndexVectors
    mindsponge.colvar.IndexDistances

mindsponge.common
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.common.get_pdb_info
    mindsponge.common.make_atom14_positions
    mindsponge.common.get_fasta_info
    mindsponge.common.get_aligned_seq

mindsponge.control
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.control.Controller
    mindsponge.control.Integrator
    mindsponge.control.LeapFrog
    mindsponge.control.VelocityVerlet
    mindsponge.control.Brownian
    mindsponge.control.Thermostat
    mindsponge.control.BerendsenThermostat
    mindsponge.control.Langevin
    mindsponge.control.Barostat
    mindsponge.control.BerendsenBarostat
    mindsponge.control.Constraint
    mindsponge.control.Lincs

mindsponge.core
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.core.Sponge
    mindsponge.core.SimulationCell
    mindsponge.core.RunOneStepCell
    mindsponge.core.AnalyseCell
    mindsponge.core.EnergySummation
    mindsponge.core.wrapper.EnergyWrapper
    mindsponge.core.wrapper.get_energy_wrapper
    mindsponge.core.wrapper.EnergySummation

mindsponge.data
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.data.elements
    mindsponge.data.element_dict
    mindsponge.data.element_name
    mindsponge.data.element_set
    mindsponge.data.atomic_mass
    mindsponge.data.str_to_tensor
    mindsponge.data.tensor_to_str
    mindsponge.data.get_class_parameters
    mindsponge.data.get_hyper_parameter
    mindsponge.data.get_hyper_string
    mindsponge.data.set_class_parameters
    mindsponge.data.set_hyper_parameter
    mindsponge.data.set_class_into_hyper_param
    mindsponge.data.load_checkpoint
    mindsponge.data.load_hyperparam
    mindsponge.data.load_hyper_param_into_class
    mindsponge.data.get_template
    mindsponge.data.get_template_index
    mindsponge.data.get_molecule
    mindsponge.data.ForceFieldParameters
    mindsponge.data.get_forcefield
    mindsponge.data.read_yaml
    mindsponge.data.write_yaml
    mindsponge.data.update_dict
    mindsponge.data.get_bonded_types
    mindsponge.data.get_dihedral_types
    mindsponge.data.get_improper_types

mindsponge.function
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.function.PI
    mindsponge.function.inv
    mindsponge.function.keepdim_sum
    mindsponge.function.keepdim_mean
    mindsponge.function.keepdim_prod
    mindsponge.function.keep_norm_last_dim
    mindsponge.function.norm_last_dim
    mindsponge.function.reduce_any
    mindsponge.function.reduce_all
    mindsponge.function.concat_last_dim
    mindsponge.function.concat_penulti
    mindsponge.function.pbc_box_reshape
    mindsponge.function.periodic_image
    mindsponge.function.displace_in_box
    mindsponge.function.vector_in_box
    mindsponge.function.get_vector_without_pbc
    mindsponge.function.get_vector_with_pbc
    mindsponge.function.get_vector
    mindsponge.function.gather_vectors
    mindsponge.function.gather_values
    mindsponge.function.calc_distance_without_pbc
    mindsponge.function.calc_distance_with_pbc
    mindsponge.function.calc_distance
    mindsponge.function.calc_angle_between_vectors
    mindsponge.function.calc_angle_without_pbc
    mindsponge.function.calc_angle_with_pbc
    mindsponge.function.calc_angle
    mindsponge.function.calc_torsion_for_vectors
    mindsponge.function.calc_torsion_without_pbc
    mindsponge.function.calc_torsion_with_pbc
    mindsponge.function.calc_torsion
    mindsponge.function.get_kinetic_energy
    mindsponge.function.get_integer
    mindsponge.function.get_ndarray
    mindsponge.function.GetVector
    mindsponge.function.GetDistance
    mindsponge.function.VelocityGenerator
    mindsponge.function.GetDistanceShift
    mindsponge.function.GetShiftGrad
    mindsponge.function.AVOGADRO_NUMBER
    mindsponge.function.BOLTZMANN_CONSTANT
    mindsponge.function.GAS_CONSTANT
    mindsponge.function.ELEMENTARY_CHARGE
    mindsponge.function.VACCUM_PERMITTIVITY
    mindsponge.function.COULOMB_CONSTANT
    mindsponge.function.STANDARD_ATMOSPHERE
    mindsponge.function.Length
    mindsponge.function.Energy
    mindsponge.function.get_length_ref
    mindsponge.function.get_length_unit
    mindsponge.function.get_length_unit_name
    mindsponge.function.get_energy_ref
    mindsponge.function.get_energy_unit
    mindsponge.function.get_energy_unit_name
    mindsponge.function.length_convert
    mindsponge.function.energy_convert
    mindsponge.function.Units
    mindsponge.function.global_units
    mindsponge.function.set_global_length_unit
    mindsponge.function.set_global_energy_unit
    mindsponge.function.set_global_units
    
mindsponge.metrics
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.metrics.CV
    mindsponge.metrics.BalancedMSE
    mindsponge.metrics.BinaryFocal
    mindsponge.metrics.MultiClassFocal

mindsponge.optimizer
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.optimizer.Updater
    mindsponge.optimizer.DynamicUpdater
    mindsponge.optimizer.SteepestDescent

mindsponge.partition
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.partition.FullConnectNeighbours
    mindsponge.partition.DistanceNeighbours
    mindsponge.partition.GridNeighbours
    mindsponge.partition.NeighbourList

mindsponge.potential
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.potential.PotentialCell
    mindsponge.potential.ForceFieldBase
    mindsponge.potential.ForceField
    mindsponge.potential.EnergyCell
    mindsponge.potential.NonbondEnergy
    mindsponge.potential.BondEnergy
    mindsponge.potential.AngleEnergy
    mindsponge.potential.DihedralEnergy
    mindsponge.potential.CoulombEnergy
    mindsponge.potential.LennardJonesEnergy
    mindsponge.potential.NonbondPairwiseEnergy
    mindsponge.potential.Bias
    mindsponge.potential.OscillatorBias
    mindsponge.potential.SphericalRestrict

mindsponge.system
----------------------

.. msplatformautosummary::
    :toctree: mindsponge
    :nosignatures:
    :template: classtemplate.rst

    mindsponge.system.Molecule
    mindsponge.system.Protein
    mindsponge.system.Residue
    mindsponge.system.AminoAcid
    mindsponge.system.modeling.rotate_by_axis
    mindsponge.system.modeling.add_h
    mindsponge.system.modeling.AddHydrogen
    mindsponge.system.modeling.ReadPdbByMindsponge
    mindsponge.system.modeling.gen_pdb
    mindsponge.system.modeling.read_pdb