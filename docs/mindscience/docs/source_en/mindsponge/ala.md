# MindSPONGE Molecular Simulation Practice

Translator: [LiangRio](https://gitee.com/liangrio)

<!-- TOC -->

- [MindSPONGE Molecular Simulation Practice](#mindsponge-molecular-simulation-practice)
    - [Overview](#overview)
    - [Overall Execution](#overall-execution)
    - [Preparation](#preparation)
    - [Example of Simulated Polypeptide Aqueous Solution System](#example-of-simulated-polypeptide-aqueous-solution-system)
        - [Preparing Input Files](#preparing-input-files)
        - [Loading Data](#loading-data)
        - [Constructing Simulation Process](#constructing-simulation-process)
        - [Running Script](#running-script)
        - [Running Result](#running-result)
    - [Performance](#performance)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindscience/docs/source_zh_cn/mindsponge/ala.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

Molecular simulation is a method of exploiting computer to simulate the structure and behavior of molecules by using the molecular model at the atomic-level, and then simulate the physical and chemical properties of the molecular system. It builds a set of models and algorithms based on the experiment and through the basic principles, so as to calculate the reasonable molecular structure and molecular behavior.

In recent years, molecular simulation technology has been developed rapidly and widely used in many fields. In the field of medical design, it can be used to study the mechanism of action of virus and drugs. In the field of biological science, it can be used to characterize the multi-level structure and properties of proteins. In the field of materials science, it can be used to study the structure and mechanical properties, material optimization design. In the field of chemistry, it can be used to study surface catalysis and mechanism. In the field of petrochemical industry, it can be used for structure characterization, synthesis design, adsorption and diffusion of molecular sieve catalyst, construction and characterization of polymer chain and structure of crystalline or amorphous bulk polymer, and prediction of important properties including blending behavior, mechanical properties, diffusion, cohesion and so on.

MindSPONGE in MindSpore is molecular simulation library jointly developed by the Gao Yiqin research group of PKU and Shenzhen Bay Laboratory and Huawei MindSpore team. MindSPONGE has the features like high-performance, modularization, etc. MindSPONGE can complete the traditional molecular simulation process efficiently based on MindSpore's automatic parallelism, graph-computing fusion and other features. MindSPONGE can combine AI methods such as neural networks with traditional molecular simulations by utilizing MindSpore's feature of automatic differentiation.

This tutorial mainly introduces how to use MindSPONGE, which is built in MindSpore, to perform high performance molecular simulation on the GPU.

> Here you can download the complete sample code: <https://gitee.com/mindspore/mindscience/tree/r0.1/MindSPONGE/examples/polypeptide>.

## Overall Execution

1. Prepare input files of molecular simulation, load data, and determine the molecular system of calculation;
2. Define and initialize the MindSPONGE module, make sure the calculation process;
3. Run training script, output thermodynamic information of the simulation, and check the result.

## Preparation

Before practicing, make sure you have MindSpore installed correctly. If not, you can turn to [MindSpore Installation](https://www.mindspore.cn/install/en).

## Example of Simulated Polypeptide Aqueous Solution System

MindSPONGE has advantages of high-performance and usability, and this tutorial uses MindSPONGE to simulate polypeptide aqueous solution system. The simulated system is an alanine tripeptide aqueous solution system.

### Preparing Input Files

The simulated system of this tutorial requires 3 input files:

- Property file (file suffix`.in`), declares the basic conditions for the simulation, parameter control to the whole simulation process.
- Topology file (file suffix`.param7`), describes the topological relations and parameters of the internal molecules in the system.
- Coordinate file (file suffix`.rst7`), describes the initial coordinates of each atom in the system.

Topology and Coordinate files can be modeling completed by tleap (download address <http://ambermd.org/GetAmber.php>, comply with the GPL), which is a built-in tool in AmberTools, through the modeling process.

The modeling process is as follows:

- Open tleap

```bash
tleap
```

- Load force field ff14SB that built-in in tleap

```bash
> source leaprc.protein.ff14SB
```

- Build model of alanine tripeptide

```bash
> ala = sequence {ALA ALA ALA}
```

- Use tleap to load its force field tip3p

```bash
> source leaprc.water.tip3p
```

- Use `slovatebox` in tleap to dissolve alanine tripeptide chain, complete the system construction. `10.0`, represents the water we add is over 10 Angstrom far away from the border of molecular we dissolve and the system.

```bash
> solvatebox ala TIP3PBOX 10.0
```

- Save constructed system as file suffix `parm7` and `rst7`

```bash
> saveamberparm ala ala.parm7 ala_350_cool_290.rst7
```

After constructing the Topology file(`WATER_ALA.parm7`) and Coordinate file(`WATER_ALA_350_COOL_290.RST7`) that needed through tleap, it is required to declare basic conditions of simulation by Property file, which executes parameter control to the whole simulation process. Take Property file `NVT_299_10ns.in` in this tutorial as an example, contents are as follows:

```text
NVT 290k
   mode = 1,                              # Simulation mode ; mode=1 for NVT ensemble
   dt= 0.001,                             # Time step in picoseconds (ps). The time length of each MD step
   step_limit = 1,                        # Total step limit, number of MD steps run
   thermostat=1,                          # Thermostat for temperature ; thermostat=0 for Langevin thermostat
   langevin_gamma=1.0,                    # Gamma_ln for Langevin thermostat represents coupling strength between thermostat and system
   target_temperature=290,                # Target temperature
   write_information_interval=1000,       # Output frequency
   amber_irest=0,                         # Input style ;  amber_irest=1 for using amber style input & rst7 file contains veclocity
   cut=10.0,                              # Nonbonded cutoff distance in Angstroms
```

- `mode`, Molecular Dynamics (MD) mode, `1` represents the simulation uses `NVT` ensemble.
- `dt`, represents the step size in the simulation.
- `step_limit`, represents total steps in the simulation.
- `thermostat`, represents the method of temperature control, `1` represents to use `Liujian-Langevin`.
- `langevin_gamma`, represents `Gamma_In` parameters in the thermostat.
- `target_temperature`, represents the target temperature.
- `amber_irest`, represents the input mode, `0` represents to use the amber mode to input, and files suffix `rst7` do not include the attribute `veclocity`.
- `cut`, represents the distance of non-bonding interaction.

### Loading Data

After completing the construction of input files, save files under the path `data` to local workplace, the directory structure is as follows:

```text
└─polypeptide
    ├─data
    │      NVT_290_10ns.in                 # specific MD simulation setting
    │      WATER_ALA.parm7                 # topology file include atom & residue & bond & nonbond information
    │      WATER_ALA_350_cool_290.rst7     # restart file record atom coordinate & velocity and box information

```

Read the parameters needed by the simulation system from three input files, and use them for calculation in MindSpore. The loading code is as follows:

```python
import argparse
from mindspore import context

parser = argparse.ArgumentParser(description='Sponge Controller')
parser.add_argument('--i', type=str, default=None, help='input file')
parser.add_argument('--amber_parm', type=str, default=None, help='paramter file in AMBER type')
parser.add_argument('--c', type=str, default=None, help='initial coordinates file')
parser.add_argument('--r', type=str, default="restrt", help='')
parser.add_argument('--x', type=str, default="mdcrd", help='')
parser.add_argument('--o', type=str, default="mdout", help="")
parser.add_argument('--box', type=str, default="mdbox", help='')
parser.add_argument('--device_id', type=int, default=0, help='')
args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id, save_graphs=False)
```

### Constructing Simulation Process

By using computational force module and computational energy module defined in MindSPONGE, the system reaches the equilibrium state we need through multiple iterations of molecular dynamics process evolves, and energy and other data obtained in each simulation step is recorded. For convenience, this tutorial set `1` as the number of iterations, the code for constructing the simulation process is as follows:

```python
from mindsponge.md.simulation import Simulation
from mindspore import Tensor

if __name__ == "__main__":
    simulation = Simulation(args_opt)
    save_path = args_opt.o
    for steps in range(simulation.md_info.step_limit):
        print_step = steps % simulation.ntwx
        if steps == simulation.md_info.step_limit - 1:
            print_step = 0
        temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene, _ = simulation(Tensor(steps), Tensor(print_step))
        # compute energy and temperature

```

### Running Script

Execute the following command, start `main.py` training script for training:

```text
python main.py --i /path/NVT_290_10ns.in \
               --amber_parm /path/WATER_ALA.parm7 \
               --c /path/WATER_ALA_350_cool_290.rst7 \
               --o /path/ala_NVT_290_10ns.out

```

- `i` is property file of MD simulation, which control simulation process.
- `amber_parm` is topology file of MD simulation system.
- `c` is initial coordinate file we input.
- `o` is log file output after simulation, which records energy and other data obtained in each simulation step.
- `path` is the path to the file, this path is denoted as `data` in this tutorial.

During training, property file (file suffix`.in`), topology file (file suffix`.param7`) and coordinate file (file suffix`.rst7`) can be used under specified temperatures to perform simulation, compute force and energy, perform molecular dynamics process evolves.

### Running Result

After training, output file `ala_NVT_290_10ns.out` can be obtained, which records the change of system energy and can be viewed for thermodynamic information of the simulation system. When viewing `ala_NVT_290_10ns.out`, contents are as follows:

```text
_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ _ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_
      1 0.788   -5836.521         48.745       0.891         14.904      9.041    194.479  763.169    -6867.750
   ...
```

Types of energy output in the simulation process are recorded, namely iterations(steps), temperature(TEMP), total energy(TOT_POT_E), bond length(BOND_ENE), bond angle(ANGLE_ENE), dihedral angle interactions(DIHEDRAL_ENE), and none-bonded interaction that includes electrostatic force and Leonard-Jones interaction.

## Performance

| Parameter                 |   GPU |
| -------------------------- |---------------------------------- |
| Resource                   | GPU (Tesla V100 SXM2); memory 16 GB
| Upload date              |
| MindSpore version          | 1.2
| Training parameter        | step=1
| Output                    | numpy file
| Speed                      | 15.0 ms/step
| Total time                 | 5.7 s
| Script                    | [Link](https://gitee.com/mindspore/mindscience/tree/r0.1/MindSPONGE/examples/polypeptide/scripts)
