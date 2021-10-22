# 模拟丙氨酸三肽水溶液体系

<!-- TOC -->

- [模拟丙氨酸三肽水溶液体系](#模拟丙氨酸三肽水溶液体系)
    - [概述](#概述)
    - [整体执行](#整体执行)
    - [准备环节](#准备环节)
    - [模拟多肽水溶液体系示例](#模拟多肽水溶液体系示例)
        - [准备输入文件](#准备输入文件)
        - [加载数据](#加载数据)
        - [构建模拟流程](#构建模拟流程)
        - [运行脚本](#运行脚本)
        - [运行结果](#运行结果)
    - [性能描述](#性能描述)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindscience/docs/source_zh_cn/mindsponge/ala.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## 概述

本篇教程将主要介绍如何在GPU上，使用MindSPONGE进行丙氨酸三肽水溶液体系模拟。

## 整体执行

1. 准备分子模拟输入文件，加载数据，确定计算的分子体系；
2. 定义 SPONGE 模块并初始化，确定计算流程；
3. 运行训练脚本，输出模拟的热力学信息文件，并查看结果；

## 准备环节

实践前，确保已经正确安装MindSpore。如果没有，可以通过[MindSpore安装页面](https://www.mindspore.cn/install)安装MindSpore。

教程中的体系结构文件建模由AmberTools中自带的tleap工具（下载地址<http://ambermd.org/GetAmber.php>， 遵守GPL协议）完成。

## 模拟多肽水溶液体系示例

SPONGE具有高性能及易用的优势，本教程使用SPONGE模拟多肽水溶液体系。模拟体系为丙氨酸三肽水溶液体系。

### 准备输入文件

本教程模拟体系中需要加载三个输入文件，分别是：

- 属性文件（后缀为`.in`的文件），声明模拟的基本条件，对整个模拟过程进行参数控制。
- 拓扑文件（后缀为`.param7`的文件），拓扑文件描述的是体系内部分子的拓扑关系及各种参数。
- 坐标文件（后缀为`.rst7`的文件），坐标文件描述的是每个原子在体系中的初始时刻的坐标及速度。

拓扑文件和坐标文件可以通过建模过程由AmberTools中自带的tleap工具（下载地址<http://ambermd.org/GetAmber.php>， 遵守GPL协议）建模完成。建模过程如下：

- 打开tleap

    ```bash
    tleap
    ```

- 加载tleap自带的ff14SB力场

    ```bash
    > source leaprc.protein.ff14SB
    ```

- 搭建丙氨酸三肽模型

    ```bash
    > ala = sequence {ALA ALA ALA}
    ```

- 利用tleap加载其自带的tip3p力场

    ```bash
    > source leaprc.water.tip3p
    ```

- 利用tleap中的`slovatebox`溶解丙氨酸三肽链， 完成体系构建。`10.0`代表加入的水距离我们溶解的分子及体系边界至少在`10.0`埃以上

    ```bash
    > solvatebox ala TIP3PBOX 10.0
    ```

- 将建好的体系保存成`parm7`及`rst7`文件

    ```bash
    > saveamberparm ala WATER_ALA.parm7 WATER_ALA_350_cool_290.rst7
    ```

通过tleap构建了所需要的拓扑文件（`WATER_ALA.parm7`）和坐标文件（`WATER_ALA_350_cool_290.rst7`）后，需要通过属性文件声明模拟的基本条件，对整个模拟过程进行参数控制。以本教程中的属性文件`NVT_290_10ns.in`为例，其文件内容如下：

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

- `mode`，分子动力学（MD）模式，`1`表示模拟采用`NVT`系综。
- `dt`，表示模拟步长。
- `step_limit`，表示模拟总步数。
- `thermostat`，表示控温方法，`1`表示采用的是`Liujian-Langevin`方法。
- `langevin_gamma`，表示控温器中的`Gamma_ln`参数。
- `target_temperature`，表示目标温度。
- `amber_irest`，表示输入方式，`0`表示使用amber方式输入，`rst7`文件中不包含`veclocity`属性。
- `cut`，表示非键相互作用的距离。

### 加载数据

完成输入文件的构建后，将文件存放在本地工作区的`data`路径下，其目录结构如下：

```text
└─data
    ├─polypeptide
    │      NVT_290_10ns.in                 # specific MD simulation setting
    │      WATER_ALA.parm7                 # topology file include atom & residue & bond & nonbond information
    │      WATER_ALA_350_cool_290.rst7     # restart file record atom coordinate & velocity and box information
```

从三个输入文件中，读取模拟体系需要的参数，用于MindSpore的计算。加载代码如下：

```python
import argparse
import time
from mindspore import context

parser = argparse.ArgumentParser(description='Sponge Controller')
parser.add_argument('--i', type=str, default=None, help='Input .in file')
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

### 构建模拟流程

使用SPONGE中定义的计算力模块和计算能量模块，通过多次迭代进行分子动力学过程演化，使得体系达到我们所需要的平衡态，并记录每一个模拟步骤中得到的能量等数据。为了方便起见，本教程的计算迭代次数设置为`1`，其模拟流程构建代码如下：

```python
from mindsponge.md.simulation import Simulation
from mindspore import Tensor

if __name__ == "__main__":
    simulation = Simulation(args_opt)

    start = time.time()
    compiler = args_opt.o
    save_path = args_opt.o
    simulation.main_initial()
    for steps in range(simulation.md_info.step_limit):
        print_step = steps % simulation.ntwx
        if steps == simulation.md_info.step_limit - 1:
            print_step = 0
        temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, LJ_energy_sum, ee_ene, _ = simulation(Tensor(steps), Tensor(print_step))
        # compute energy and temperature
```

### 运行脚本

执行以下命令，启动训练脚本`main.py`进行训练：

```text
python main.py --i /path/NVT_290_10ns.in \
               --amber_parm /path/WATER_ALA.parm7 \
               --c /path/WATER_ALA_350_cool_290.rst7 \
               --o /path/ala_NVT_290_10ns.out
```

- -`i` 为MD模拟的属性文件，控制模拟过程
- -`amber_parm` 为MD模拟体系的拓扑文件
- -`c` 为我们输入的初始坐标文件
- -`o` 为我们模拟输出的记录文件，其记录了输出每步的能量等信息
- -`path` 为文件所在的路径，在本教程中为`data/polypeptide`

训练过程中，使用属性文件（后缀为`.in`的文件）、拓扑文件（后缀为`.param7`的文件）以及坐标文件（后缀为`.rst7`的文件），通过在指定温度下进行模拟，计算力和能量，进行分子动力学过程演化。

### 运行结果

训练结束后，可以得到输出文件`ala_NVT_290_10ns.out`，体系能量变化被记录在了该文件中，可以查看模拟体系的热力学信息。查看`ala_NVT_290_10ns.out`可以看到如下内容：

```text
_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ _ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_
      1 0.788   -5836.521         48.745       0.891         14.904      9.041    194.479  763.169    -6867.750
   ...
```

其中记录了模拟过程中输出的各类能量， 分别是迭代次数（_steps_），温度（_TEMP_），总能量（_TOT_POT_E_），键长（_BOND_ENE_），键角（_ANGLE_ENE_），二面角相互作用（_DIHEDRAL_ENE_），非键相互作用，其包含静电力及Leonard-Jones相互作用。

## 性能描述

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
