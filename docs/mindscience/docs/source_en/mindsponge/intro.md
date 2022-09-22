# MindSPONGE Introduction

MindSpore SPONGE(Simulation Package tOwards Next GEneration molecular modelling) is a toolkit for Computational Biology based on AI framework [MindSpore](https://www.mindspore.cn/)，which supports MD, folding and so on. It aims to provide efficient AI computational biology software for a wide range of scientific researchers, staff, teachers and students.

![MindSPONGE Architecture](images/archi.png)

## Installation

### Recommended Hardware

| Hardware      | os              | Status |
| :------------ | :-------------- | :--- |
| Ascend 910    | Ubuntu-x86      | ✔️ |
|               | Ubuntu-aarch64  | ✔️ |
|               | EulerOS-aarch64 | ✔️ |
|               | CentOS-x86      | ✔️ |
|               | CentOS-aarch64  | ✔️ |
| GPU CUDA 10.1 | Ubuntu-x86      | ✔️ |

### Dependency

- Python>=3.7
- MindSpore>=1.8

MindSpore installation tutorial please refer to [MindSpore](https://www.mindspore.cn/install)

### source code install

```bash
git clone https://gitee.com/mindspore/mindscience.git
cd mindscience/MindSPONGE
```

- dependency install

    ```bash
    pip install -r requirements.txt
    ```

- Ascend backend

    ```bash
    bash build.sh -e ascend
    ```

- GPU backend

    Enable `c` if you want to use Cybertron.

    Enable `t` if you want to use traditional MD.

    ```bash
    export CUDA_PATH={your_cuda_path}
    bash build.sh -e gpu -j32 -t on -c on
    ```

- Install whl package

    ```bash
    cd output/
    pip install mindscience_sponge*.whl
    pip install mindscience_cybertron*.whl
    ```

## Examples

### Protein Violation Computation

Although the structure predicted by structure prediciton model(e.g. AlphaFold2, MEGA-Fold) has ideal bond-length and bond-angle on most atoms, whether there is conflict between atoms and peptide bond information are also particularly important. Violation can measure the conflict well and is an import metric for protein relaxation.

The formula for violation computation as below:

\begin{align}
\mathcal L_{viol} = \mathcal L_{bondlength }+\mathcal L_{bondangle }+\mathcal L_{clash } .
\end{align}

```bash
import mindspore as ms
from mindspore import set_context
from mindspore import Tensor
from mindsponge.common.utils import get_pdb_info
from mindsponge.metrics.structure_violations import get_structural_violations

# set which gpu to use, in default use 0 card
ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", device_id=0)
input_pdb = "xxx.pdb"

# extract features from pdb
features = get_pdb_info(input_pdb)

violations = get_structural_violations(Tensor(features.get("atom14_gt_exists")).astype(ms.float32),
                                       Tensor(features.get("residue_index")).astype(ms.float32),
                                       Tensor(features.get("aatype")).astype(ms.int32),
                                       Tensor(features.get("residx_atom14_to_atom37")).astype(ms.int32),
                                       Tensor(features.get("atom14_gt_positions")).astype(ms.float32))
violation_all = violations[-1]
```

### Transfer between rotation matrix and quaternion

geometry module provides basic operations for quaternion, rotation matrix and vectors.

```bash
from mindsponge.common.geometry import initial_affine
from mindsponge.common.geometry import quat_to_rot, rot_to_quat
# quaternion is a mindspore tensor
# rotation_matrix is a tuple of mindspore tensor, length is 9
# translation is a tuple of mindsproe tensor, length is 3
quat, rot, trans = initial_affine(128) # 128 is the num of residues
transformed_rot = quat_to_rot(quat)
transformed_quat = rot_to_quat(rot)
```

### A simple example for molecular dynamics

```bash
import numpy as np
import mindspore as ms
from mindspore import set_context
from mindsponge import Sponge
from mindsponge import Molecule
from mindsponge import ForceFieldBase
from mindsponge import DynamicUpdater
from mindsponge.potential import BondEnergy, AngleEnergy
from mindsponge.callback import WriteH5MD, RunInfo
from mindsponge.function import VelocityGenerator
from mindsponge.control import LeapFrog

set_context(mode=ms.GRAPH_MODE, device_target="GPU")

system = Molecule(
    atoms=['O', 'H', 'H'],
    coordinate=[[0, 0, 0], [0.1, 0, 0], [-0.0333, 0.0943, 0]],
    bond=[[[0, 1], [0, 2]]],
)

bond_energy = BondEnergy(
    index=system.bond,
    force_constant=[[345000, 345000]],
    bond_length=[[0.1, 0.1]],
)

angle_energy = AngleEnergy(
    index=[[1, 0, 2]],
    force_constant=[[383]],
    bond_angle=[[109.47 / 180 * np.pi]],
)

energy = ForceFieldBase(energy=[bond_energy, angle_energy])

velocity_generator = VelocityGenerator(300)
velocity = velocity_generator(system.coordinate.shape, system.atom_mass)

opt = DynamicUpdater(
    system,
    integrator=LeapFrog(system),
    time_step=1e-3,
    velocity=velocity,
)

md = Sponge(system, energy, opt)

run_info = RunInfo(10)
cb_h5md = WriteH5MD(system, 'test.h5md', save_freq=10, write_velocity=True, write_force=True)

md.run(1000, callbacks=[run_info, cb_h5md])
```

## SIG

### CO-CHAIR

- Shenzhen Bay Laboratory [Yi Isaac Yang](https://gitee.com/helloyesterday)
- Chang Ping Laboratory [Jun Zhang](https://gitee.com/jz_90)
- Chang Ping Laboratory [Sirui Liu](https://gitee.com/sirui63)

### Special Interesting Group 🏠

MindSpore SPONGE SIG (Special Interesting Group) is a team composed of a group of people who are interested and have a mission to make achievements in the field of AI × biological computing.

MindSpore SPONGE SIG group provides efficient and easy-to-use AI computational biology software for researchers, teachers and students, and provides a platform for people with strong abilities or strong interests in this field to communicate and cooperate together.

At present, the SIG group has six core teachers. After members joining the SIG group, our teachers will lead the team to carry out scientific research and develop the software function development. Of course, members are also welcome to do research on their own topics using MindSPONGE.

In the SIG group, we will hold various activities, including summer school, public lecture, technology communication meeting and other large-scale activities. Small-scale activities like weekly meetings, blogs writing will also be held in the group. By joining the activities, there will be lots of chances to communicate with our experts. During the summer school program ended on August 15th, we invited 13 teachers to have a five-day lecture mainly including three themes of MindSpore basics, molecular dynamics and advanced AI × Science courses. You can get the replay [here](https://www.bilibili.com/video/BV1pB4y167yS?spm_id_from=333.999.0.0&vd_source=94e532d8ff646603295d235e65ef1453).

In the SIG group, we will also release the public intelligence task and [open source internship task](https://gitee.com/mindspore/community/issues/I561LI?from=project-issue), welcome everyone to claim it.

### Core Contributor 🧑‍🤝‍🧑

- [Yi Qin Gao Research Group](https://www.chem.pku.edu.cn/gaoyq/):  [Yi Isaac Yang](https://gitee.com/helloyesterday)，[Jun Zhang](https://gitee.com/jz_90)，[Sirui Liu](https://gitee.com/sirui63)，[Yijie Xia](https://gitee.com/xiayijie)，[Diqing Chen](https://gitee.com/dechin)，[Yu-Peng Huang](https://gitee.com/gao_hyp_xyj_admin)