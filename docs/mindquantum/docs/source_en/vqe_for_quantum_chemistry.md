# VQE Application in Quantum Chemistry Computing

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindquantum/docs/source_en/vqe_for_quantum_chemistry.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Quantum chemistry refers to solving the numerical values of the time-dependent or time-independent Schrödinger equations by using the basic theory and method of quantum mechanics. Quantum chemical simulation on high-performance computers has become an important method to study the physical and chemical properties of materials. However, the exact solution of the Schrödinger equation has exponential complexity, which severely constrains the scale of the chemical system that can be simulated. The development of quantum computing in recent years provides a feasible way to solve this problem. It is expected that the Schrödinger equation can be solved with high accuracy on quantum computers under the complexity of polynomials.

[Peruzzo et al.](https://doi.org/10.1038/ncomms5213) first applied the VQE and [unitary coupled-cluster theory](https://linkinghub.elsevier.com/retrieve/pii/S0009261489873725) to quantum chemistry simulation in 2014 to solve the ground state energy of He-H<sup>+</sup>. The VQE is a hybrid quantum-classical algorithm and is widely used in chemical simulation based on quantum algorithms. This tutorial describes how to use the VQE to solve the ground-state energy of a molecular system.

This tutorial consists of the following parts:

1. Introduction to the quantum chemistry
2. VQE application
3. Using MindQuantum to perform VQE simulation with efficient and automatic derivation

> This document applies to the CPU environment.
> You can obtain the complete executable sample code at <https://gitee.com/mindspore/mindquantum/blob/master/tutorials/source/7.vqe_for_quantum_chemistry.py>.

## Environment Preparation

In this tutorial, the following environments need to be installed:

- NumPy
- SciPy
- [MindQuantum](https://gitee.com/mindspore/mindquantum)
- [MindSpore](https://gitee.com/mindspore/mindspore)
- PySCF
- OpenFermion
- OpenFermion-PySCF

> The preceding dependencies can be installed by running the `pip` command.

## Importing Dependencies

Import the modules on which this tutorial depends.

```python
import numpy as np
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import mindquantum as mq
from mindquantum import Hamiltonian
from mindquantum.gate import X, RX
from mindquantum.circuit import Circuit, generate_uccsd
from mindquantum.nn import generate_pqc_operator
import mindspore as ms
import mindspore.context as context
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
```

## Quantum Chemistry Computing Method

The core of quantum chemistry is to solve the Schrödinger equation. In general, the solution of time-dependent Schrödinger equation is more complex, so Born-Oppenheimer approximation (BO approximation) is introduced. In BO approximation, the mass of the nucleus is far greater than that of electrons, and the velocity of the nucleus is far lower than that of electrons. Therefore, the nucleus and electrons can be separated from each other, and the time-independent electron motion equation (also called the time-independent Schrödinger equation) can be obtained as follows:

$$
\hat{H} |\Psi\rangle = E |\Psi\rangle
$$

$\hat{H}$ contains the following three items:

$$
\hat{H} = \hat{K} _{e} + \hat{V} _{ee} + \hat{V} _{Ne}
$$

They are electron kinetic energy, electron-electron potential energy and electron-nuclear potential energy.

There are many numerical algorithms that can be used to solve the time-independent Schrödinger equation. This tutorial introduces one of these methods: the wave function. Wave function directly solves the eigenfunction and eigenenergy of a given molecular Hamiltonian. At present, there are a large number of open-source software packages, such as [PySCF](http://pyscf.org/), which can be implemented. Here is a simple example: lithium hydride molecules, using the OpenFermion and OpenFermion-PySCF plug-ins. First, define the molecular structure:

```python
dist = 1.5
geometry = [
    ["Li", [0.0, 0.0, 0.0 * dist]],
    ["H",  [0.0, 0.0, 1.0 * dist]],
]
basis = "sto3g"
spin = 0
print("Geometry: \n", geometry)
```

```bash
Geometry:
    [['Li', [0.0, 0.0, 0.0]], ['H', [0.0, 0.0, 1.5]]]
```

The code above defines a Li-H key with a length of 1.5Å molecules. The STO-3G basis set is used for computing. Then, OpenFermion-PySCF is used to call PySCF to perform Hartree-Fock (HF), coupled-cluster with singles and doubles (CCSD), and full configuration interaction (FCI) computing. These three methods belong to the wave function. Before starting the computing, first make a brief introduction to these methods.

### Wave Function

One of the methods to solve the time-independent Schrödinger equation is the [Hartree-Fock (HF)](https://doi.org/10.1098/rspa.1935.0085) method, which was proposed by Hartree et al. in the 1930s and is the basic method in quantum chemistry computing. The HF method introduces a single determinant approximation, that is, a wave function of the $N$-electronic system is represented by a wave function in a determinant form:

$$
| \Psi \rangle = | \psi_{1} \psi_{2} \psi_{3} \dots \psi_{N} \rangle
$$

Where $| \psi_{1} \psi_{2} \psi_{3} \dots \rangle$ represents the Nth-order determinants formed by a set of spin-orbit wave functions $\{ \pi_{i} \}$.
The spin-orbit wave function $\psi_{i}$ may be further expanded with a set of basis functions in known forms:

$$\psi_{i} = \phi_{i} \eta_{i}$$
$$\phi_{i} = \sum_{\mu}{C_{\mu i} \chi_{\mu}}$$

$\{\chi_{\mu}\}$ is referred to as a basis function, and may be a Gaussian function or the like.
This approximation considers the exchange between electrons, but neglects the correlation between electrons, so it cannot correctly compute the properties such as dissociation energy.

The improvement of the HF method can be derived from the wave function expansion theorem. The wave function expansion theorem can be expressed as follows: if $\{ \psi_{i} \}$ is a complete set of spin-orbit wave functions, the $N$-electronic system wave function may be accurately expanded by a determinant wave function formed by $\{ \psi_{i} \}$:

$$
| \Psi \rangle = \sum^{\infty} _ {i_{1} < i_{2} < \dots < i_{N}} {C_{i_{1} i_{2} \dots i_{N}} | \psi_{i_{1}} \psi_{i_{2}} \dots \psi_{i_{N}} \rangle}
$$

You can obtain the configuration interaction (CI) method:

$$
| \Psi_{CI} \rangle = C_{0} | \Psi_{HF} \rangle + \sum^{a\rightarrow\infty} _{i\in occ\\\\a\not\in occ}{C^{a} _{i} | \Psi^{a} _{i} \rangle } + \sum^{ab\rightarrow\infty} _{ij\in occ\\\\ab\not\in occ}{C^{ab} _{ij} | \Psi^{ab} _{ij} \rangle }
$$

$| \Psi^{a}_{i} \rangle + \dots$ in the preceding formula represents a single excitation wave function of the electron from the orbit $i$ to the orbit $a$, and so on. A CI that considers only single excitation and double excitation is called a configuration interaction with singles and doubles (CISD). The CI that takes into account all the ground-state HF wave functions to N excitation wave functions is called full configuration interaction (FCI). The FCI wave function is the exact solution of the time-independent Schrödinger equation under the given basis function.

### Second Quantization

Under the second quantization expression, the Hamiltonian of the system has the following form:

$$
\hat{H} = \sum_{p, q}{h^{p} _ {q} E^{p} _ {q}} + \sum_{p, q, r, s}{\frac{1}{2} g^{pq} _ {rs} E^{pq} _ {rs} }
$$

$E^{p} _{q}$ and $E^{pq}_ {rs}$ are as follows:

$$
E^{p} _ {q} = a^{\dagger} _ {p} a_{q}
$$
$$
E^{pq} _ {rs} = a^{\dagger} _ {p} a^{\dagger} _ {q} a_{r} a_{s}
$$

$a^{\dagger} _{p}$ and $a_ {q}$ are creation operator and annihilation operator, respectively.

The excited-state wave function can be expressed conveniently by using a second quantization expression method:

$$
| \Psi^{abc\dots} _ {ijk\dots} \rangle = a^{\dagger} _ {a} a^{\dagger} _ {b} a^{\dagger} _ {c} \dots a_{i} a_{j} a_{k} \dots | \Psi \rangle
$$

An improvement to the CI method is the coupled-cluster theory (CC). Exponential operators are introduced to CC:

$$
| \Psi_{CC} \rangle = \exp{(\hat{T})} | \Psi_{HF} \rangle
$$

The coupled-cluster operator $\hat{T}$ is the sum of excitation operators.

$$
\hat{T} = \sum_{p\not\in occ\\\\q\in occ}{\theta^{p} _ {q} E^{p} _ {q}} + \sum_{pq\not\in occ\\\\rs\in occ}{\theta^{pq} _ {rs} E^{pq} _ {rs}} + \dots
$$

$\theta$ is similar to $C$ in the CI method, and is the parameter to be solved. It is easy to know from the Taylor's expansion of the exponent that even if the coupled-cluster operator $\hat{T}$ includes only a low-order excitation term, the $\exp{(\hat{T})}$ may also implicitly include a high-order excitation term. This also makes a convergence speed of the CC method to the FCI wave function much faster than that of the CI. For truncation to K excitation, for example, K=2, the accuracy of the CCSD exceeds that of the CISD.

<!--
Generally, if a method can achieve chemical accuracy, that is, the difference between the energy computed by this method and the FCI energy is less than 1 kcal/mol, it is considered that the method has good accuracy. Truncation to three-excited CCSD(T) can meet this standard in most cases.
-->

The effect of electron correlation is to reduce the total energy, so the ground state energy of HF is slightly higher than that of CCSD and FCI. In addition, it is easy to find that the computing volume of FCI is much greater than that of CCSD and HF. The `MolecularData` function encapsulated by OpenFermion and the `run_pyscf` function encapsulated by OpenFermion-PySCF are used for demonstration.

```python
molecule_of = MolecularData(
    geometry,
    basis,
    multiplicity=2 * spin + 1
)
molecule_of = run_pyscf(
    molecule_of,
    run_scf=1,
    run_ccsd=1,
    run_fci=1
)

print("Hartree-Fock energy: %20.16f Ha" % (molecule_of.hf_energy))
print("CCSD energy: %20.16f Ha" % (molecule_of.ccsd_energy))
print("FCI energy: %20.16f Ha" % (molecule_of.fci_energy))
```

```bash
Hartree-Fock energy:  -7.8633576215351200 Ha
CCSD energy:  -7.8823529091527051 Ha
FCI energy:  -7.8823622867987249 Ha
```

In the preceding example, HF, CCSD, and FCI are used to compute the total energy. If you collect statistics on the runtime, you will find that $T_{HF}<T_{CCSD}\ll T_{FCI}$. It is more obvious if you use the system with larger calculation amount, such as ethylene molecule. In addition, for the total computed energy, you will find that $E_{HF}>E_{CCSD}>E_{FCI}$. After the computing is complete, save the result to the `molecule_file` file (`molecule_of.filename`).

```python
molecule_of.save()
molecule_file = molecule_of.filename
print(molecule_file)
```

```bash
/home/xuxs/anaconda3/envs/p37/lib/python3.7/site-packages/openfermion/testing/data/H1-Li1_sto3g_singlet
```

One of the major obstacles to quantum chemistry is the volume of computation. As the system size (electron number and atomic number) increases, the time required for solving the FCI wave function and ground state energy increases by about $2^{N}$. Even for small molecules such as ethylene molecules, FCI computing is not easy. Quantum computers provide a possible solution to this problem. Research shows that quantum computers can simulate the time-dependent evolution of Hamiltonian in terms of polynomial time complexity. Compared with classical computers, quantum computers exponentially accelerate the chemical simulation on quantum processors. This tutorial introduces one of the quantum algorithms: VQE.

## Variational Quantum Eigensolver (VQE)

The VQE is a hybrid quantum-classical algorithm. It uses the variational principle to solve the ground state wave function. The optimization of variational parameters is carried out on the classical computer.

### Variational Principle

The variational principle may be expressed in the following form:

$$
E_{0} \le \frac{\langle \Psi_{t} | \hat{H} | \Psi_{t} \rangle}{\langle \Psi_{t} | \Psi_{t} \rangle}
$$

In the preceding formula, $| \Psi_{t} \rangle$ indicates the probe wave function. The variational principle shows that the ground state energy obtained by any probe wave function is always greater than or equal to the real ground state energy under certain conditions. The variational principle provides a method for solving the molecular ground state Schrödinger equation. A parameterized function $f(\theta)$ is used as an approximation of the accurate ground state wave function, and the accurate ground state energy is approximated by optimizing the parameter $\theta$.

### Initial State Preparation

The $N$-electron HF wave function also has a very concise form under the quadratic quantization expression:

$$
| \Psi_{HF} \rangle = \prod^{i=0} _{N-1}{a^{\dagger} _{i}| 0 \rangle}
$$

The above formula builds a bridge from quantum chemical wave function to quantum computing: $|0\rangle$ is used to represent a non-occupied orbit, and $|1\rangle$ is used to represent an orbit occupied by an electron. Therefore, the $N$-electron HF wave function may be mapped to a string of $M+N$ quantum bits $| 00\dots 11\dots \rangle$. $M$ indicates the number of unoccupied tracks.

The following code builds an HF initial state wave function corresponding to the LiH molecule. In Jordan-Wigner transformation, $N$ $\text{X}$ gates are applied to $|000\dots\rangle$.

```python
hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(molecule_of.n_electrons)])
print(hartreefock_wfn_circuit)
```

```bash
X(0)
X(1)
X(2)
X(3)
```

We can build a probe wave function in the following form:

$$
| \Psi_{t} \rangle = U(\theta) | \Psi_{HF} \rangle
$$

$U(\theta)$ represents a unitary transformation that may be simulated by using a quantum circuit. $| \Psi_{HF} \rangle$ is used as an initial state, and may be conveniently prepared by using a plurality of single-bit $\text{X}$ gates. A specific form of the $U(\theta) | \Psi_{HF} \rangle$ is also referred to as wave function ansatz.

### Wave Function Ansatz

The coupled-cluster theory mentioned above is a very efficient wave function ansatz. To use it on a quantum computer, you need to make the following modifications:

$$
| \Psi_{UCC} \rangle = \exp{(\hat{T} - \hat{T}^{\dagger})} | \Psi_{HF} \rangle
$$

UCC is short for unitary coupled-cluster theory. $\hat{T}^{\dagger}$ represents the Hermite conjugate of $\hat{T}$. In this way, $\exp{(\hat{T} - \hat{T}^{\dagger})}$ is the unitary operator. [Peruzzo et al.](https://doi.org/10.1038/ncomms5213) first performed chemical simulation experiments on quantum computers using VQE and unitary coupled-cluster with singles and doubles (UCCSD) in 2014. It should be noted that, by default, the parameter $\{\theta\}$ in the coupled-cluster operator is a real number. There is no problem with this hypothesis in molecular systems. In periodic systems, the study of [Liu Jie et al.](https://doi.org/10.1021/acs.jctc.0c00881) suggests that a unitary coupled-cluster can result in errors due to the neglect of the complex numbers. This tutorial does not discuss the application of unitary coupled-cluster in periodic systems.

The `generate_uccsd` function in the circuit module of MindQuantum can be used to read the computing result saved in `molecule_file`, build the UCCSD wave function by one click, and obtain the corresponding quantum circuit.

```python
ansatz_circuit, \
init_amplitudes, \
ansatz_parameter_names, \
hamiltonian_QubitOp, \
n_qubits, n_electrons = generate_uccsd(molecule_file, th=-1)
```

```bash
ccsd:-7.882352909152705.
fci:-7.882362286798725.
```

`generate_uccsd` packs functions related to the unitary coupled-cluster, including multiple steps such as deriving a molecular Hamiltonian, building a unitary coupled-cluster ansatz operator, and extracting a coupled-cluster coefficient computed by CCSD. This function reads the molecule by entering its file path. The parameter `th` indicates the to-be-updated gradient threshold of a parameter in the quantum circuit. In the section [Building a Unitary Coupled-Cluster Ansatz Step by Step](#building-a-unitary-coupled-cluster-ansatz-step-by-step), we will demonstrate how to use the related interfaces of MindQuantum to complete the steps. A complete quantum circuit includes an HF initial state and a UCCSD ansatz, as shown in the following code:

```python
total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()
print("Number of parameters: %d" % (len(ansatz_parameter_names)))
```

```bash
==============================Circuit Summary==============================
|Total number of gates  : 12612.                                          |
|Parameter gates        : 640.                                            |
|with 44 parameters are : p40, p9, p8, p3, p32, p28, p15, p4, p18, p22... |
|Number qubit of circuit: 12                                              |
===========================================================================
Number of parameters: 44
```

For the LiH molecule, the UCCSD wave function ansatz includes 44 variational parameters. The total number of quantum bit gates of the circuit is 12612, and a total of 12 quantum bits are needed for simulation.

### VQE Procedure

The procedure for solving the molecular ground state by using the VQE is as follows:

1. Prepare the HF initial state: $| 00\dots11\dots \rangle$.
2. Define the wave function ansatz, such as UCCSD.
3. Convert the wave function into a parameterized quantum circuit.
4. Initialize the variational parameters, for example, set all parameters to 0.
5. Obtain the energy $E(\theta)$ of the molecular Hamiltonian under the set of variational parameters and the derivative $\{ {\partial E} / {\partial \theta_{i}} \}$ of the energy about the parameters by means of multiple measurements on the quantum computer.
6. Use optimization algorithms, such as gradient descent and BFGS, to update variational parameters on classical computers.
7. Transfer the new variational parameters to the quantum circuit for updating.
8. Repeat steps 5 to 7 until the convergence criteria are met.
9. End.

In step 5, the derivative $\{ {\partial E} / {\partial \theta_{i}} \}$ of the energy about the parameter may be computed by using a parameter-shift rule on a quantum computer, or may be computed by simulating a parameter-shift rule or a finite difference method in a simulator. This is a relatively time-consuming process. Based on the MindSpore framework, MindQuantum provides the automatic derivation function similar to machine learning, which can efficiently compute the derivatives of parameterized quantum circuits in simulation. The following uses MindQuantum to build a parameterized UCCSD quantum circuit with an automatic derivation function:

```python
molecule_pqc = generate_pqc_operator(
    ["null"], ansatz_parameter_names,
    RX("null").on(0) + total_circuit,
    Hamiltonian(hamiltonian_QubitOp))
```

MindQuantum needs to provide two sets of circuits (and parameters) as the encoding circuit and Ansatz circuit. Here, `RX("null")` is used as an encoding circuit, and invalidate `null` by setting it to 0. You can obtain the energy $E(\theta)=\langle \Psi_{UCC}(\theta) | \hat{H} | \Psi_{UCC}(\theta) \rangle$ corresponding to the variational parameter and the derivative of each variational parameter by transferring a specific value of the parameter to `molecule_pqc`.

Next, steps 5 to 7 in VQE optimization need to be performed, that is, parameterized quantum circuits need to be optimized. Based on the MindSpore framework, you can use the parameterized quantum circuit operator `molecule_pqc` to build a neural network model, and then optimize the variational parameters by using a method similar to training the neural network.

```python
class PQCNet(ms.nn.Cell):
    def __init__(self, pqc):
        super(PQCNet, self).__init__()
        self.pqc = pqc
        self.weight =  Parameter(initializer("Zeros",
            len(self.pqc.ansatz_params_names)),
            name="weight")
        self.encoder_data_dummy = ms.Tensor([[0]],
            self.weight.dtype)

    def construct(self):
        energy, _, grads = self.pqc(self.encoder_data_dummy, self.weight)
        return energy

molecule_pqcnet = PQCNet(molecule_pqc)
```

Here, we manually build a basic `PQCNet` as a model example. This model can be used similar to a conventional machine learning model, for example, optimizing weights and calculating derivatives. A better choice is to use `MindQuantumAnsatzOnlyLayer` encapsulated in MindQuantum, which will be demonstrated later.

The built `PQCNet` uses the `"Zeros"` keyword to initialize all variational parameters to 0. The computing result of CCSD or second order Møller-Plesset perturbation theory (MP2) can also be used as the initial value of the variational parameters of unitary coupled-clusters. In this case, $E(\vec{0})=\langle \Psi_{UCC}(\vec{0}) | \hat{H} | \Psi_{UCC}(\vec{0}) \rangle = E_{HF}$.

```python
initial_energy = molecule_pqcnet()
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))
```

```bash
Initial energy:  -7.8633575439453125
```

Finally, the Adam optimizer of MindSpore is used for optimization. The learning rate is set to $1\times 10^{-2}$, and the optimization termination standard is set to $\left.|\epsilon|\right.= \left.|E^{k+1} - E^{k}|\right. \le 1\times 10^{-8}$.

```python
optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

eps = 1.e-8
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while abs(energy_diff) > eps:
    energy_i = train_pqcnet().asnumpy()
    if iter_idx % 5 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1

print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Optimized amplitudes: \n", molecule_pqcnet.weight.asnumpy())
```

```bash
Step   0 energy  -7.8633575439453125
Step   5 energy  -7.8726239204406738
Step  10 energy  -7.8821778297424316
Step  15 energy  -7.8822836875915527
Step  20 energy  -7.8823199272155762
Step  25 energy  -7.8823370933532715
Step  30 energy  -7.8823437690734863
Step  35 energy  -7.8618836402893066
Step  40 energy  -7.8671770095825195
Step  45 energy  -7.8751692771911621
Step  50 energy  -7.8822755813598633
Step  55 energy  -7.8812966346740723
Step  60 energy  -7.8823189735412598
Step  65 energy  -7.8823523521423340
Optimization completed at step  67
Optimized energy:  -7.8823528289794922
Optimized amplitudes:
    [ 2.3980068e-04  1.8912849e-03  3.5044324e-02  1.6005965e-02
    -1.9985158e-07  9.0940151e-04  1.6222824e-05  1.4160988e-02
    -1.1072063e-07  9.0867787e-04  1.3825165e-05  1.4166672e-02
    -5.4699212e-04  4.2679289e-04  2.8641545e-03  5.3817011e-02
    2.3320253e-04  1.7034533e-07  6.6684343e-08 -2.7686235e-07
    7.2332718e-08  1.2834757e-05 -1.0439425e-04  7.1826143e-08
    3.6483241e-06  6.1677817e-08  3.1003920e-06  7.9770159e-04
    -5.4951470e-02  3.0904056e-03 -4.4321241e-05  8.5840838e-07
    -1.9589644e-08 -4.9430941e-08  8.6163556e-07 -2.5008637e-07
    2.1493735e-08 -4.6331229e-06  3.0904033e-03  9.5311613e-08
    -4.8755901e-08  2.0483398e-08 -3.9453280e-06  3.7235476e-04]
```

It can be seen that the computing result of unitary coupled-cluster is very close to that of FCI, and has good accuracy.

## Building a Unitary Coupled-Cluster Ansatz Step by Step

<a id="step-by-step"></a>

In the preceding part, the `generate_uccsd` is used to build all the content required for designing the unitary coupled-cluster. In this section, the steps are split, we get the coupled-cluster operator, the corresponding quantum circuit and the initial guess of the variational parameters from the classical CCSD results.
First, import some extra dependencies, including the related functions of the HiQfermion module in MindQuantum.

```python
from mindquantum.hiqfermion.transforms import Transform
from mindquantum.hiqfermion.ucc import get_qubit_hamiltonian
from mindquantum.hiqfermion.ucc import uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes
from mindquantum.circuit import TimeEvolution
from mindquantum.nn import MindQuantumAnsatzOnlyLayer
```

The molecule Hamiltonian uses `get_qubit_hamiltonian` to read the previous computing result. The result is as follows:

```python
hamiltonian_QubitOp = get_qubit_hamiltonian(molecule_of)
```

The unitary coupled-cluster operator $ \hat{T} - \hat{T}^{\dagger} $ can be built using `uccsd_singlet_generator`. Provide the total number of quantum bits (total number of spin orbits) and the total number of electrons, and set `anti_hermitian=True`.

```python
ucc_fermion_ops = uccsd_singlet_generator(
    molecule_of.n_qubits, molecule_of.n_electrons, anti_hermitian=True)
```

The `ucc_fermion_ops` built in the previous step is parameterized. Use the Jordan-Wigner transformation to map the Fermi excitation operator to the Pauli operator:

```python
ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
```

Next, we need to obtain the quantum circuit corresponding to the unitary operator $ \exp{(\hat{T} - \hat{T}^{\dagger})} $. `TimeEvolution` can generate the circuit corresponding to $ \exp{(-i\hat{H}t)} $, where $\hat{H} $is a Hermitian operator, and $t$ is a real number. Note that when `TimeEvolution` is used, `ucc_qubit_ops` already contains the complex number factor $i$. Therefore, you need to divide `ucc_qubit_ops` by $i$ or extract the imaginary part of `ucc_qubit_ops`.

```python
ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
ansatz_parameter_names = ansatz_circuit.para_name
```

`ansatz_parameter_names` is used to record the parameter names in the circuit. So far, we have obtained the contents required by the VQE quantum circuit, including the Hamiltonian `hamiltonian_QubitOp` and the parameterized wave function ansatz `ansatz_circuit`. By referring to the preceding steps, we can obtain a complete state preparation circuit. `hartreefock_wfn_circuit` mentioned above is used as the Hartree-Fock reference state:

```python
total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()
```

```bash
======================================Circuit Summary======================================
|Total number of gates  : 12612.                                                          |
|Parameter gates        : 640.                                                            |
|with 44 parameters are : d1_3, d2_26, d2_6, d2_1, d2_2, d2_14, d1_1, s_1, d2_16, d2_11...|
|Number qubit of circuit: 12                                                              |
===========================================================================================
```

Next, you need to provide a reasonable initial value for the variational parameter. The `PQCNet` built in the preceding text uses 0 as the initial guess by default, which is feasible in most cases. However, using CCSD's computational data as a starting point for UCC may be better. Use the `uccsd_singlet_get_packed_amplitudes` function to extract CCSD parameters from `molecule_of`.

```python
init_amplitudes_ccsd = uccsd_singlet_get_packed_amplitudes(
    molecule_of.ccsd_single_amps, molecule_of.ccsd_double_amps, molecule_of.n_qubits, molecule_of.n_electrons)
init_amplitudes_ccsd = [init_amplitudes_ccsd[param_i] for param_i in ansatz_parameter_names]
```

`MindQuantumAnsatzOnlyLayer` can be used to easily obtain a machine learning model based on a parameterized quantum circuit by using a parameter and a quantum circuit:

```python
molecule_pqcnet = MindQuantumAnsatzOnlyLayer(
    ansatz_parameter_names, total_circuit, Hamiltonian(hamiltonian_QubitOp.real))
```

`init_amplitudes_ccsd` (coupled-cluster coefficient computed by CCSD) is used as an initial variational parameter:

```python
molecule_pqcnet.weight = Parameter(ms.Tensor(init_amplitudes_ccsd, molecule_pqcnet.weight.dtype))
initial_energy = molecule_pqcnet()
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))
```

```bash
Initial energy:  -7.8173098564147949
```

In this example, CCSD's initial guess does not provide a better starting point. You can test and explore more molecules and more types of initial values (such as initial guesses of random numbers). Finally, the VQE is optimized. The optimizer still uses Adam, and the convergence standard remains unchanged. The code used for optimization is basically the same as that described in the preceding sections. You only need to update the corresponding variables.

```python
optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

print("eps: ", eps)
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while abs(energy_diff) > eps:
    energy_i = train_pqcnet().asnumpy()
    if iter_idx % 5 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1

print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Optimized amplitudes: \n", molecule_pqcnet.weight.asnumpy())
```

```bash
eps:  1e-08
Step   0 energy  -7.8173098564147949
Step   5 energy  -7.8740763664245605
Step  10 energy  -7.8818783760070801
Step  15 energy  -7.8821649551391602
Step  20 energy  -7.8822622299194336
Step  25 energy  -7.8823084831237793
Step  30 energy  -7.8823180198669434
Step  35 energy  -7.8737111091613770
Step  40 energy  -7.8724455833435059
Step  45 energy  -7.8801403045654297
Step  50 energy  -7.8821926116943359
Step  55 energy  -7.8818311691284180
Step  60 energy  -7.8823456764221191
Optimization completed at step  64
Optimized energy:  -7.8823523521423340
Optimized amplitudes:
    [-2.4216002e-04  1.8924323e-03 -3.4653045e-02  1.5943546e-02
    3.6362690e-07  9.0936717e-04 -1.7181528e-05  1.4154296e-02
    -4.4650793e-08  9.0864423e-04 -2.6399141e-06  1.4159971e-02
    5.4558384e-04  4.2672374e-04 -2.8494308e-03  5.3833455e-02
    2.3033506e-04  1.2578158e-06  3.3855862e-08  7.3955505e-08
    -5.2005623e-07  2.9746575e-08  1.2325607e-08  1.1919828e-05
    -1.0492613e-04  7.9503102e-04  3.8478893e-06  5.9738107e-07
    -5.4855812e-02  3.0889052e-03  7.9252044e-05 -1.5384763e-06
    -1.5373821e-06 -3.0784176e-07 -3.5303248e-08  1.7360321e-08
    4.4359115e-07 -4.9067144e-06  3.0889027e-03  1.3888703e-07
    -1.6715177e-08  6.3234533e-09 -7.5149819e-07  3.7140178e-04]
```

## Summary

In this case, the ground state energy of the LiH molecule is obtained by using the quantum neural network in two methods. In the first method, we use the `generate_uccsd` function packaged by MindQuantum to generate a quantum neural network that can solve this problem. In the second method, we build a similar quantum neural network step by step. The final results are the same.
