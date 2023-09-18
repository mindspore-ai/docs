# VQE Application in Quantum Chemistry Computing

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindquantum/docs/source_en/vqe_for_quantum_chemistry.md)

## Overview

Quantum chemistry refers to solving the numerical values of the time-dependent or time-independent Schrödinger equations by using the basic theory and method of quantum mechanics. Quantum chemical simulation on high-performance computers has become an important method to study the physical and chemical properties of materials. However, the exact solution of the Schrödinger equation has exponential complexity, which severely constrains the scale of the chemical system that can be simulated. The development of quantum computing in recent years provides a feasible way to solve this problem. It is expected that the Schrödinger equation can be solved with high accuracy on quantum computers under the complexity of polynomials.

[Peruzzo et al.](https://doi.org/10.1038/ncomms5213) first applied the VQE and [unitary coupled-cluster theory](https://linkinghub.elsevier.com/retrieve/pii/S0009261489873725) to quantum chemistry simulation in 2014 to solve the ground state energy of He-H<sup>+</sup>. The VQE is a hybrid quantum-classical algorithm and is widely used in chemical simulation based on quantum algorithms. This tutorial describes how to use the VQE to solve the ground-state energy of a molecular system.

This tutorial consists of the following parts:

1. Introduction to the quantum chemistry
2. VQE application
3. Using MindSpore Quantum to perform VQE simulation with efficient and automatic derivation

> This document applies to the CPU environment.

## Environment Preparation

In this tutorial, the following environments need to be installed:

- NumPy
- SciPy
- [MindSpore Quantum](https://gitee.com/mindspore/mindquantum)
- [MindSpore](https://gitee.com/mindspore/mindspore)
- PySCF
- OpenFermion
- OpenFermion-PySCF

> The preceding dependencies can be installed by running the `pip` command.

## Importing Dependencies

Import the modules on which this tutorial depends.

```python
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import generate_uccsd
import mindspore as ms

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
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
    ["H", [0.0, 0.0, 1.0 * dist]],
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
| \Psi_{CI} \rangle = C_{0} | \Psi_{HF} \rangle + \sum^{a\rightarrow\infty} _{i\in occ, a\not\in occ}{C^{a} _{i} | \Psi^{a} _{i} \rangle } + \sum^{ab\rightarrow\infty} _{ij\in occ, ab\not\in occ}{C^{ab} _{ij} | \Psi^{ab} _{ij} \rangle }
$$

$| \Psi^{a}_{i} \rangle + \dots$ in the preceding formula represents a single excitation wave function of the electron from the orbit $i$ to the orbit $a$, and so on. A CI that considers only single excitation and double excitation is called a configuration interaction with singles and doubles (CISD). The CI that takes into account all the ground-state HF wave functions to N excitation wave functions is called full configuration interaction (FCI). The FCI wave function is the exact solution of the time-independent Schrödinger equation under the given basis function.

### Second Quantization

Under the second quantization expression, the Hamiltonian of the system has the following form:

$$
\hat{H} = \sum_{p, q}{h^p_q E^p_q} + \sum_{p, q, r, s}{\frac{1}{2} g^{pq} _ {rs} E^{pq}_{rs}}
$$

$E^{p} _{q}$ and $E^{pq}_ {rs}$ are as follows:

$$
\begin{align*}
E^{pq} _{rs} &= a^{\dagger} _{p} a^{\dagger} _{q} a _ {r} a _ {s}\\
E^p_q &= a^{\dagger}_pa_q
\end{align*}
$$

$a^{\dagger} _{p}$ and $a_ {q}$ are creation operator and annihilation operator, respectively.

The excited-state wave function can be expressed conveniently by using a second quantization expression method:

$$
| \Psi^{abc\dots} _ {ijk\dots} \rangle = a^{\dagger} _ {a} a^{\dagger} _ {b} a^{\dagger} _ {c} \dots a _ {i} a_{j} a_{k} \dots | \Psi \rangle
$$

An improvement to the CI method is the coupled-cluster theory (CC). Exponential operators are introduced to CC:

$$
| \Psi_{CC} \rangle = \exp{(\hat{T})} | \Psi_{HF} \rangle
$$

The coupled-cluster operator $\hat{T}$ is the sum of excitation operators.

$$
\hat{T} = \sum_{p\not\in occ, q\in occ}{\theta^{p} _ {q} E^{p} _ {q}} + \sum_{pq\not\in occ, rs\in occ}{\theta^{pq} _ {rs} E^{pq} _ {rs}} + \dots
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
Hartree-Fock energy:  -7.8633576215351129 Ha
CCSD energy:  -7.8823529091526972 Ha
FCI energy:  -7.8823622867987213 Ha
```

In the preceding example, HF, CCSD, and FCI are used to compute the total energy. If you collect statistics on the runtime, you will find that $T_{HF}<T_{CCSD}\ll T_{FCI}$. It is more obvious if you use the system with larger calculation amount, such as ethylene molecule. In addition, for the total computed energy, you will find that $E_{HF}>E_{CCSD}>E_{FCI}$. After the computing is complete, save the result to the `molecule_file` file (`molecule_of.filename`).

```python
molecule_of.save()
molecule_file = molecule_of.filename
print(molecule_file.split('/')[-1])
```

```text
H1-Li1_sto3g_singlet
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
q0: ──X──

q1: ──X──

q2: ──X──

q3: ──X──
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

The `generate_uccsd` function in the circuit module of MindSpore Quantum can be used to read the computing result saved in `molecule_file`, build the UCCSD wave function by one click, and obtain the corresponding quantum circuit.

```python
ansatz_circuit, \
init_amplitudes, \
ansatz_parameter_names, \
hamiltonian_QubitOp, \
n_qubits, n_electrons = generate_uccsd(molecule_file, threshold=-1)
```

```bash
ccsd:-7.882352909152697.
fci:-7.882362286798721.
```

`generate_uccsd` packs functions related to the unitary coupled-cluster, including multiple steps such as deriving a molecular Hamiltonian, building a unitary coupled-cluster ansatz operator, and extracting a coupled-cluster coefficient computed by CCSD. This function reads the molecule by entering its file path. The parameter `th` indicates the to-be-updated gradient threshold of a parameter in the quantum circuit. In the section [Building a Unitary Coupled-Cluster Ansatz Step by Step](#building-a-unitary-coupled-cluster-ansatz-step-by-step), we will demonstrate how to use the related interfaces of MindSpore Quantum to complete the steps. A complete quantum circuit includes an HF initial state and a UCCSD ansatz, as shown in the following code:

```python
total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()
print("Number of parameters: %d" % (len(ansatz_parameter_names)))
```

```bash
============================Circuit Summary============================
|Total number of gates  : 15172.                                      |
|Parameter gates        : 640.                                        |
|with 44 parameters are :                                             |
|p0, p8, p1, p9, p2, p10, p3, p11, p4, p12..                        . |
|Number qubit of circuit: 12                                          |
=======================================================================
Number of parameters: 44
```

For the LiH molecule, the UCCSD wave function ansatz includes 44 variational parameters. The total number of quantum bit gates of the circuit is 12612, and a total of 12 quantum bits are needed for simulation.

### VQE Procedure

The procedure for solving the molecular ground state by using the VQE is as follows:

1. Prepare the HF initial state: $| 00\dots11\dots \rangle$.
2. Define the wave function ansatz, such as UCCSD.
3. Convert the wave function into a variational quantum circuit.
4. Initialize the variational parameters, for example, set all parameters to 0.
5. Obtain the energy $E(\theta)$ of the molecular Hamiltonian under the set of variational parameters and the derivative $\{ {\partial E} / {\partial \theta_{i}} \}$ of the energy about the parameters by means of multiple measurements on the quantum computer.
6. Use optimization algorithms, such as gradient descent and BFGS, to update variational parameters on classical computers.
7. Transfer the new variational parameters to the quantum circuit for updating.
8. Repeat steps 5 to 7 until the convergence criteria are met.
9. End.

In step 5, the derivative $\{ {\partial E} / {\partial \theta_{i}} \}$ of the energy about the parameter may be computed by using a parameter-shift rule on a quantum computer, or may be computed by simulating a parameter-shift rule or a finite difference method in a simulator. This is a relatively time-consuming process. Based on the MindSpore framework, MindSpore Quantum provides the automatic derivation function similar to machine learning, which can efficiently compute the derivatives of variational quantum circuits in simulation. The following uses MindSpore Quantum to build a parameterized UCCSD quantum circuit with an automatic derivation function:

```python
sim = Simulator('mqvector', total_circuit.n_qubits)
molecule_pqc = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_QubitOp), total_circuit)
```

You can obtain the energy $E(\theta)=\langle \Psi_{UCC}(\theta) | \hat{H} | \Psi_{UCC}(\theta) \rangle$ corresponding to the variational parameter and the derivative of each variational parameter by transferring a specific value of the parameter to `molecule_pqc`.

For example, we can use the following code to get the expectation of hamiltonian and the corresponding gradient when initial parameters of variational quantum circuit is zero.

```python
import numpy as np

n_params = len(total_circuit.params_name)
p0 = np.zeros(n_params)
f, g = molecule_pqc(p0)
print("Energy: ", f, "\nshape: ", f.shape, '\n')
print("Gradient: ", g, "\nshape: ", g.shape)
```

```bash
Energy:  [[-7.86335762+0.j]]
shape:  (1, 1)

Gradient:  [[[-1.17000470e-10+0.j -8.60518140e-02+0.j  7.34841121e-09+0.j
   -4.85545093e-02+0.j -4.27880918e-16+0.j -3.92769093e-02+0.j
   -2.03178123e-14+0.j -9.59481736e-02+0.j -2.19963362e-15+0.j
   -3.92769093e-02+0.j  2.01070916e-15+0.j -9.59481736e-02+0.j
   -1.36457968e-10+0.j -2.89649669e-02+0.j  4.02404639e-10+0.j
   -4.91813235e-01+0.j -9.35292655e-04+0.j -1.66209006e-16+0.j
   -3.46620125e-17+0.j  1.54862629e-17+0.j  1.54555976e-16+0.j
    2.53664212e-17+0.j -2.06994596e-17+0.j  5.06813144e-03+0.j
    1.08542342e-02+0.j -1.28614257e-02+0.j -4.90653894e-17+0.j
    1.10397120e-17+0.j  1.33973783e-01+0.j -3.03063678e-02+0.j
    6.13317367e-19+0.j  6.13317367e-19+0.j  7.90297720e-29+0.j
    5.24386349e-17+0.j -3.70848761e-17+0.j  3.33490478e-17+0.j
    1.29095036e-28+0.j  1.15391966e-16+0.j -3.03063678e-02+0.j
   -5.24386349e-17+0.j  4.89887785e-17+0.j -4.22508287e-17+0.j
   -2.44510201e-17+0.j -1.68035030e-03+0.j]]]
shape:  (1, 1, 44)
```

Throw the above calculation, we get the energy and gradient value and the user can use these data according their practical needs. Now we following the step of (5)~(7) of optimization of VQE, to optimize of variational quantum circuit. Here we use the optimizer in scipy to optimize our quantum circuit. First, we need to define the optimization function that scipy required:

```python
def fun(p0, molecule_pqc, energy_list=None):
    f, g = molecule_pqc(p0)
    f = np.real(f)[0, 0]
    g = np.real(g)[0, 0]
    if energy_list is not None:
        energy_list.append(f)
        if len(energy_list) % 5 == 0:
            print(f"Step: {len(energy_list)},\tenergy: {f}")
    return f, g

fun(p0, molecule_pqc)
```

```bash
(-7.863357621536957,
 array([-1.17000470e-10, -8.60518140e-02,  7.34841121e-09, -4.85545093e-02,
        -4.27880918e-16, -3.92769093e-02, -2.03178123e-14, -9.59481736e-02,
        -2.19963362e-15, -3.92769093e-02,  2.01070916e-15, -9.59481736e-02,
        -1.36457968e-10, -2.89649669e-02,  4.02404639e-10, -4.91813235e-01,
        -9.35292655e-04, -1.66209006e-16, -3.46620125e-17,  1.54862629e-17,
         1.54555976e-16,  2.53664212e-17, -2.06994596e-17,  5.06813144e-03,
         1.08542342e-02, -1.28614257e-02, -4.90653894e-17,  1.10397120e-17,
         1.33973783e-01, -3.03063678e-02,  6.13317367e-19,  6.13317367e-19,
         7.90297720e-29,  5.24386349e-17, -3.70848761e-17,  3.33490478e-17,
         1.29095036e-28,  1.15391966e-16, -3.03063678e-02, -5.24386349e-17,
         4.89887785e-17, -4.22508287e-17, -2.44510201e-17, -1.68035030e-03]))
```

Here, the `fun` that we define can correctly to return the data that we need: a real energy value, and a array of gradient value with the same size of parameters. Now, we use `bfgs` optimizer in scipy to finish the optimization.

```python
from scipy.optimize import minimize

energy_list = []
res = minimize(fun, p0, args=(molecule_pqc, energy_list), method='bfgs', jac=True)
```

```bash
Step: 5, energy: -7.880227726053225
Step: 10, energy: -7.881817123969861
Step: 15, energy: -7.882213242986122
Step: 20, energy: -7.882345337008459
Step: 25, energy: -7.882352494991635
Step: 30, energy: -7.882352691272213
Step: 35, energy: -7.882352707864624
Step: 40, energy: -7.882352708256735
Step: 45, energy: -7.882352708339958
```

So, we finished the gradient optimization of variational quantum circuit. Here, `energy_list` is going to store the energy during optimization. Here, we briefly introduce the usage of `minimize`:

- `fun`: The first arg is the function you want to optimize.
- `p0`: The second arg is the initial value of variables.
- `args`: The other argument of `fun` except the first argument. According the definition of `fun`, we choose `args=(molecule_pqc, energy_list)`.
- `method`: The optimization algorithm. Here we use a second order optimization algorithm `bfgs`. For more optimization algorithm, please refer: [scipy tutorial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
- `jac`: To info that whether `fun` return gradient. Here we use `True`, because MindSpore Quantum can calculate the accuracy gradient value of variational quantum circuit. If use `False`, `minimize` framework will calculate the approximated gradient value base on difference method.

`res` is the optimization result of `scipy`, including optimized parameters, the optimized value and evolution steps.

```python
print(f"Ground state: \n{res.fun}\n")
print(f"FCI: \n-7.882362286798721\n")
print(f"Optimized amplitudes: \n{res.x}")
```

```bash
Ground state:
-7.882352708347493

FCI:
-7.882362286798721

Optimized amplitudes:
[ 2.38712434e-04  1.89072212e-03  3.52371962e-02  1.60368062e-02
  4.58341956e-09  9.09417397e-04  8.87731878e-10  1.41638572e-02
  8.70408136e-10  9.08690861e-04  1.01871578e-09  1.41700430e-02
 -5.47655818e-04  4.26823250e-04  2.87179882e-03  5.38109243e-02
  2.34704374e-04 -1.53416806e-08  8.33679682e-08 -8.59399531e-08
 -1.09241359e-08  8.69631759e-08 -8.85803852e-08  1.33003007e-05
 -1.04125728e-04  7.99021032e-04 -6.16712752e-10 -5.09574759e-10
 -5.50005269e-02  3.09114892e-03 -4.78658817e-11 -1.05874198e-07
  1.05877675e-07  4.10043904e-07  2.63845620e-07 -2.76942779e-07
 -1.69080845e-13  2.10388157e-09  3.09108797e-03  1.90755827e-08
  1.86729203e-07 -1.76508410e-07  6.07581905e-10  3.72843789e-04]
```

We can see here the result of ucc method is very close to FCI method with very good accuracy.

## Building a Unitary Coupled-Cluster Ansatz Step by Step

<a id="step-by-step"></a>

In the preceding part, the `generate_uccsd` is used to build all the content required for designing the unitary coupled-cluster. In this section, the steps are split, we get the coupled-cluster operator, the corresponding quantum circuit and the initial guess of the variational parameters from the classical CCSD results.
First, import some extra dependencies, including the related functions of the HiQfermion module in MindSpore Quantum.

```python
from mindquantum.algorithm.nisq import Transform
from mindquantum.algorithm.nisq import get_qubit_hamiltonian
from mindquantum.algorithm.nisq import uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes
from mindquantum.core.operators import TimeEvolution
```

The molecule Hamiltonian uses `get_qubit_hamiltonian` to read the previous computing result. The result is as follows:

```python
hamiltonian_QubitOp = get_qubit_hamiltonian(molecule_of)
```

The unitary coupled-cluster operator $\hat{T} - \hat{T}^{\dagger}$ can be built using `uccsd_singlet_generator`. Provide the total number of quantum bits (total number of spin orbits) and the total number of electrons, and set `anti_hermitian=True`.

```python
ucc_fermion_ops = uccsd_singlet_generator(
    molecule_of.n_qubits, molecule_of.n_electrons, anti_hermitian=True)
```

The `ucc_fermion_ops` built in the previous step is parameterized. Use the Jordan-Wigner transformation to map the Fermi excitation operator to the Pauli operator:

```python
ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
```

Next, we need to obtain the quantum circuit corresponding to the unitary operator $\exp{(\hat{T} - \hat{T}^{\dagger})}$. `TimeEvolution` can generate the circuit corresponding to $\exp{(-i\hat{H}t)}$, where $\hat{H}$ is a Hermitian operator, and $t$ is a real number. Note that when `TimeEvolution` is used, `ucc_qubit_ops` already contains the complex number factor $i$. Therefore, you need to divide `ucc_qubit_ops` by $i$ or extract the imaginary part of `ucc_qubit_ops`.

```python
ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
ansatz_parameter_names = ansatz_circuit.params_name
```

`ansatz_parameter_names` is used to record the parameter names in the circuit. So far, we have obtained the contents required by the VQE quantum circuit, including the Hamiltonian `hamiltonian_QubitOp` and the parameterized wave function ansatz `ansatz_circuit`. By referring to the preceding steps, we can obtain a complete state preparation circuit. `hartreefock_wfn_circuit` mentioned above is used as the Hartree-Fock reference state:

```python
total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()
```

```bash
==================================Circuit Summary==================================
|Total number of gates  : 15172.                                                  |
|Parameter gates        : 640.                                                    |
|with 44 parameters are :                                                         |
|s_0, d1_0, s_1, d1_1, s_2, d1_2, s_3, d1_3, s_4, d1_4..                        . |
|Number qubit of circuit: 12                                                      |
===================================================================================
```

Next, you need to provide a reasonable initial value for the variational parameter. The `PQCNet` built in the preceding text uses 0 as the initial guess by default, which is feasible in most cases. However, using CCSD's computational data as a starting point for UCC may be better. Use the `uccsd_singlet_get_packed_amplitudes` function to extract CCSD parameters from `molecule_of`.

```python
init_amplitudes_ccsd = uccsd_singlet_get_packed_amplitudes(
    molecule_of.ccsd_single_amps, molecule_of.ccsd_double_amps, molecule_of.n_qubits, molecule_of.n_electrons)
init_amplitudes_ccsd = [init_amplitudes_ccsd[param_i] for param_i in ansatz_parameter_names]
```

Just like the previous method, we can get the `grad_ops` with MindSpore Quantum, and optimize it with scipy.

```python
grad_ops = Simulator('mqvector', total_circuit.n_qubits).get_expectation_with_grad(
    Hamiltonian(hamiltonian_QubitOp.real),
    total_circuit)
```

`init_amplitudes_ccsd` (coupled-cluster coefficient computed by CCSD) is used as an initial variational parameter:

```python
energy_list = []
res = minimize(fun, init_amplitudes_ccsd, args=(grad_ops, energy_list), method='bfgs', jac=True)
```

```bash
Step: 5, energy: -7.878223282730547
Step: 10, energy: -7.880288481438961
Step: 15, energy: -7.882035668304055
Step: 20, energy: -7.882302370885741
Step: 25, energy: -7.882349803534313
Step: 30, energy: -7.882352702053751
Step: 35, energy: -7.8823527077335065
Step: 40, energy: -7.882352708347106
```

The final optimized result is shown as below.

```python
print(f"Ground state: \n{res.fun}\n")
print(f"FCI: \n-7.882362286798721\n")
print(f"Optimized amplitudes: \n{res.x}")
```

```bash
Ground state:
-7.882352708347106

FCI:
-7.882362286798721

Optimized amplitudes:
[-2.38716797e-04  1.89072948e-03 -3.52373113e-02  1.60368505e-02
 -1.65211897e-08  9.09419406e-04  6.97535496e-10  1.41633024e-02
 -6.40543396e-09  9.08685812e-04  3.62517408e-10  1.41706530e-02
  5.47788025e-04  4.26824061e-04 -2.87153659e-03  5.38109309e-02
  2.34736444e-04 -2.07143981e-07  1.78941118e-07 -1.62503048e-07
 -8.80911165e-08  4.07521713e-08 -3.40755199e-08  1.32909026e-05
  7.99087755e-04 -1.04066519e-04 -1.07974132e-09 -3.40797600e-10
 -5.50004943e-02  3.09140289e-03 -2.51213790e-09 -2.71345445e-11
  1.93711194e-10 -1.81505812e-07 -1.08665216e-07  1.19335275e-07
 -7.25358274e-12 -9.25316075e-10  3.09081391e-03 -4.66785554e-08
 -5.15818095e-08  5.28564624e-08 -3.02691203e-10  3.72803168e-04]
```

## Summary

In this case, the ground state energy of the LiH molecule is obtained by using scipy in two methods. In the first method, we use the `generate_uccsd` function packaged by MindSpore Quantum to generate a quantum neural network that can solve this problem. In the second method, we build a similar gradient operator step by step. The final results are the same.
