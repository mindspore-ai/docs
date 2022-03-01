# Quantum Phase Estimation Algorithm

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindquantum/docs/source_en/quantum_phase_estimation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## Overview

Quantum Phase Estimation Algorithm, or QPE for short, is the key to many quantum algorithms. Suppose a unitary operator $U$, which acts on its eigenstate $|u\rangle$ will have a phase $e^{2\pi i \varphi}$, now we assume that the eigenvalue of the $U$ operator is unknown, that is, $\varphi$ is unknown, but the $U$ operator and the eigenstate $|u\rangle$ are known, The role of the phase estimation algorithm is to estimate this phase $\varphi$.

![quantum phase estimation](./images/quantum_phase_estimation.png)

## Algorithm Analysis

The implementation of the quantum phase estimation algorithm requires two registers, the first register contains $t$ qubits initially at $|0\rangle$, the number of bits is related to the accuracy of the final phase estimation result and the success probability of the algorithm; the second register is initialized on the eigenstate $|u\rangle$ of the unitary operator $U$. The phase estimation algorithm is mainly divided into three steps:

1. Perform `Hadamard` gate operations on all qubits in the first register, and continuously perform `control U` gate operations on the second register, where the powers of $U$ gates are $2^0, 2^1,...,2^{t-1}$, and the control qubits are $q_{t-1}, q_{t-2},..., q_{1}, q_{0}$. Then the state in the first register will change to $$|\psi_1\rangle=\frac{1}{2^{t/2}}\left(|0\rangle+e^{i2\pi 2^{t-1}\varphi}|1\rangle\right)\left(|0\rangle+e^{i2\pi2^{t-2}\varphi}|1\rangle\right)...\left(|0\rangle+e^{i2\pi 2^{0}\varphi}|1\rangle\right) = \frac{1}{2^{t/2}}\sum_{k=0}^{2^t-1}e^{i2\pi\varphi k}|k\rangle$$ where $k$ is the decimal representation of the direct product state, for example, $k=0$ means that all t qubits in the first register are in the ground state $|00...00\rangle$, $k=2$ means $|00...10\rangle$, and so on.

2. Perform the inverse quantum Fourier transform on the first register, which is expressed as $QFT^\dagger$ in the circuit, and perform the inverse quantum Fourier transform on $|\psi_1\rangle$ to obtain $|\psi_2\rangle$ $$|\psi_2\rangle=QFT^\dagger|\psi_1\rangle =\frac{1}{2^t}\sum_{x=0}^{2^t-1}a_x|x\rangle$$ where $$a_x=\sum_{k=0}^{2^t-1}e^{2\pi i k(\varphi-x/2^t)}$$ is the probability amplitude corresponding to the eigenbasis vector $|x\rangle$ ($x=0.1,...,2^t$). It can be obtained from the above formula that when $2^t\varphi$ is an integer and $x=2^t\varphi$ is satisfied, the probability amplitude takes the maximum value of 1, at this time, the last state of the first register can accurately reflect $\varphi$; when $2^t\varphi$ is not an integer, $x$ is the estimate of $\varphi$, and the larger $t$, the higher the estimation accuracy.

3. Measure the qubits of the first register to obtain the final state of the first register $f=\sum_{x}^{2^t-1}a_x|x\rangle$, $x=0,1,...,2^t$, and find the maximum amplitude $a_{max}$ from it, then the $x$ in the corresponding eigenvector $|x\rangle$ divided by $2^t$ is the estimated value of phase.

## QPE code implementation

The following is an example to demonstrate how to implement the quantum phase estimation algorithm in MindQuantum. The `T` gate is selected as the unitary operator for estimation, from the definition of

$$T|1\rangle=e^{i\pi/4}|1\rangle$$

it can be known that the phase angle to be estimated is $\varphi=\frac{1}{8}$.

Now suppose we don't know the phase information of the `T` gate, but only know that the unitary operator $U$ is a `T` gate and the eigenstate is $|1\rangle$ , then we need to use the quantum phase estimation algorithm to find its corresponding eigenvalue, that is, we need to estimate the phase angle on the eigenvalue index.

First import the relevant dependencies.

```python
from mindquantum.core import Circuit, UN, T, H, X, Power, BARRIER
from mindquantum.simulator import Simulator
from mindquantum.algorithm import qft
import numpy as np
```

`UN` can specify quantum gates, target bits and control bits to build gate operations in the circuit; `Power` can get the exponential form of the specified quantum gate. Because we know that the eigenstate of the `T` gate is $|1\rangle$, the second register only needs 1 bit, and the more bits in the first register, the more accurate the result, here we use 4 bits.

So we need to build a 5-bit circuit, $q_0, q_1, q_2, q_3$ bits are used for estimation, belong to the first register, and $q_4$ belong to the second register to pass in the eigenstate of the $T$ operator.

Use `UN` to perform `Hadamard` gate operation on $q_0, q_1, q_2, q_3$, and use `X` gate to flip $q_4$ to obtain the eigenstate $|1\rangle$ of `T` gate.

```python
# pylint: disable=W0104
n = 4
circ = Circuit()
circ += UN(H, n) # Act h gate on the first 4 bits
circ += X.on(n)  # Act X gate on q4
circ
```

```text
q0: ──H──

q1: ──H──

q2: ──H──

q3: ──H──

q4: ──X──
```

With $q_4$ as the target bit, add the control $T^{2^i}$ gate.

```python
# pylint: disable=W0104
for i in range(n):
    circ += Power(T, 2**i).on(n, n - i - 1) # Add T^2^i gate, where q4 is the target bit and n-i-1 is the control bit
circ
```

```text
q0: ──H──────────────────────────●───
                                 │
q1: ──H───────────────────●──────┼───
                          │      │
q2: ──H────────────●──────┼──────┼───
                   │      │      │
q3: ──H─────●──────┼──────┼──────┼───
            │      │      │      │
q4: ──X────T^1────T^2────T^4────T^8──
```

Perform an inverse quantum Fourier transform on the bits in the first register.

```python
# pylint: disable=W0104
circ += BARRIER
circ += qft(range(n)).hermitian() # Inverse transform of quantum Fourier transform applied to the first 4 bits
circ
```

```text
q0: ──H──────────────────────────●──────────@───────────────────────────────────────────────────────PS(-π/8)────PS(-π/4)────PS(-π/2)────H──
                                 │          │                                                          │           │           │
q1: ──H───────────────────●──────┼─────@────┼──────────────────────────PS(-π/4)────PS(-π/2)────H───────┼───────────┼───────────●───────────
                          │      │     │    │                             │           │                │           │
q2: ──H────────────●──────┼──────┼─────@────┼─────────PS(-π/2)────H───────┼───────────●────────────────┼───────────●───────────────────────
                   │      │      │          │            │                │                            │
q3: ──H─────●──────┼──────┼──────┼──────────@────H───────●────────────────●────────────────────────────●───────────────────────────────────
            │      │      │      │
q4: ──X────T^1────T^2────T^4────T^8────────────────────────────────────────────────────────────────────────────────────────────────────────
```

Select the backend, pass in the total number of bits to create a simulator, evolve the quantum circuit, and get the final state.

```python
# pylint: disable=W0104
from mindquantum import Measure
sim = Simulator('projectq', circ.n_qubits)                      # Create an emulator
sim.apply_circuit(circ)                                         # Evolving the circuit with the simulator
qs = sim.get_qs()                                               # Obtain the evolved quantum state
res = sim.sampling(UN(Measure(), circ.n_qubits - 1), shots=100) # Add a measurement gate to register 1 and sample the circuit 100 times to obtain statistical results
res
```

```text
shots: 100
Keys: q3 q2 q1 q0│0.00     0.2         0.4         0.6         0.8         1.0
─────────────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
             0100│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                 │
{'0100': 100}
```

It should be noted that the reading order of the measurement result as a binary string should be $|q_0q_1q_2q_3\rangle$, so we get that the measurement result of register 1 is `0010`, the probability amplitude is 1, and the final state can accurately reflect the phase $\varphi$. But `0010` is a binary result, so we convert it back to decimal and divide by $2^n$ to get our final estimate: $\varphi=\frac{2}{2^4}=\frac{1}{8}$.

We can also find out the position of the amplitude maximum $a_{max}$ in the first register by the quantum state `qs` obtained from the circuit evolution, and then obtain the corresponding eigenbasis vector $|x\rangle$, where $x$ is divided by $2^t$ to be the estimated value of the phase.

```python
index = np.argmax(np.abs(qs))
print(bin(index)[2:])
```

```text
10100
```

It should be noted that `qs` corresponds to the final state of the entire quantum circuit, so the obtained `index` also includes the bits in the second register, and the $|x\rangle$ corresponding to $a_{max}$ in the final state of the first register cannot be directly obtained, and it is necessary to convert the `index` into binary and remove the bits corresponding to $q4$, and then the $|x\rangle$ of the first register is obtained.

```python
bit_string = bin(index)[2:].zfill(circ.n_qubits)[1:]        # Convert index to 01 string and remove q4
bit_string = bit_string[::-1]                               # Adjust the bit string order to q0q1q2q3
print(bit_string)
```

```text
0010
```

Convert binary back to decimal again to get our final estimate.

```python
# pylint: disable=W0104
theta_exp = int(bit_string, 2) / 2**n
theta_exp
```

```text
0.125
```

It can be seen that the estimated phase obtained is approximately equal to $\varphi$.

## Reference

[1] Michael A. Nielsen and Isaac L. Chuang. [Quantum computation and quantum information](https://www.cambridge.org/9781107002173)
