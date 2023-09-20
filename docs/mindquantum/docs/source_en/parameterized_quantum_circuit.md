# Variational Quantum Circuit

Translator: [Wei_zz](https://gitee.com/wei-zz)

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindquantum/docs/source_en/parameterized_quantum_circuit.md)

## Summary

Variational quantum circuit(VQC), is an approach for Quantum Machine Learning. The MindSpore Quantum (mixing framework of quantum and classic machine learning) can process variational quantum circuit and get the derivation of all observation to every parameter respectively by auto differentiating the circuit using quantum neural network.
The process of constructing a quantum circuit and circuit evolution by parameterized simulator operators is as follows:

- Initialize a quantum circuit.

- According to requirements, add parameterized quantum gates or non-parameterized quantum gates to the circuit.

- Process gradient solution or state of evolution by PQC simulator operators.

## Preparing Environment

Import required modules.

```python
import numpy as np                                          # Import numpy library and abbreviate to np
from mindquantum.core.gates import X, Y, Z, H, RX, RY, RZ   # Import the quantum gate H, X, Y, Z, RX, RY, RZ
```

Note:

1. numpy is a powerful Python library for performing calculations on multidimensional arrays, supporting a large number of dimensional arrays and matrices, in addition to providing a large library of mathematical functions for arrays.

2. mindquantum, a hybrid quantum-classical computing framework, supports the training and inference of a wide range of quantum neural networks.

3. The quantum gates to be executed in the built quantum lines need to be imported from the mindquantum.core module.

## Quantum Gate

A quantum gate is the basic logic unit to operate quantum bit. For a classic circuit, any logic circuit can consist of some basic logic gates, similarly, any quantum circuit can consist of some basic quantum gates like gates or C-NOT gates acting on a single bit. Commonly used quantum gates include $\text{X}$ gates, $\text{Y}$ gates, $\text{Z}$ gates, $\text{Hadamard}$ gates, $\text{CNOT}$ gates and some revolving gates.

In general, quantum gates can be classified into parametric and non-parametric quantum gates. For example, the non-parametric quantum gates are `X` gate, `Y` gate, `Z` gate, `Hadamard` gate (`H` gate) and `CNOT` gate, which have the following matrix forms, respectively:

$$
\text{X}=
\left(
    \begin{matrix}
        0&1\\
        1&0
    \end{matrix}
\right),
\text{Y}=
\left(
    \begin{matrix}
        0&-i\\
        i&0
    \end{matrix}
\right),
\text{Z}=
\left(
    \begin{matrix}
        1&0\\
        0&-1
    \end{matrix}
\right),
\text{H}=\frac{1}{\sqrt{2}}
\left(
    \begin{matrix}
        1&1\\
        1&-1
    \end{matrix}
\right),
\text{CNOT}=
\left(
    \begin{matrix}
        1&0&0&0\\
        0&1&0&0\\
        0&0&0&1\\
        0&0&1&0
    \end{matrix}
\right).$$

Print the matrix form of the above quantum gates separately and we can get:

```python
print('Gate name:', X)
X.matrix()
```

```text
Gate name: X
```

```text
array([[0, 1],
       [1, 0]])
```

```python
print('Gate name:', Y)
Y.matrix()
```

```text
Gate name: Y
```

```text
array([[ 0.+0.j, -0.-1.j],
       [ 0.+1.j,  0.+0.j]])
```

Note: For each item in the matrix, "0." on the left indicates the real part of the fractional form (floating point number) (if the real part is negative, "-" is displayed before the fractional part, otherwise it is non-negative by default), "0. " on the right indicates the imaginary part of the decimal form (floating point number) (if the imaginary part is negative, "-" is displayed before the decimal, otherwise "+" is displayed), and j indicates the imaginary unit $i$.

```python
print('Gate name:', Z)
Z.matrix()
```

```text
Gate name: Z
```

```text
array([[ 1,  0],
       [ 0, -1]])
```

```python
print('Gate name:', H)
H.matrix()
```

```text
Gate name: H
```

```text
array([[ 0.70710678,  0.70710678],
       [ 0.70710678, -0.70710678]])
```

For `CNOT` gates, they are essentially Controlled-`X` gates, so in MindSpore Quantum, if we need to execute a `CNOT` gate, we only need to set the control bits and the target bits of the `X` gate (in fact, for any quantum gate we can set the control bits and the target bits of the desired quantum gate operation). For example:

```python
cnot = X.on(0, 1)   # X gate acts on bit 0 quantum bit and is controlled by bit 1 quantum bit
print(cnot)
```

```text
X(0 <-: 1)
```

Note:

1. The `X(1 <-: 0)` denotes that the bit 0 quantum bit is the target bit, the bit 1 quantum bit is the control bit, and the bit 0 quantum bit is controlled by the bit 1 quantum bit. Perform `X`-gate operation on bit 0 quantum bit if bit 1 is 1, otherwise no operation is performed.

The above describes some quantum gates without parameters. Next, we will introduce some quantum gates with parameters (such as the revolving gate `RX` gate, `RY` gate and `RZ` gate), which can be obtained by giving certain definite values of the rotation angle $\theta$ that act differently. In addition, these quantum gates with parameters are important building blocks for the subsequent construction of quantum neural networks.

For example, the `RX` gate, `RY` gate and `RZ` gate have the following matrix form:

$$\text{RX}(\theta)= e^{-\frac{i\theta X}{2}}=\cos\left(\frac{\theta}{2}\right)\cdot I-i\sin\left(\frac{\theta}{2}\right)\cdot X=
\left(
    \begin{matrix}
        \cos\left(\frac{\theta}{2}\right)&-i\sin\left(\frac{\theta}{2}\right)\\
        -i\sin\left(\frac{\theta}{2}\right)&\cos\left(\frac{\theta}{2}\right)
    \end{matrix}
\right),$$

$$\text{RY}(\theta)= e^{-\frac{i\theta Y}{2}}=\cos\left(\frac{\theta}{2}\right)\cdot I-i\sin\left(\frac{\theta}{2}\right)\cdot Y=
\left(
    \begin{matrix}
        \cos\left(\frac{\theta}{2}\right)&-\sin\left(\frac{\theta}{2}\right)\\
        \sin\left(\frac{\theta}{2}\right)&\cos\left(\frac{\theta}{2}\right)
    \end{matrix}
\right),$$

$$\text{RZ}(\theta)= e^{-\frac{i\theta Z}{2}}=\cos\left(\frac{\theta}{2}\right)\cdot I-i\sin\left(\frac{\theta}{2}\right)\cdot Z=
\left(
    \begin{matrix}
        e^{-\frac{i\theta}{2}}&0\\
        0&e^{\frac{i\theta}{2}}
    \end{matrix}
\right).$$

We make $\theta$ be $0, \frac{\pi}{2}$ and $\pi$, respectively, and then print the matrix forms of $\text{RX}(0)$ gates, $\text{RY}(\frac{\pi}{2}$) gates and $\text{RZ}(\pi)$ gates. And we can obtain:

```python
rx = RX('theta')
print('Gate name:', rx)
rx.matrix({'theta': 0})   # Assign a value of theta to 0
```

```text
Gate name: RX(theta)
```

```text
array([[1.+0.j, 0.+0.j],
       [0.+0.j, 1.+0.j]])
```

When $\theta=0$, at this point the $\text{RX}(0)$ gate is the familiar `I` gate.

```python
ry = RY('theta')
print('Gate name:', ry)
ry.matrix({'theta': np.pi/2})   # pi needs to be imported from np, assigning the value of theta to pi/2
```

```text
Gate name: RY(theta)
```

```text
array([[ 0.70710678+0.j, -0.70710678+0.j],
       [ 0.70710678+0.j,  0.70710678+0.j]])
```

When $\theta=\frac{\pi}{2}$, at this point the $\text{RY}(\frac{\pi}{2})$ gate is the familiar `H` gate.

```python
rz = RZ('theta')
print('Gate name:', rz)
np.round(rz.matrix({'theta': np.pi}))   # The value of pi is assigned to theta, and because of the problem of imprecise floating point numbers in computers, the rounded value of the floating point number is returned by the function np.round.
```

```text
Gate name: RZ(theta)
```

```text
array([[0.-1.j, 0.+0.j],
       [0.+0.j, 0.+1.j]])
```

When $\theta=\pi$, at this point the $\text{RZ}(\pi)$ gate is the familiar `Z` gate. (differing by one global phase $-i$)

## Quantum Circuit

Quantum circuit is a structure used to effectively organize various quantum logic gates. We can initialize the quantum circuit through the list of quantum gates, or expand the quantum circuit by adding a quantum gate or circuit through addition(`+`), and multiplying by an integer through multiplication(`*`). Here we will construct the following quantum circuit and print the relevant information of the quantum circuit. In the following figure, `q0`, `q1` and `q2` represent three qubits respectively. The quantum circuit consists of three quantum gates, namely the Hadamard gate acting on `q0` bit, the $CNOT$ gate acting on `q1` bit and controlled by `q0` bit, and the $\text{RY}$ revolving gate acting on `q2` bit.

![quantum circuit](./images/quantum_circuit.png)

The construction of a quantum line can be accomplished quickly by adding quantum gates acting on different quantum bits in the quantum line.

```python
from mindquantum.core.circuit import Circuit     # Import Circuit module for building quantum lines

encoder = Circuit()                              # Initialize quantum lines
encoder += H.on(0)                               # H-gate acts at bit 0 quantum bit
encoder += X.on(1, 0)                            # The X gate acts on the bit 1 quantum bit and is controlled by the bit 0 quantum bit
encoder += RY('theta').on(2)                     # RY(theta) gate acts on the bit 2 quantum bit

print(encoder)                                   # Print Encoder
encoder.summary()                                # Summarize Encoder quantum lines
```

```text
q0: ──────H────────●──
                   │
q1: ───────────────X──

q2: ──RY(theta)───────
=========Circuit Summary=========
|Total number of gates  : 3.    |
|Parameter gates        : 1.    |
|with 1 parameters are  : theta.|
|Number qubit of circuit: 3     |
=================================
```

In the Jupyter Notebook environment, you can call the `.svg()` interface of the quantum line to draw the image format of the quantum line. Calling the `.svg().to_file(filename='circuit.svg')` interface of the quantum line saves the svg image of the quantum line to local.

```python
encoder.svg()
```

From the Summary of Encoder, we can see that the quantum line consists of three quantum gates, one of which is a quantum gate with parameters and has the parameter theta, and the number of quantum bits regulated by this quantum line is 3.

Therefore, we can build the corresponding quantum line according to the problem we need to solve. Go and build your first quantum line!

```python
from mindquantum.utils.show_info import InfoTable

InfoTable('mindquantum', 'scipy', 'numpy')
```

|       Software |                  Version |
| -------------: | -----------------------: |
|    mindquantum |                    0.9.0 |
|          scipy |                   1.10.1 |
|          numpy |                   1.23.5 |
|         System |                     Info |
|         Python |                   3.8.17 |
|             OS |            Windows AMD64 |
|         Memory |                  8.39 GB |
| CPU Max Thread |                        8 |
|           Date | Mon Sep 18 12:00:52 2023 |
