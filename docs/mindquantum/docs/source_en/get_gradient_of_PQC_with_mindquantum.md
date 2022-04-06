# Advanced gradient calculation of variational quantum circuits

Translator: [unseenme](https://gitee.com/unseenme)

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindquantum/docs/source_en/get_gradient_of_PQC_with_mindquantum.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

In MindQuantum, we can obtain the gradient of a variable quantum circuit by the get_expectation_with_grad method of the Simulator class. In this tutorial, we will further introduce other functions of this method to help you achieve more advanced usage methods.

## Model introduction

The get_expectation_with_grad method is mainly used to calculate the value of the following expression and the gradient of the parameters in the circuit.

$$E(\boldsymbol{\theta})=\left<\varphi\right|U^\dagger_l(\boldsymbol{\theta})HU_r(\boldsymbol{\theta})\left|\psi\right>$$

The interface of this method is defined as follows

```python
Simulator.get_expectation_with_grad(
    hams,
    circ_right,
    circ_left=None,
    simulator_left=None,
    encoder_params_name=None,
    ansatz_params_name=None,
    parallel_worker=None
)
```

Then, we will introduce the meaning of each parameter one by one.

1. hams. The type of Hamiltonian required for the Hamiltonian in the circuit is Hamiltonian in mindquantum, or a list array containing multiple Hamiltonians. For the latter case, the framework will calculate the expected values of the circuit with respect to all Hamiltonians and the gradient of each expected value with respect to the circuit parameters at the same time.
2. circ_right. It is the $U_r(\boldsymbol{\theta})$ in the formula.
3. circ_left. It is the $U_l^\dagger(\boldsymbol{theta})$ in the formula. When it is the default value None, circ_left and circ_right are the same circuits.
4. simulator_left. It is the simulator that contains the $\left|\varphi\right>$ in the formula. You can set the state of the emulator to the state you need by the emulator's set_qs, apply_gate or apply_circuit methods. When it is the default value None, $\left|\varphi\right>=\left|\psi\right>$, and $\left|\psi\right>$ is the quantum state contained in the current simulator.
5. encoder_params_name. It indicates which quantum gates with parameters in $U_l(\boldsymbol{\theta})$ and $U_r(\boldsymbol{\theta})$ are encoder. In the quantum neural network, the parameter corresponding to the encoder is the number that the user needs to input, and does not participate in the training. When it is the default value None, there is no encoder in the circuit.
6. ansatz_params_name. It indicates which quantum gates with parameters in $U_l(\boldsymbol{\theta})$ and $U_r(\boldsymbol{\theta})$ are ansatz. In the quantum neural network, the parameters corresponding to ansatz are initialized by the system or the user, and then updated by the system according to the gradient to participate in the training. When it is the default value None, all parameter gates in the circuit are ansatz.
7. parallel_worker. When the hams contains multiple Hamiltonians or the input of the encoder contains multiple sample points, MindQuantum will reasonably perform parallel operations based on this integer as a reference.

## Expected values of multiple Hamiltonians at multiple input sample points

In this task, we want to calculate the expected value of the Hamiltonian $Z_0, X_0, Y_0$ for the following quantum circuit when $\alpha=\text{arctan}(\sqrt{2}), \pi/2$.

```python
import numpy as np
from mindquantum import QubitOperator
from mindquantum import Simulator
from mindquantum import Circuit, TimeEvolution, Hamiltonian, H

# Define the axis of rotation in Hilbert space
axis = QubitOperator('Y0', 1 / np.sqrt(2)) + QubitOperator('X0', -1 / np.sqrt(2))
# Define the order of the trotter decomposition
trotter_order = 4
# Trotter decomposition of rotation using TimeEvolution
encoder = TimeEvolution(axis, {'alpha': 0.5 / trotter_order}).circuit * trotter_order
encoder
```

```text
q0: ──RY(0.176776695296637*alpha)────RX(-0.176776695296637*alpha)────RY(0.176776695296637*alpha)────RX(-0.176776695296637*alpha)────RY(0.176776695296637*alpha)────RX(-0.176776695296637*alpha)────RY(0.176776695296637*alpha)────RX(-0.176776695296637*alpha)──
```

Next, define the Hamiltonian of the expected value to be calculated:

```python
# Define the Hamiltonian Set
hams = [Hamiltonian(QubitOperator('X0')), Hamiltonian(QubitOperator('Y0')), Hamiltonian(QubitOperator('Z0'))]
hams
```

```text
[1.0 [X0] , 1.0 [Y0] , 1.0 [Z0] ]
```

Get the operator for the expected value and gradient:

```python
grad_ops = Simulator('projectq', 1).get_expectation_with_grad(hams, encoder, encoder_params_name=encoder.params_name, parallel_worker=6)
grad_ops
```

```text
<mindquantum.simulator.simulator.GradOpsWrapper at 0x7f03a4a60430>
```

Define the value of alpha:

```python
alpha = np.array([[np.arctan(np.sqrt(2))], [np.pi/2]])
alpha
```

```text
array([[0.95531662],
       [1.57079633]])
```

```python
f, g = grad_ops(alpha)
print(f)
print(f'shape: {f.shape}')
print('\n')
print(g)
print(f'shape: {g.shape}')
```

```text
[[0.59389047+0.j 0.55828416+0.j 0.57932107+0.j]
 [0.77269648+0.j 0.63465887+0.j 0.01217645+0.j]]
shape: (2, 3)


[[[ 0.45790207+0.j]
  [ 0.35200884+0.j]
  [-0.80864423+0.j]]

 [[ 0.10989151+0.j]
  [-0.11512098+0.j]
  [-0.9732094 +0.j]]]
shape: (2, 3, 1)
```

### Result analysis

According to the above results, we can see that the dimension of the expected value f is (2, 3). It is not difficult to find that each row of f corresponds to a different expected value of the Hamiltonian for each sample point, and each column of f corresponds to the expected value of each Hamiltonian under different samples. For the gradient g, we also have similar conclusions, except that the last dimension represents different circuit parameters.

## Calculating the inner product of different quantum states

According to the model, we only need to set the Hamiltonian as the unit operator and $U_l(\boldsymbol{\theta})$ as an empty quantum circuit, then we can use $U_r(\boldsymbol{\theta})$ to rotate $\left|\psi\right>$ to $\left|\varphi\right>$, which requires calculating the inner product between $\left|\varphi\right>$ and the rotated quantum state.

Here, we compute the inner product between the quantum state after the evolution of the zero state by the following quantum circuit and the uniform superposition.

```python
circuit = Circuit().ry('a', 0).rz('b', 0).ry('c', 0)
circuit
```

```text
q0: ──RY(a)────RZ(b)────RY(c)──
```

Prepare a simulator containing uniform superposition states:

```python
sim_l = Simulator('projectq', 1)
sim_l.apply_gate(H.on(0))
sim_l
```

```text
projectq simulator with 1 qubit.
Current quantum state:
√2/2¦0⟩
√2/2¦1⟩
```

Prepare the unit Hamiltonian:

```python
ham = Hamiltonian(QubitOperator(""))
```

Get the inner product and gradient computation operators:

```python
grad_ops = Simulator('projectq', 1).get_expectation_with_grad(ham, circuit, Circuit(), simulator_left=sim_l)
```

Choose the appropriate parameters:

```python
rot_angle = np.array([7.902762e-01, 2.139225e-04, 7.795934e-01])
```

```python
f, g = grad_ops(rot_angle)
print(f)
print('\n')
print(g)
```

```text
[[0.99999989-7.52279618e-05j]]


[[[ 2.31681689e-04+3.80179652e-05j -5.34806192e-05-3.51659884e-01j
    2.31681689e-04-3.80179652e-05j]]]
```

### Result analysis

According to the calculation results, we find that the inner product of the last two states is close to 1, indicating that we can prepare a uniform superposition state with high fidelity by the above circuit.

```python
print(circuit.get_qs(pr=rot_angle, ket=True))
```

```text
(0.7074343486186319-0.00010695972396782116j)¦0⟩
(0.7067790538448511+√5/3906250j)¦1⟩
```

To find out more about MindQuantum's API, please click: https://mindspore.cn/mindquantum/
