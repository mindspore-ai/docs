# S-parameter Simulation of Patch Antenna Based on Differentiable FDTD

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindelec/docs/source_en/AD_FDTD_forward.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

This tutorial introduces the method for solving electromagnetic positive problems provided by MindElec   based on device-to-device differentiable FDTD. The process of solving Maxwell's equations by the finite-difference time-domain (FDTD) method is equivalent to a recurrent convolutional network (RCNN). The device-to-device differentiable FDTD can be obtained by rewriting the update process with the differentiable operator of MindSpore. Compared with the data-driven black-box model, the solution process of the differentiable FDTD method strictly satisfies the constraints of Maxwell's equations, and the accuracy is comparable to that of traditional numerical algorithms.

> This example is for GPU processors and you can download the full sample code here:
> <https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/AD_FDTD/fdtd_forward>

## Maxwell's Equations

Active Maxwell's equations are the classical control equations for electromagnetic simulation, which are a set of partial differential equations describing the relationship between electric and magnetic fields and charge density and current density, in the following form:

$$
\nabla\times E=-\mu \dfrac{\partial H}{\partial t},
$$

$$
\nabla\times H=\epsilon \dfrac{\partial E}{\partial t}+\sigma E+ J(x, t)
$$

where $\epsilon,\mu,\sigma$ are the absolute permittivity, absolute magnetic permeability, and electrical conductivity of the medium, respectively. $J(x, t)$ is the excitation source in the electromagnetic simulation process, which is usually expressed in the form of a port pulse. The port in this case is a line port, which can be expressed as:

$$
J(x, t)=H(x - x_0)H(x_1 - x)g(t)
$$

where $x_0$ and $x_1$ are the starting and ending positions of the pilot port, respectively, $H(x)$ is the step function, and $g(t)$ is the functional expression of the pulse signal.

## Simulation Process

The specific flow of MindElec for antenna S parameter simulation is as follows:

1. Define antenna structure, excitation port location and type, sampling port.

2. Define the excitation source time domain waveform.

3. Building neural networks.

4. Solve and evaluate the results.

## S-parameter Simulation of Patch Invert_f Antenna

The case simulates the S-parameter of the patch invert_f antenna. The antenna structure is shown in the figure below.

![invert_f_structure](https://gitee.com/mindspore/docs/raw/master/docs/mindelec/docs/source_zh_cn/images/AD_FDTD/fdtd_forward/invert_f_structure.png)

### Import Dependencies

Import the modules and interfaces that this tutorial depends on:

```python
import os
import argparse
import numpy as np
from src import estimate_time_interval, compare_s
from src import CFSParameters, Gaussian
from src import Antenna, SParameterSolver
from src import GridHelper, UniformBrick, PECPlate, VoltageSource
from src import VoltageMonitor, CurrentMonitor
from src import full3d
```

### Defining the Excitation Source Time Domain Waveform

The time domain waveform of the excitation source in this case is a Gaussian pulse. FDTD uses the leap-frog scheme to update the electric and magnetic fields separately, and the excitation source in this case is a voltage source, so the time domain waveform value of the excitation source on the half time step should be calculated.

```python
def get_waveform_t(nt, dt, fmax):
    """
    Compute waveforms at time t.

    Args:
        nt (int): Number of time steps.
        dt (float): Time interval.
        fmax (float): Maximum freuqency of Gaussian wave

    Returns:
        waveform_t (Tensor, shape=(nt,)): Waveforms.
    """
    t = (np.arange(0, nt) + 0.5) * dt
    waveform = Gaussian(fmax)
    waveform_t = waveform(t)
    return waveform_t, t
```

### Defining the Antenna Structure, Excitation Port and Sampling Port

Users can customize the antenna structure, excitation port and sampling port on the grid according to the antenna design drawing. First, according to the split size, total antenna size, PML layer thickness, air layer thickness, the program automatically generates the FDTD `grid`. Then, user can define antenna structure, excitation port and sampling port on `grid` with the help of various components provided by the program according to the antenna design drawing, such as dielectric substrate (uniform dielectric block `UniformBrick`), metal patch (`PECPlate`), voltage source (`VoltageSource`), voltage sampling port (`VoltageMonitor`) and current sampling port (`CurrentMonitor`).

```python
def get_invert_f_antenna(air_buffers, npml):
    """ Get grid for IFA. """
    cell_lengths = (0.262e-3, 0.4e-3, 0.4e-3)
    obj_lengths = (0.787e-3, 40e-3, 40e-3)
    cell_numbers = (
        2 * npml + 2 * air_buffers[0] + int(obj_lengths[0] / cell_lengths[0]),
        2 * npml + 2 * air_buffers[1] + int(obj_lengths[1] / cell_lengths[1]),
        2 * npml + 2 * air_buffers[2] + int(obj_lengths[2] / cell_lengths[2]),
    )

    grid = GridHelper(cell_numbers, cell_lengths, origin=(
        npml + air_buffers[0] + int(obj_lengths[0] / cell_lengths[0]),
        npml + air_buffers[1],
        npml + air_buffers[2],
    ))

    # Define antenna
    grid[-3:0, 0:100, 0:100] = UniformBrick(epsr=2.2)
    grid[0, 0:71, 60:66] = PECPlate('x')
    grid[0, 40:71, 75:81] = PECPlate('x')
    grid[0, 65:71, 21:81] = PECPlate('x')
    grid[0, 52:58, 40:81] = PECPlate('x')
    grid[-3:0, 40, 75:81] = PECPlate('y')
    grid[-3, 0:40, 0:100] = PECPlate('x')

    # Define sources
    grid[-3:0, 0, 60:66] =\
        VoltageSource(amplitude=1., r=50., polarization='xp')

    # Define monitors
    grid[-3:0, 0, 61:66] = VoltageMonitor('xp')
    grid[-1, 0, 60:66] = CurrentMonitor('xp')
    return grid
```

It should be noted that when defining the antenna structure, excitation port location and sampling port location on the `grid`, users can specify the object location either directly by the grid number or by the spatial coordinates. However, users need to be aware that specifying object positions by spatial coordinates may introduce modeling errors. For example, users can mix grid numbers and spatial coordinates to define the antenna structure:

```python
    ...
    # Define antenna
    grid[-0.787e-3:0, 0:40e-3, 0:40e-3] = UniformBrick(epsr=2.2)
    grid[0, 0:28.4e-3, 24e-3:26.4e-3] = PECPlate('x')
    grid[0, 16e-3:28.4e-3, 30e-3:32.4e-3] = PECPlate('x')
    grid[0, 26e-3:28.4e-3, 8.4e-3:32.4e-3] = PECPlate('x')
    grid[0, 20.8e-3:23.2e-3, 16e-3:32.4e-3] = PECPlate('x')
    grid[-0.787e-3:0, 16e-3, 30e-3:32.4e-3] = PECPlate('y')
    grid[-0.787e-3, 0:16e-3, 0:40e-3] = PECPlate('x')
    ...
```

### Building Neural Network and Solving

Define the differentiable FDTD network, then define the S-parameter solver object `solver` and call the `solve` interface for solving.

```python
    # define fdtd network
    fdtd_net = full3d.ADFDTD(grid_helper.cell_numbers, grid_helper.cell_lengths,
                             nt, dt, ns, antenna, cpml)
    # define solver
    solver = SParameterSolver(fdtd_net)

    # solve
    _ = solver.solve(waveform_t)
```

### Solution

Define the sampling frequency and call the `eval` port to get the S parameter on the sampling frequency.

```python
    # sampling frequencies
    fs = np.linspace(0., fmax, 501, endpoint=True)

    # eval
    s_parameters = solver.eval(fs, t)
```

The comparison of the S-parameters calculated by the program with the results of reference is as follows:

![PIFA_S11](https://gitee.com/mindspore/docs/raw/master/docs/mindelec/docs/source_zh_cn/images/AD_FDTD/fdtd_forward/invert_f_s_parameters.png)

## S-parameter Simulation of Patch Microstrip Filter

This case simulates the S-parameters of the patch microstrip filter. The device structure is shown in the figure below.

![microstrip_filter_structure](https://gitee.com/mindspore/docs/raw/master/docs/mindelec/docs/source_zh_cn/images/AD_FDTD/fdtd_forward/microstrip_filter_structure.png)

### Import Dependencies

Import the modules and interfaces that this tutorial depends on:

```python
import os
import argparse
import numpy as np
from src import estimate_time_interval, compare_s
from src import CFSParameters, Gaussian
from src import Antenna, SParameterSolver
from src import GridHelper, UniformBrick, PECPlate, VoltageSource, Resistor
from src import VoltageMonitor, CurrentMonitor
from src import full3d
```

### Defining the Excitation Source Time Domain Waveform

The excitation source time domain waveform in this case is a Gaussian pulse. FDTD uses the leap-frog scheme to update the electric and magnetic fields separately, and the excitation source in this case is a voltage source, so the time domain waveform value of the excitation source on the half time step should be calculated.

```python
def get_waveform_t(nt, dt, fmax):
    """
    Compute waveforms at time t.

    Args:
        nt (int): Number of time steps.
        dt (float): Time interval.
        fmax (float): Maximum freuqency of Gaussian wave

    Returns:
        waveform_t (Tensor, shape=(nt,)): Waveforms.
    """
    t = (np.arange(0, nt) + 0.5) * dt
    waveform = Gaussian(fmax)
    waveform_t = waveform(t)
    return waveform_t, t
```

### Defining Filter Structure, Excitation Port and Sampling Port

Users can customize the filter structure, excitation port and sampling port on the grid according to the patch filter design drawing. First, according to the split size, total filter size, PML layer thickness, air layer thickness, the program automatically generates the FDTD `grid`. Then, user can define filter structure, excitation port and sampling port on `grid` with the help of various components provided by the program according to the filter design drawing, such as dielectric substrate (uniform dielectric block `UniformBrick`), metal patch (`PECPlate`), voltage source (`VoltageSource`), resistor (`Resistor`), voltage sampling port (`VoltageMonitor`) and current sampling port (`CurrentMonitor`).

```python
def get_microstrip_filter(air_buffers, npml):
    """ microstrip filter """
    cell_lengths = (0.4064e-3, 0.4233e-3, 0.265e-3)
    obj_lengths = (50 * cell_lengths[0],
                   46 * cell_lengths[1],
                   3 * cell_lengths[2])
    cell_numbers = (
        2 * npml + 2 * air_buffers[0] + int(obj_lengths[0] / cell_lengths[0]),
        2 * npml + 2 * air_buffers[1] + int(obj_lengths[1] / cell_lengths[1]),
        2 * npml + 2 * air_buffers[2] + int(obj_lengths[2] / cell_lengths[2]),
    )

    grid = GridHelper(cell_numbers, cell_lengths, origin=(
        npml + air_buffers[0],
        npml + air_buffers[1],
        npml + air_buffers[2],
    ))

    # Define antenna
    grid[0:50, 0:46, 0:3] = UniformBrick(epsr=2.2)
    grid[14:20, 0:20, 3] = PECPlate('z')
    grid[30:36, 26:46, 3] = PECPlate('z')
    grid[0:50, 20:26, 3] = PECPlate('z')
    grid[0:50, 0:46, 0] = PECPlate('z')

    # Define sources
    grid[14:20, 0, 0:3] = VoltageSource(1., 50., 'zp')

    # Define load
    grid[30:36, 46, 0:3] = Resistor(50., 'z')

    # Define monitors
    grid[14:20, 10, 0:3] = VoltageMonitor('zp')
    grid[14:20, 10, 3] = CurrentMonitor('yp')
    grid[30:36, 36, 0:3] = VoltageMonitor('zp')
    grid[30:36, 36, 3] = CurrentMonitor('yn')

    return grid
```

It should be noted that the patch filter in this case is a two-port device, and only S11 and S21 parameters are simulated in this case. In order to calculate the multi-port S-parameters, the voltage sampling port and current sampling port need to be defined separately at each port.

### Building Neural Network and Solve

Define the differentiable FDTD network, then define the S-parameter solver object `solver` and call the `solve` interface for solving.

```python
    # define fdtd network
    fdtd_net = full3d.ADFDTD(grid_helper.cell_numbers, grid_helper.cell_lengths,
                             nt, dt, ns, antenna, cpml)
    # define solver
    solver = SParameterSolver(fdtd_net)

    # solve
    _ = solver.solve(waveform_t)
```

### Solution

Define the sampling frequency and call the `eval` port to get the S parameter on the sampling frequency:

```python
    # sampling frequencies
    fs = np.linspace(0., fmax, 1001, endpoint=True)

    # eval
    s_parameters = solver.eval(fs, t)
```

The comparison of the S-parameters calculated by the program with the results of reference is as follows:

![Patch_Microstrip_Filter_S](https://gitee.com/mindspore/docs/raw/master/docs/mindelec/docs/source_zh_cn/images/AD_FDTD/fdtd_forward/microstrip_filter_s_parameters.png)
