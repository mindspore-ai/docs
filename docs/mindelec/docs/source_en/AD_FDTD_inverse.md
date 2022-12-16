# Device-to-device Differentiable FDTD for Solving Electromagnetic Inverse Scattering

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindelec/docs/source_en/AD_FDTD_inverse.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Overview

This tutorial introduces the method for solving electromagnetic inverse problems provided by MindElec   based on device-to-device differentiable FDTD. The process of solving Maxwell's equations by the finite-difference time-domain (FDTD) method is equivalent to a recurrent convolutional network (RCNN). The device-to-device differentiable FDTD can be obtained by rewriting the update process with the differentiable operator of MindSpore. Compared with the data-driven black-box model, the solution process of the differentiable FDTD method strictly satisfies the constraints of Maxwell's equations. Using MindSpore gradient-ased optimizer, differentiable FDTD can solve various EM inverse problems.

> This example is for GPU processors and you can download the full sample code here:
> <https://gitee.com/mindspore/mindscience/tree/master/MindElec/examples/AD_FDTD/fdtd_forward>

## Maxwell's Equations

Active Maxwell's equations are the classical control equations for electromagnetic simulation, which are a set of partial differential equations describing the relationship between electric and magnetic fields and charge density and current density, in the following form:

$$
\nabla\times E=-\mu \dfrac{\partial H}{\partial t},
$$

$$
\nabla\times H=\epsilon \dfrac{\partial E}{\partial t} + J(x, t)
$$

where $\epsilon,\mu$ are the absolute permittivity and absolute magnetic permeability respectively. $J(x, t)$ is the excitation source in the electromagnetic simulation process, which is usually expressed in the form of a port pulse. This is approximated in a mathematical sense as the point source expressed in Dirac functional form, which can be expressed as:

$$
J(x, t)=\delta(x - x_0)g(t)
$$

where $x_0$ is the excitation source position and $g(t)$ is the functional expression of the pulse signal.

## Problem Description

This case solves the electromagnetic inverse scattering in two-dimensional TM mode. The two plastids are located inside the rectangular area. Four excitation sources (red triangles) and eight observation points (green dots) are set on the outside of the solution area, as shown in the following figure:

![Solution Area Setting](https://gitee.com/mindspore/docs/raw/master/docs/mindelec/docs/source_zh_cn/images/AD_FDTD/fdtd_inverse/inversion_problem_setup.png)

The process that MindElec solves the problem is as follows:

1. Obtain the time domain electric field values at the observation points with traditional numerical methods to create training datasets and evaluation data.

2. Define the excitation source location, the observation point location and the solution area.

3. Define the time domain waveform of the excitation source.

4. Construct differentiable FDTD networks.

5. Train the network and evaluate the solution results.

## Import Dependencies

Import the modules and interfaces that this tutorial depends on:

```python
import os
import argparse
import numpy as np
from mindspore import nn
import matplotlib.pyplot as plt
from src import transverse_magnetic, EMInverseSolver
from src import zeros, tensor, vstack, elu
from src import Gaussian, CFSParameters, estimate_time_interval
from src import BaseTopologyDesigner
```

## Loading the Dataset

Load the time-domain electric field values at each observation point calculated by conventional numerical methods (e.g., FDTD), as well as the true value of the relative dielectric constant.

```python
def load_labels(nt, dataset_dir):
    """
    Load labels of Ez fields and epsr.

    Args:
        nt (int): Number of time steps.
        dataset_dir (str): Dataset directory.

    Returns:
        field_labels (Tensor, shape=(nt, ns, nr)): Ez at receivers.
        epsr_labels (Tensor, shape=(nx, ny)): Ground truth for epsr.
    """

    field_label_path = os.path.join(dataset_dir, 'ez_labels.npy')
    field_labels = tensor(np.load(field_label_path))[:nt]

    epsr_label_path = os.path.join(dataset_dir, 'epsr_labels.npy')
    epsr_labels = tensor(np.load(epsr_label_path))

    return field_labels, epsr_labels
```

## Defining the Excitation Source Location, Observation Point Location and Solution Area

Inheriting the `BaseTopologyDesiger` class, users can quickly define electromagnetic inverse scattering problems. Users can set the solution area, excitation source location and observation point location in the member functions `generate_object`, `update_sources` and `get_outputs_at_each_step`, respectively. Taking this problem as an example, we define the electromagnetic inverse scattering problem as follows.

```python
class InverseDomain(BaseTopologyDesigner):
    """
    InverseDomain with customized mapping and source locations for user-defined problems.
    """

    def generate_object(self, rho):
        """Generate material tensors.

        Args:
            rho (Parameter): Parameters to be optimized in the inversion domain.

        Returns:
            epsr (Tensor, shape=(self.cell_nunbers)): Relative permittivity in the whole domain.
            sige (Tensor, shape=(self.cell_nunbers)): Conductivity in the whole domain.
        """
        # generate background material tensors
        epsr = self.background_epsr * self.grid
        sige = self.background_sige * self.grid
        # ---------------------------------------------
        # Customized Differentiable Mapping
        # ---------------------------------------------
        epsr[30:70, 30:70] = self.background_epsr + elu(rho, alpha=1e-2)
        return epsr, sige

    def update_sources(self, *args):
        """
        Set locations of sources.

        Args:
            *args: arguments

        Returns:
            jz (Tensor, shape=(ns, 1, nx+1, ny+1)): Jz tensor.
        """
        sources, _, waveform, _ = args
        jz = sources[0]
        jz[0, :, 20, 50] = waveform
        jz[1, :, 50, 20] = waveform
        jz[2, :, 80, 50] = waveform
        jz[3, :, 50, 80] = waveform
        return jz

    def get_outputs_at_each_step(self, *args):
        """Compute output each step.

        Args:
            *args: arguments

        Returns:
            rx (Tensor, shape=(ns, nr)): Ez fields at receivers.
        """
        ez, _, _ = args[0]
        rx = [
            ez[:, 0, 25, 25],
            ez[:, 0, 25, 50],
            ez[:, 0, 25, 75],
            ez[:, 0, 50, 25],
            ez[:, 0, 50, 75],
            ez[:, 0, 75, 25],
            ez[:, 0, 75, 50],
            ez[:, 0, 75, 75],
        ]
        return vstack(rx)
```

It should be noted that when mapping the variable to be optimized `rho`, to the relative permittivity `epsr` in `generate_object`, choosing the right mapping relationship can greatly speed up the solver convergence. The mapping relationship used in this case is `background_epsr + elu(rho, alpha=1e-2)`, which guarantees that no unphysical dielectric constant will appear during the solution.

## Defining the Excitation Source Time Domain Waveform

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
        waveform_t (Tensor, shape=(nt, ns, nr)): Waveforms.
    """
    t = (np.arange(0, nt) + 0.5) * dt
    waveform = Gaussian(fmax)
    waveform_t = waveform(t)
    return waveform_t
```

## Constructing the Differentiable FDTD Network

This case solves the electromagnetic inverse scattering problem in two-dimensional TM mode. The process of solving Maxwell's equations by the finite-difference in time domain (FDTD) method is equivalent to a recurrent convolutional network (RCNN). When the CFS-PML is used to truncate the infinite region, the update process of the $n$th time step of the two-dimensional FDTD in TM mode is as follows:

![FDTD Time-step Update Process](https://gitee.com/mindspore/docs/raw/master/docs/mindelec/docs/source_zh_cn/images/AD_FDTD/fdtd_inverse/FDTD_RCNN_Update_TM_Mode.png)

The device-to-device differentiable FDTD is obtained by rewriting the update process of the FDTD with the differentiable operator of MindSpore. The computation process within each time step is as follows:

```python
class FDTDLayer(nn.Cell):
    """
    One-step 2D TM-Mode FDTD.

    Args:
        cell_lengths (tuple): Lengths of Yee cells.
        cpmlx_e (Tensor): Updating coefficients for electric fields in the x-direction CPML.
        cpmlx_m (Tensor): Updating coefficients for magnetic fields in the x-direction CPML.
        cpmly_e (Tensor): Updating coefficients for electric fields in the y-direction CPML.
        cpmly_m (Tensor): Updating coefficients for magnetic fields in the y-direction CPML.
    """

    def __init__(self,
                 cell_lengths,
                 cpmlx_e, cpmlx_m,
                 cpmly_e, cpmly_m,
                 ):
        super(FDTDLayer, self).__init__()
        dx = cell_lengths[0]
        dy = cell_lengths[1]
        self.cpmlx_e = cpmlx_e
        self.cpmlx_m = cpmlx_m
        self.cpmly_e = cpmly_e
        self.cpmly_m = cpmly_m
        # operators
        self.dx_oper = ops.Conv2D(out_channel=1, kernel_size=(2, 1))
        self.dy_oper = ops.Conv2D(out_channel=1, kernel_size=(1, 2))
        self.dx_wghts = tensor([-1., 1.]).reshape((1, 1, 2, 1)) / dx
        self.dy_wghts = tensor([-1., 1.]).reshape((1, 1, 1, 2)) / dy
        self.pad_x = ops.Pad(paddings=((0, 0), (0, 0), (1, 1), (0, 0)))
        self.pad_y = ops.Pad(paddings=((0, 0), (0, 0), (0, 0), (1, 1)))

    def construct(self, jz_t, ez, hx, hy, pezx, pezy, phxy, phyx,
                  ceze, cezh, chxh, chxe, chyh, chye):
        """One-step forward propagation

        Args:
            jz_t (Tensor): Source at time t + 0.5 * dt.
            ez, hx, hy (Tensor): Ez, Hx, Hy fields.
            pezx, pezy, phxy, phyx (Tensor): CPML auxiliary fields.
            ceze, cezh (Tensor): Updating coefficients for Ez fields.
            chxh, chxe (Tensor): Updating coefficients for Hx fields.
            chyh, chye (Tensor): Updating coefficients for Hy fields.

        Returns:
            hidden_states (tuple)
        """
        # -------------------------------------------------
        # Step 1: Update H's at n+1/2 step
        # -------------------------------------------------
        # compute curl E
        dezdx = self.dx_oper(ez, self.dx_wghts) / self.cpmlx_m[2]
        dezdy = self.dy_oper(ez, self.dy_wghts) / self.cpmly_m[2]

        # update auxiliary fields
        phyx = self.cpmlx_m[0] * phyx + self.cpmlx_m[1] * dezdx
        phxy = self.cpmly_m[0] * phxy + self.cpmly_m[1] * dezdy

        # update H
        hx = chxh * hx - chxe * (dezdy + phxy)
        hy = chyh * hy + chye * (dezdx + phyx)

        # -------------------------------------------------
        # Step 2: Update E's at n+1 step
        # -------------------------------------------------
        # compute curl H
        dhydx = self.pad_x(self.dx_oper(hy, self.dx_wghts)) / self.cpmlx_e[2]
        dhxdy = self.pad_y(self.dy_oper(hx, self.dy_wghts)) / self.cpmly_e[2]

        # update auxiliary fields
        pezx = self.cpmlx_e[0] * pezx + self.cpmlx_e[1] * dhydx
        pezy = self.cpmly_e[0] * pezy + self.cpmly_e[1] * dhxdy

        # update E
        ez = ceze * ez + cezh * ((dhydx + pezx) - (dhxdy + pezy) - jz_t)

        hidden_states = (ez, hx, hy, pezx, pezy, phxy, phyx)
        return hidden_states
```

The device-to-device differentiable FDTD network is as follows:

```python
class ADFDTD(nn.Cell):
    """2D TM-Mode Differentiable FDTD Network.

    Args:
        cell_numbers (tuple): Number of Yee cells in (x, y) directions.
        cell_lengths (tuple): Lengths of Yee cells.
        nt (int): Number of time steps.
        dt (float): Time interval.
        ns (int): Number of sources.
        designer (BaseTopologyDesigner): Customized Topology designer.
        cfs_pml (CFSParameters): CFS parameter class.
        init_weights (Tensor): Initial weights.

    Returns:
        outputs (Tensor): Customized outputs.
    """

    def __init__(self,
                 cell_numbers,
                 cell_lengths,
                 nt, dt,
                 ns,
                 designer,
                 cfs_pml,
                 init_weights,
                 ):
        super(ADFDTD, self).__init__()

        self.nx = cell_numbers[0]
        self.ny = cell_numbers[1]
        self.dx = cell_lengths[0]
        self.dy = cell_lengths[1]
        self.nt = nt
        self.ns = ns
        self.dt = tensor(dt)

        self.designer = designer
        self.cfs_pml = cfs_pml
        self.rho = ms.Parameter(
            init_weights) if init_weights is not None else None

        self.mur = tensor(1.)
        self.sigm = tensor(0.)

        if self.cfs_pml is not None:
            # CFS-PML Coefficients
            cpmlx_e, cpmlx_m = self.cfs_pml.get_update_coefficients(
                self.nx, self.dx, self.dt, self.designer.background_epsr.asnumpy())
            cpmly_e, cpmly_m = self.cfs_pml.get_update_coefficients(
                self.ny, self.dy, self.dt, self.designer.background_epsr.asnumpy())

            cpmlx_e = tensor(cpmlx_e.reshape((3, 1, 1, -1, 1)))
            cpmlx_m = tensor(cpmlx_m.reshape((3, 1, 1, -1, 1)))
            cpmly_e = tensor(cpmly_e.reshape((3, 1, 1, 1, -1)))
            cpmly_m = tensor(cpmly_m.reshape((3, 1, 1, 1, -1)))

        else:
            # PEC boundary
            cpmlx_e = cpmlx_m = tensor([0., 0., 1.]).reshape((3, 1))
            cpmly_e = cpmly_m = tensor([0., 0., 1.]).reshape((3, 1))

        # FDTD layer
        self.fdtd_layer = FDTDLayer(
            cell_lengths, cpmlx_e, cpmlx_m, cpmly_e, cpmly_m)

        # auxiliary variables
        self.dte = tensor(dt / epsilon0)
        self.dtm = tensor(dt / mu0)

        # material parameters smoother
        self.smooth_kernel = 0.25 * ones((1, 1, 2, 2))
        self.smooth_oper = ops.Conv2D(
            out_channel=1, kernel_size=2, pad_mode='pad', pad=1)

    def construct(self, waveform_t):
        """
        ADFDTD-based forward propagation.

        Args:
            waveform_t (Tensor, shape=(nt,)): Time-domain waveforms.

        Returns:
            outputs (Tensor): Customized outputs.
        """
        # ----------------------------------------
        # Initialization
        # ----------------------------------------
        # constants
        ns, nt, nx, ny = self.ns, self.nt, self.nx, self.ny
        dt = self.dt

        # material grid
        epsr, sige = self.designer.generate_object(self.rho)

        # delectric smoothing
        epsrz = self.smooth_oper(epsr[None, None], self.smooth_kernel)
        sigez = self.smooth_oper(sige[None, None], self.smooth_kernel)

        # set materials on the interfaces
        (epsrz, sigez) = self.designer.modify_object(epsrz, sigez)

        # non-magnetic & magnetically lossless material
        murx = mury = self.mur
        sigmx = sigmy = self.sigm

        # updating coefficients
        ceze, cezh = fcmpt(self.dte, epsrz, sigez)
        chxh, chxe = fcmpt(self.dtm, murx, sigmx)
        chyh, chye = fcmpt(self.dtm, mury, sigmy)

        # hidden states
        ez = create_zero_tensor((ns, 1, nx + 1, ny + 1))
        hx = create_zero_tensor((ns, 1, nx + 1, ny))
        hy = create_zero_tensor((ns, 1, nx, ny + 1))

        # CFS-PML auxiliary fields
        pezx = zeros_like(ez)
        pezy = zeros_like(ez)
        phxy = zeros_like(hx)
        phyx = zeros_like(hy)

        # set source location
        jz_t = zeros_like(ez)

        # ----------------------------------------
        # Update
        # ----------------------------------------
        outputs = []

        t = 0

        while t < nt:

            jz_t = self.designer.update_sources(
                (jz_t,), (ez,), waveform_t[t], dt)

            # RNN-Style Update
            (ez, hx, hy, pezx, pezy, phxy, phyx) = self.fdtd_layer(
                jz_t, ez, hx, hy, pezx, pezy, phxy, phyx,
                ceze, cezh, chxh, chxe, chyh, chye)

            # Compute outputs
            outputs.append(
                self.designer.get_outputs_at_each_step((ez, hx, hy)))

            t = t + 1

        outputs = hstack(outputs)
        return outputs
```

## Model Training

Define the differentiable FDTD network, loss function, optimizer, iteration steps and learning rate, then define the electromagnetic inverse scattering solver object `solver` and call the `solve` interface for solving.

```python
# define fdtd network
fdtd_net = transverse_magnetic.ADFDTD(
    cell_numbers, cell_lengths, nt, dt, ns,
    inverse_domain, cpml, rho_init)

# define sovler for inverse problem
epochs = options.epochs
lr = options.lr
loss_fn = nn.MSELoss(reduction='sum')
optimizer = nn.Adam(fdtd_net.trainable_params(), learning_rate=lr)
solver = EMInverseSolver(fdtd_net, loss_fn, optimizer)

# solve
solver.solve(epochs, waveform_t, field_labels)
```

## Evaluating the Solution Results

After the solution is finished, the `eval` interface is called to evaluate the `PSNR` and `SSIM` of the inversion results.

```python
epsr, _ = solver.eval(epsr_labels)
```

The relative permittivity of `PSNR` and `SSIM` obtained by inversion based on the above method are:

```log
[epsr] PSNR: 27.835317 dB, SSIM: 0.963564
```

The relative permittivity distribution obtained by the inversion is shown in the figure below:

![inversion result](https://gitee.com/mindspore/docs/raw/master/docs/mindelec/docs/source_zh_cn/images/AD_FDTD/fdtd_inverse/epsr_reconstructed.png)
