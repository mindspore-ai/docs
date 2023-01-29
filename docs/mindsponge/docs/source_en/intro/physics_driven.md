# Physics Driven

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindsponge/docs/source_en/intro/physics_driven.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

Traditional molecular dynamics simulations mainly use physical knowledge to perform computational simulations of molecular systems.

Usually, the system is set up to consist of many particles, and each particle represents one atom (in all-atom simulations) or several atoms (in coarse-grained simulations). There are certain interactions between the particles, and the equations of motion can be solved according to theoretical mechanics, so that the trajectories can be obtained for dynamic studies.

According to the statistical mechanics, as long as the simulation is long enough, each conformation in the trajectories is distributed with equal probability in the corresponding ensemble, which makes molecular dynamics simulations able to perform thermodynamic studies.

The interactions existing between particles in molecular dynamics are also known as the force field. The force field determines both the motion of the molecules and the ensemble of the simulated system.

Building a suitable force field for the real world can simulate the microscopic world in a real physical scenario. It is also possible to add the artificially designed bias potential to the force field, and thus physically driven molecular sampling results can be obtained.