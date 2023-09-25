## D-ODE-Solvers
<sub>Official implementation of the paper "Distilling ODE Solvers of Diffusion Models into Smaller Steps"</sub>

The code consists of two parts depending on diffusion models.

The noise prediction models with DDIM, iPNDM, DPM-Solver, and DEIS is based on the code base of [DPM-solver](https://github.com/LuChengTHU/dpm-solver/tree/main/examples/ddpm_and_guided-diffusion) and [DEIS](https://github.com/qsh-zh/deis/tree/main). 

The data prediction models with DDIM and EDM is heavily based on [EDM codebase](https://github.com/NVlabs/edm).

Please follow their instructions to set up each environment and download pretrained models. 

In noise prediction models, you need jax library to run DEIS. Please run follow command after you set up your environment referring to DPM-Solver repository.
```.bash
# for pytorch user
pip install "jax[cpu]"
```
