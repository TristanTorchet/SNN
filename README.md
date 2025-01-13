# Welcome to my endeavor to reproduce the most famous SNN papers in JAX 

[comment]: <> (This is a comment, it will not be included)

The following is a list of paper that I reproduced 

Table of contents
=================
1. [Cramer 2020](#cramer20) (work in progress)
2. [Perez-Nieves 2021](#perez2021) (work in progress)

## 1. Cramer 2020 <a name="cramer20"></a>
Features: 
- 3 networks: 
    - vanilla feedforward with 1/2/3 hidden layers of 128 neurons
    - vanilla RSNN with 1 hidden layer of 1024 neurons
- Neuron model
    - LIF
    - DPI synapses
- Regularization: 
    - Global firing upper bound
    - Local firing lower bound
- Hyperparameters
    - $\tau_{mem} = 20ms$
    - $\tau_{syn} = 10ms$
    - $v_{reset} = 0V$
    - $v_{rest} = 0V$
    - $v_{th} = 1V$
    - $dt = 0.1ms$
    - $T = 1s$ - simulation time
    - $lr = 0.001$
    - $\beta_{SG} = 100$
    - $\theta_l = 0.01$ - Regularization local lower threshold
    - $s_l = 1$ - Regularization local lower strength
    - $\theta_u = 100$ -  Regularization global upper threshold
    - $s_u = 0.06$ - Regularization global upper strength
    - Weight initialization: Kaiming uniform: $\mathcal{U} \sim [-\sqrt{6/fan_{in}}, \sqrt{6/fan_{in}}]$
## 2. Perez-Nieves 2021 <a name="pn21"></a>
Features:
- RSNN with 1 hidden layer of 128 neurons
- Heterogeneous synaptic time constants and membrane time constants
    - clip(x, exp(-1/3), 0.995) for $\alpha$ and $\beta$, thus clipping the time constants: $3\Delta t < \tau < \frac{-\Delta t}{\log(0.995)}=200\Delta t$
        - this was motivated to clip $\tau$ to 100ms with $\Delta t = 0.5ms$ 
- Regularization: None
- Hyperparameter: 
  - Same as Cramer 2020
  - Adam optimizer: $\beta_1 = 0.9$, $\beta_2 = 0.999$
  - Time constant initialization: $\tau_{mem} \sim log(\mathcal{N})$


