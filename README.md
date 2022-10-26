# Physics Informed Neural Networks

This repository implements PINNs from Raissi et al. 2017. Specifically, the notebook adapts the code implementation of Data-Driven Solutions of Nonlinear Partial Differential Equations from https://github.com/maziarraissi/PINNs, and the code by Michael Ito, to solve the force-field equation for solar modulation of cosmic rays. Michael Ito's code adaption use the TF2 API where the main mechanisms of the PINN arise in the train_step function efficiently computing higher order derivatives of custom loss functions through the use of the GradientTape data structure. 

In this application, our PINN is $h(r, p) = \frac{\partial f}{\partial r} + \frac{RV}{3k} \frac{\partial f}{\partial p}$ where $k=\beta(p)k_1(r)k_2(r)$ and $\beta = \frac{p}{\sqrt{p^2 + M^2}}$. We will approximate $f(r, p)$ using a neural network.

We have no initial data, but our boundary data will be given by $f(r_{HP}, p) = \frac{J(r_{HP}, T)}{p^2} = \frac{(T+M)^\gamma}{p^2}$, where $r_{HP} = 120$ AU (i.e. the radius of Heliopause), $M=0.938$ GeV, $\gamma$ is between $-2$ and $-3$, and $T = \sqrt{p^2 + M^2} - M$. Or, vice versa, $p = \sqrt{T^2 + 2TM}$.
