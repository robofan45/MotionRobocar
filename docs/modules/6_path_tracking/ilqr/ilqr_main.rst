Iterative linear quadratic regulator (iLQR)
-------------------------------------------

.. image:: https://raw.githubusercontent.com/robofan45/ilqr-trajectory-optimization/main/assets/demo.gif
   :alt: iLQR iterations converge on a left turn into a goal pose

iLQR finds the acceleration and steering sequence that takes a car from rest to a goal pose, stopped there, with as little control effort as possible. It starts from a car that does nothing and improves the whole trajectory at every iteration.

Algorithm
~~~~~~~~~

Each iteration:

1. Linearise the car model around the current trajectory, with analytic Jacobians.
2. Solve the resulting linear-quadratic problem backwards in time. Every step gets a feedforward change :math:`k_t` and a feedback gain :math:`K_t`:

.. math:: k_t = -Q_{uu}^{-1} Q_u, \qquad K_t = -Q_{uu}^{-1} Q_{ux}

3. Roll the corrected controls forward through the real, nonlinear model, halving the step size :math:`\alpha` until the cost goes down:

.. math:: \hat u_t = u_t + \alpha k_t + K_t (\hat x_t - x_t), \qquad \hat x_{t+1} = f(\hat x_t, \hat u_t)

4. Adapt the regularisation of the value function's Hessian: smaller after a successful step, larger after a failed one.

The first iteration never steers: at rest, the linearised car cannot turn, so the first improvement only drives straight ahead.

Code Link
~~~~~~~~~

.. autofunction:: PathTracking.ilqr.ilqr.ilqr

Standalone project and report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm is also published as a standalone project, `robofan45/ilqr-trajectory-optimization <https://github.com/robofan45/ilqr-trajectory-optimization>`_, with an installable package, tests that run on Linux, macOS and Windows, and an IEEE-style technical report (`PDF <https://github.com/robofan45/ilqr-trajectory-optimization/blob/main/docs/report.pdf>`_) with measurements and a full bibliography.

Reference
~~~~~~~~~

-  W. Li and E. Todorov, "Iterative Linear Quadratic Regulator Design for Nonlinear Biological Movement Systems", ICINCO, 2004
-  Y. Tassa, T. Erez and E. Todorov, "Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization", IROS, 2012
