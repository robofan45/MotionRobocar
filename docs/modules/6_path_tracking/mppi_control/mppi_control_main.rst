Model predictive path integral (MPPI) control
---------------------------------------------

.. image:: https://raw.githubusercontent.com/robofan45/mppi-control/main/assets/demo.gif
   :alt: MPPI drives a car along a course and around two obstacles

MPPI chooses each command by simulating hundreds of randomly perturbed control sequences and averaging them with weights that decay exponentially with their cost. It needs neither gradients nor a convex model, so its cost can be as blunt as a fixed penalty for every simulated collision. Here it drives a kinematic bicycle model along a course and around two obstacles placed on it.

Algorithm
~~~~~~~~~

Every control step of 0.1 s:

1. Draw 300 noise sequences and add them to the current plan of accelerations and steering rates.
2. Simulate the car 2.5 s ahead with each perturbed plan, all samples at once.
3. Score each trajectory: distance to the course, error of the speed along the course, intrusion into a margin around the obstacles, and a fixed cost for every step inside an obstacle.
4. Move the plan towards the cheap samples, each weighted by the exponential of its negative cost:

.. math:: w_k = \frac{\exp\left(-(S_k-\rho)/\lambda\right)}{\sum_j \exp\left(-(S_j-\rho)/\lambda\right)}, \qquad \bar u_t \leftarrow \bar u_t + \sum_k w_k \varepsilon_t^k, \qquad \rho = \min_k S_k

5. Apply the first command, shift the plan by one step and repeat.

The steering angle is part of the car's state and MPPI samples its rate of change, so the wheel turns smoothly. The speed term counts only the speed along the course: with the raw speed, circling beside an obstacle at full speed would cost less than driving around it.

Code Link
~~~~~~~~~

.. autofunction:: PathTracking.mppi_control.mppi_control.mppi_control

Standalone project and report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm is also published as a standalone project, `robofan45/mppi-control <https://github.com/robofan45/mppi-control>`_, with an installable package, tests that run on Linux, macOS and Windows, and an IEEE-style technical report (`PDF <https://github.com/robofan45/mppi-control/blob/main/docs/report.pdf>`_) with measurements and a full bibliography.

Reference
~~~~~~~~~

-  G. Williams, A. Aldrich and E. A. Theodorou, "Model Predictive Path Integral Control: From Theory to Parallel Computation", Journal of Guidance, Control, and Dynamics, 2017
-  G. Williams et al., "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving", IEEE Transactions on Robotics, 2018
