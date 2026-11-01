Monte Carlo localization with a lidar
-------------------------------------

.. image:: https://raw.githubusercontent.com/robofan45/monte-carlo-localization/main/assets/demo.gif
   :alt: Particles spread over a floor plan converge on the robot

A robot drives through a known floor plan without knowing where it started. Monte Carlo localization tracks its pose with a cloud of particles, each one a guess of the pose, which begins spread over every free part of the map.

Algorithm
~~~~~~~~~

Every step:

1. Move every particle with the noisy odometry.
2. Weight every particle by how well the lidar scan, placed at its pose, lines up with the walls. The likelihood field scores each beam by the distance :math:`d_k` from its endpoint to the nearest wall, read from a distance map computed once:

.. math:: p(z_k \mid x) = z_\text{hit} \exp\left(-\frac{d_k^2}{2\sigma_\text{hit}^2}\right) + \frac{z_\text{rand}}{z_\text{max}}

3. Resample: particles that explain the scan are copied, the others disappear.
4. Augmented MCL: keep a fast and a slow average of the scan likelihood, and replace particles by random poses with probability :math:`\max(0, 1 - w_\text{fast}/w_\text{slow})`, so that a cloud that settled on the wrong place can still find the right one.

Code Link
~~~~~~~~~

.. autofunction:: Localization.monte_carlo_localization.monte_carlo_localization.log_likelihoods

Standalone project and report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm is also published as a standalone project, `robofan45/monte-carlo-localization <https://github.com/robofan45/monte-carlo-localization>`_, with an installable package, tests that run on Linux, macOS and Windows, and an IEEE-style technical report (`PDF <https://github.com/robofan45/monte-carlo-localization/blob/main/docs/report.pdf>`_) with measurements and a full bibliography.

Reference
~~~~~~~~~

-  S\. Thrun, W. Burgard and D. Fox, "Probabilistic Robotics", MIT Press, 2005
-  F\. Dellaert, D. Fox, W. Burgard and S. Thrun, "Monte Carlo Localization for Mobile Robots", ICRA, 1999
