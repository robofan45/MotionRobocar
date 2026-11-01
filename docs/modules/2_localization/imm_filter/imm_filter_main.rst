Interacting multiple model (IMM) filter
---------------------------------------

.. image:: https://raw.githubusercontent.com/robofan45/imm-filter/main/assets/demo.gif
   :alt: The IMM estimate follows a vehicle through two turns

A vehicle drives straight, turns left, drives straight and turns right, and only its position is measured. No single motion model fits all of that, so the IMM filter runs one Kalman filter per model, straight motion and a turn in each direction, and keeps a probability for each.

Algorithm
~~~~~~~~~

Every measurement goes through four steps:

1. Mixing: each filter starts from a blend of all filters' estimates, weighted by the probability of switching between models.
2. Model-matched filtering: each filter predicts with its own model and updates with the measurement.
3. Model probabilities: the models that predicted the measurement well gain probability, in proportion to the likelihood :math:`\Lambda_j` of the measurement under each model.
4. Combination: the output is the probability-weighted blend of the filters:

.. math:: \hat x = \sum_j \mu_j \hat x_j, \qquad P = \sum_j \mu_j \left[ P_j + (\hat x_j - \hat x)(\hat x_j - \hat x)^\top \right]

Compared with a single constant-velocity Kalman filter whose process noise is tuned for its lowest error, the IMM filter's error is the same in turns as on straight segments.

Code Link
~~~~~~~~~

.. autofunction:: Localization.imm_filter.imm_filter.imm_step

Standalone project and report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm is also published as a standalone project, `robofan45/imm-filter <https://github.com/robofan45/imm-filter>`_, with an installable package, tests that run on Linux, macOS and Windows, and an IEEE-style technical report (`PDF <https://github.com/robofan45/imm-filter/blob/main/docs/report.pdf>`_) with measurements and a full bibliography.

Reference
~~~~~~~~~

-  H\. A. P. Blom and Y. Bar-Shalom, "The Interacting Multiple Model Algorithm for Systems with Markovian Switching Coefficients", IEEE Transactions on Automatic Control, 1988
-  Y\. Bar-Shalom, X. R. Li and T. Kirubarajan, "Estimation with Applications to Tracking and Navigation", Wiley, 2001
