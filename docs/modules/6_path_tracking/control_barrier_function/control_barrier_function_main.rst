Control barrier function (CBF) safety filter
--------------------------------------------

.. image:: https://raw.githubusercontent.com/robofan45/cbf-safety-filter/main/assets/demo.gif
   :alt: A safety filter steers a robot around three obstacles on its way to the goal

A robot's own controller heads straight for its goal and knows nothing about the obstacles. A safety filter sits between that controller and the robot and changes each command as little as possible, so that the robot never enters an obstacle.

Algorithm
~~~~~~~~~

Each obstacle :math:`i` with centre :math:`c_i` and radius :math:`R_i` (enlarged by the robot's radius) gives a barrier :math:`h_i(p) = \lVert p - c_i \rVert^2 - R_i^2`, positive while the robot is clear of it. Every step the filter solves the quadratic program

.. math:: u^* = \arg\min_u \lVert u - u_\text{nom} \rVert^2 \quad \text{subject to} \quad 2 (p - c_i)^\top u \ge -\alpha \, h_i(p) \ \text{for all } i

so a barrier may shrink, but more and more slowly as it nears zero, and never crosses it. A command that is already safe passes unchanged. With a two-dimensional command the solution is the nominal command, its projection onto one constraint line, or a corner of two, so the program is solved exactly by checking these candidates. Because each barrier is convex, the condition also holds along the whole straight segment the robot travels in a step.

Code Link
~~~~~~~~~

.. autofunction:: PathTracking.control_barrier_function.control_barrier_function.safety_filter

Standalone project and report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm is also published as a standalone project, `robofan45/cbf-safety-filter <https://github.com/robofan45/cbf-safety-filter>`_, with an installable package, tests that run on Linux, macOS and Windows, and an IEEE-style technical report (`PDF <https://github.com/robofan45/cbf-safety-filter/blob/main/docs/report.pdf>`_) with measurements and a full bibliography.

Reference
~~~~~~~~~

-  A. D. Ames, X. Xu, J. W. Grizzle and P. Tabuada, "Control Barrier Function Based Quadratic Programs for Safety Critical Systems", IEEE Transactions on Automatic Control, 2017
-  A. D. Ames et al., "Control Barrier Functions: Theory and Applications", European Control Conference, 2019
