ORCA multi-robot collision avoidance
------------------------------------

.. image:: https://raw.githubusercontent.com/robofan45/orca-collision-avoidance/main/assets/demo.gif
   :alt: Ten robots swap sides of a circle and flow around the centre like a roundabout

Ten robots on a circle must each reach the point opposite, so they all meet in the middle. Every step, each robot chooses its own velocity from the positions and velocities of the others, without communication.

Algorithm
~~~~~~~~~

For every neighbour :math:`B`, robot :math:`A` finds the smallest change :math:`u` of their relative velocity that avoids a collision within 3 s, takes half of it, and keeps the half-plane of velocities

.. math:: \mathrm{ORCA}^\tau_{A|B} = \left\{ v \;\middle|\; \left(v - (v_A + \tfrac12 u)\right) \cdot n \ge 0 \right\}

A small linear program then picks, inside all these half-planes and the speed limit, the velocity closest to the one heading straight for the goal. If both robots of a pair choose from their half-planes, they cannot collide within the time horizon. Robots in perfect symmetry would stall in a ring, so every robot keeps slightly to the right, and the crowd flows through the middle like a roundabout.

Code Link
~~~~~~~~~

.. autofunction:: PathPlanning.ORCA.orca.new_velocity

Standalone project and report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm is also published as a standalone project, `robofan45/orca-collision-avoidance <https://github.com/robofan45/orca-collision-avoidance>`_, with an installable package, tests that run on Linux, macOS and Windows, and an IEEE-style technical report (`PDF <https://github.com/robofan45/orca-collision-avoidance/blob/main/docs/report.pdf>`_) with measurements and a full bibliography.

Reference
~~~~~~~~~

-  J\. van den Berg, S. J. Guy, M. Lin and D. Manocha, "Reciprocal n-Body Collision Avoidance", Robotics Research: The 14th International Symposium ISRR, Springer, 2011
-  `RVO2 Library <https://gamma.cs.unc.edu/RVO2/>`_, the authors' reference implementation
