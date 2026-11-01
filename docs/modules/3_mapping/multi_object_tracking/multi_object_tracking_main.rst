Multi-object tracking
---------------------

.. image:: https://raw.githubusercontent.com/robofan45/multi-object-tracking/main/assets/demo.gif
   :alt: Five cars keep one track each through missed and false detections

Five cars cross a sensor's field of view. Each scan misses some of them and adds a few false detections at random places, and does not say which detection belongs to which car. The tracker keeps one constant-velocity Kalman filter per track.

Algorithm
~~~~~~~~~

Every scan:

1. Predict every track to the time of the scan.
2. Gate: rule out pairs whose squared Mahalanobis distance :math:`d_{ij}^2` exceeds 9.21, which keeps 99% of true detections.
3. Pair tracks with detections using the Hungarian algorithm, so that as many as possible are paired at the lowest total cost. The cost is the negative log-likelihood of the detection, which adds a penalty for the track's uncertainty:

.. math:: c_{ij} = -2 \ln \mathcal{N}(z_j; H \hat x_i, S_i) = d_{ij}^2 + \ln\det(2\pi S_i)

   With the Mahalanobis distance alone, a track started a moment ago from a false detection looks close to everything and steals detections from established tracks.

4. Update the paired tracks; start a tentative track at every unpaired detection; confirm a track after three detections and delete one that keeps being missed.

Code Link
~~~~~~~~~

.. autofunction:: Mapping.multi_object_tracking.multi_object_tracking.tracker_step

Standalone project and report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm is also published as a standalone project, `robofan45/multi-object-tracking <https://github.com/robofan45/multi-object-tracking>`_, with an installable package, tests that run on Linux, macOS and Windows, and an IEEE-style technical report (`PDF <https://github.com/robofan45/multi-object-tracking/blob/main/docs/report.pdf>`_) with measurements and a full bibliography.

Reference
~~~~~~~~~

-  Y\. Bar-Shalom, P. K. Willett and X. Tian, "Tracking and Data Fusion: A Handbook of Algorithms", YBS Publishing, 2011
-  H\. W. Kuhn, "The Hungarian Method for the Assignment Problem", Naval Research Logistics Quarterly, 1955
