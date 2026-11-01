.. _`How to run sample codes`:

How to run sample codes
-------------------------

In this chapter, we will explain the setup process for running each sample code
in Python Robotics Projects and describe the contents of each directory.

Steps to setup and run sample codes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install `Python 3.12.x`_

Note that older versions of Python3 might work, but we recommend using
the specified version, because the sample codes are only tested with it.

2. If you prefer to use conda for package management, install
Anaconda or Miniconda to use `conda`_ command.

3. Clone or download this repo and go into dir.

.. code-block::

    >$ cd Python-Robotics-Projects


4. Install the required libraries.

We have prepared requirements management files for `conda`_ and `pip`_ under
the requirements directory. Using these files makes it easy to install the necessary libraries.

using conda :

.. code-block::

    >$ conda env create -f requirements/environment.yml

using pip :

.. code-block::

    >$ pip install -r requirements/requirements.txt


5. Execute python script in each directory.

For example, to run the sample code of `Extended Kalman Filter` in the
`localization` directory, execute the following command:

.. code-block::

    >$ cd localization/extended_kalman_filter

	>$ python extended_kalman_filter.py

Then, you can see an animation of the EKF algorithm based localization.

Please refer to the `Directory structure`_ section for more details on the contents of each directory.

6. Add star to this repo if you like it 😃.

.. _`Python 3.12.x`: https://www.python.org/
.. _`conda`: https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html
.. _`pip`: https://pip.pypa.io/en/stable/

.. _`Directory structure`:

Directory structure
~~~~~~~~~~~~~~~~~~~~

The top-level directory structure of Python Robotics Projects is as follows:

Sample codes directories:

- ``AerialNavigation`` : the sample codes of aerial navigation algorithms for drones and rocket landing.
- ``ArmNavigation`` : the sample codes of arm navigation algorithms for robotic arms.
- ``Localization`` : the sample codes of localization algorithms.
- ``Bipedal`` : the sample codes of bipedal walking algorithms for legged robots.
- Control : the sample codes of control algorithms for robotic systems.
- ``Mapping`` : the sample codes of mapping or obstacle shape recognition algorithms.
- ``PathPlanning`` : the sample codes of path planning algorithms.
- ``PathTracking`` : the sample codes of path tracking algorithms for car like robots.
- ``SLAM`` : the sample codes of SLAM algorithms.

Other directories:

- ``docs`` : This directory contains the documentation of Python Robotics Projects.
- ``requirements`` : This directory contains the requirements management files.
- ``tests`` : This directory contains the unit test files.
- ``utils`` : This directory contains utility functions used in some sample codes in common.
- ``.github`` : This directory contains the GitHub Actions configuration files.
- ``.circleci`` : This directory contains the CircleCI configuration files.

The structure of this document is the same as that of the sample code
directories mentioned above.
For more details, please refer to the :ref:`How to read this textbook` section.


