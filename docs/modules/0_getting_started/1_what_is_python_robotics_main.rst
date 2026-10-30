.. _`What is MotionRobocar?`:

What is MotionRobocar?
------------------------

This is an Open Source Software (OSS) project: MotionRobocar, which is a Python code collection of robotics algorithms.
The code is released under the MIT license.

This project has three main philosophies below:

Philosophy 1. Easy to understand each algorithm's basic idea.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goal is for beginners in robotics to understand the basic ideas behind each algorithm.

`Python`_ programming language is adopted in this project to achieve the goal.
Python is a high-level programming language that is easy to read and write.
If the code is not easy to read, it would be difficult to achieve our goal of
allowing beginners to understand the algorithms.
Python has great libraries for matrix operation, mathematical and scientific operation,
and visualization, which makes code more readable because such operations
don’t need to be re-implemented.
Having the core Python packages allows the user to focus on the algorithms,
rather than the implementations.

MotionRobocar provides not only the code but also intuitive animations that
visually demonstrate the process and behavior of each algorithm over time.

The animations are drawn with the `Matplotlib`_ library when you run each sample code.

MotionRobocar also provides a textbook that explains the basic ideas of each algorithm.
The MotionRobocar textbook allows you to learn fundamental algorithms used in
robotics with minimal mathematical formulas and textual explanations,
based on MotionRobocar’ sample codes and animations.
The contents of this document, like the code, are managed in the MotionRobocar
repository and converted into HTML documents using `Sphinx`_,
which is a Python-based documentation builder.
Please refer to this section ":ref:`How to read this textbook`" for information on
how to read this document for learning.


.. _`Python`: https://www.python.org/


Philosophy 2. Widely used and practical algorithms are selected.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second philosophy is that implemented algorithms have to be practical
and widely used in both academia and industry.
We believe learning these algorithms will be useful in many applications.
For example, Kalman filters and particle filter for localization,
grid mapping for mapping,
dynamic programming based approaches and sampling based approaches for path planning,
and optimal control based approach for path tracking.
These algorithms are implemented and explained in the textbook in this project.

Philosophy 3. Minimum dependency.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each sample code of MotionRobocar is written in Python3 and only depends on
some standard modules for readability and ease to setup and use.


.. _`Requirements`:

Requirements
============

MotionRobocar depends only on the following five libraries,
including Python itself, to run each sample code.

-  `Python 3.12.x`_ (for Python runtime)
-  `NumPy`_ (for matrix operation)
-  `SciPy`_ (for scientific operation)
-  `cvxpy`_ (for convex optimization)
-  `Matplotlib`_ (for visualization)

If you only need to run the code, the five libraries mentioned above are sufficient.
However, for code development or creating documentation for the textbook,
the following additional libraries are required.

-  `pytest`_ (for unit tests)
-  `pytest-xdist`_ (for parallel unit tests)
-  `mypy`_ (for type check)
-  `Sphinx`_ (for document generation)
-  `ruff`_ (for code style check)

.. _`Python 3.12.x`: https://www.python.org/
.. _`NumPy`: https://numpy.org/
.. _`SciPy`: https://scipy.org/
.. _`Matplotlib`: https://matplotlib.org/
.. _`cvxpy`: https://www.cvxpy.org/
.. _`pytest`: https://docs.pytest.org/en/latest/
.. _`pytest-xdist`: https://github.com/pytest-dev/pytest-xdist
.. _`mypy`: https://mypy-lang.org/
.. _`Sphinx`: https://www.sphinx-doc.org/en/master/index.html
.. _`ruff`: https://github.com/astral-sh/ruff

For instructions on installing the above libraries, please refer to
this section ":ref:`How to run sample codes`".

