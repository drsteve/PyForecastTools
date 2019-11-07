.. PyForecastTools documentation master file, created by
   sphinx-quickstart on Mon Jun 11 16:05:02 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyForecastTools documentation
=============================

A Python module to provide model validation and forecast verification tools,
including a set of convenient plot functions. A selection of capabilites
provided by PyForecastTools includes:
 * Accuracy and bias metrics for continuous predictands
     - Unscaled/absolute measures
     - Relative measures
     - Scaled measures
 * 2x2 and NxN contingency table classes
     - Wide range of contingency table metrics and scores
     - Multiple methods of calculating confidence intervals on scores
 * Convenient plotting for visually comparing models and data
     - Quantile-Quantile plots
     - Taylor diagrams
     - ROC curves
     - Reliability diagrams

The module builds on the scientific Python stack (Python, Numpy, MatPlotLib)
and uses the dmarray class from SpacePy's datamodel.

SpacePy is available through the Python Package Index, MacPorts, and is under
version control at [github.com/spacepy/spacepy](https://github.com/spacepy/spacepy)
If SpacePy is not available a reduced functionality implementation of the class
is provided with this package.

PyForecastTools is available through the Python Package Index and can be installed
simply with

> pip install PyForecastTools --user

To install (local user), run

> python setup.py install --user

After installation, the module can then be imported (within a Python script or 
interpreter) by

> import verify

For help, please see the docstrings for each function and/or class.


..module::verify

Contents:
=========

.. toctree::
   :maxdepth: 1

   metrics
   categorical
   plot
   datamodel


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

