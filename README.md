# PyForecastTools

[![Build Status](https://travis-ci.org/drsteve/PyForecastTools.svg?branch=master)](jttps://travis-ci.org/drsteve/PyForecastTools)

A Python module to provide model validation and forecast verification tools.
The module builds on the scientific Python stack (Python, Numpy) and uses
the dmarray class from SpacePy's datamodel.

SpacePy is available through the Python Package Index, MacPorts, and is under
version control at sourceforge.net/p/spacepy/
If SpacePy is not available a reduced functionality implementation of the class
is provided with this package.

To install (local user), run

> python setup.py install --user

The module can then be imported (within a Python script or interpreter) by

> import verify

For help, please see the docstrings for each function and/or class.
