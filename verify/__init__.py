"""Module containing verification and performance metrics

With the exception of the ContingencyNxN and Contingency2x2 classes,
the inputs for all metrics are assumed to be array-like and 1D. Bad
values are assumed to be stored as NaN and these are excluded in
metric calculations.

With the exception of the ContingencyNxN and Contingency2x2 classes,
the inputs for all metrics are assumed to be array-like and 1D. Bad
values are assumed to be stored as NaN and these are excluded in
metric calculations.

Author: Steve Morley
Institution: Los Alamos National Laboratory
Contact: smorley@lanl.gov
Los Alamos National Laboratory

Copyright (c) 2017, Los Alamos National Security, LLC
All rights reserved.
"""
from .metrics import *
from .categorical import *
from . import plot
