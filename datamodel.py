#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified numpy array based on SpacePy's datamodel
"""

from __future__ import division
import copy
import datetime
import itertools
from functools import partial
import os
import re
import numpy

try:
    import StringIO # can't use cStringIO as we might have unicode
except ImportError:
    import io as StringIO


class dmarray(numpy.ndarray):
    """
    """
    Allowed_Attributes = ['attrs']

    def __new__(cls, input_array, attrs=None, dtype=None):
       # Input array is an already formed ndarray instance
       # We first cast to be our class type
       if not dtype:
           obj = numpy.asarray(input_array).view(cls)
       else:
           obj = numpy.asarray(input_array).view(cls).astype(dtype)
       # add the new attribute to the created instance
       if attrs != None:
           obj.attrs = attrs
       else:
           obj.attrs = {}
       # Finally, return the newly created object:
       return obj

    def __array_finalize__(self, obj):
       # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        for val in self.Allowed_Attributes:
            self.__setattr__(val, copy.deepcopy(getattr(obj, val, {})))

    def __array_wrap__(self, out_arr, context=None):
        #check for zero-dims (numpy bug means subclass behaviour isn't consistent with ndarray
        #this traps most of the bad behaviour ( std() and var() still problems)
        if out_arr.ndim > 0:
            return numpy.ndarray.__array_wrap__(self, out_arr, context)
        else:
            return numpy.ndarray.__array_wrap__(self, out_arr, context).tolist()

    def __reduce__(self):
        """This is called when pickling, see:
        http://www.mail-archive.com/numpy-discussion@scipy.org/msg02446.html
        for this particular example.
        Only the attributes in Allowed_Attributes can exist
        """
        object_state = list(numpy.ndarray.__reduce__(self))
        subclass_state = tuple([tuple([val, self.__getattribute__(val)]) for val in self.Allowed_Attributes])
        object_state[2] = (object_state[2],subclass_state)
        return tuple(object_state)

    def __setstate__(self, state):
        """Used for unpickling after __reduce__ the self.attrs is recovered from
        the way it was saved and reset.
        """
        nd_state, own_state = state
        numpy.ndarray.__setstate__(self,nd_state)
        for i, val in enumerate(own_state):
            if not val[0] in self.Allowed_Attributes: # this is attrs
                self.Allowed_Attributes.append(own_state[i][0])
            self.__setattr__(own_state[i][0], own_state[i][1])

    def __setattr__(self, name, value):
        """Make sure that .attrs is the only attribute that we are allowing
        dmarray_ne took 15.324803 s
        dmarray_eq took 15.665865 s
        dmarray_assert took 16.025478 s
        It looks like != is the fastest, but not by much over 10000000 __setattr__
        """
        if name == 'Allowed_Attributes':
            pass
        elif not name in self.Allowed_Attributes:
            raise(TypeError("Only attribute listed in Allowed_Attributes can be set"))
        super(dmarray, self).__setattr__(name, value)

    def addAttribute(self, name, value=None):
        """Method to add an attribute to a dmarray
        equivalent to
        a = datamodel.dmarray([1,2,3])
        a.Allowed_Attributes = a.Allowed_Attributes + ['blabla']
        """
        if name in self.Allowed_Attributes:
            raise(NameError('{0} is already an attribute cannot add again'.format(name)))
        self.Allowed_Attributes.append(name)
        self.__setattr__(name, value)

    def _saveAttrs(self):
        Allowed_Attributes = self.Allowed_Attributes
        backup = []
        for atr in Allowed_Attributes:
            backup.append( (atr, dmcopy(self.__getattribute__(atr)) ) )
        return backup

    @classmethod
    def _replaceAttrs(cls, arr, backup):
        for key, val in backup:
            if key != 'attrs':
                try:
                    arr.addAttribute(key)
                except NameError:
                    pass
            arr.__setattr__(key, val)
        return arr







































