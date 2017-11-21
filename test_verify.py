#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import numpy.testing as npt
import verify

class contingency2x2_construction(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        self.predlist = [0,1,1,0,1]
        self.obslist = [1,0,1,1,1]
        self.res = [[2,1],[2,0]]

    def tearDown(self):
        super(self.__class__, self).tearDown()

    def test_fromBool_list(self):
        '''Check construction from Boolean series'''
        ctab = verify.Contingency2x2.fromBoolean(self.predlist, self.obslist)
        npt.assert_array_equal(self.res, ctab)

    def test_fromBool_arrays(self):
        '''Check construction from Boolean series'''
        ctab = verify.Contingency2x2.fromBoolean(np.asarray(self.predlist), np.asarray(self.obslist))
        npt.assert_array_equal(self.res, ctab)

    def test_direct_list(self):
        '''check construction from direct provision of rates as list'''
        ctab = verify.Contingency2x2(self.res)
        npt.assert_array_equal(self.res, ctab)

    def test_direct_array(self):
        '''check construction from direct provision of rates as array'''
        ctab = verify.Contingency2x2(np.asarray(self.res))
        npt.assert_array_equal(self.res, ctab)

class contingency2x2_calc(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        self.predlist = [0,1,1,0,1]
        self.obslist = [1,0,1,1,1]
        self.MC = 0.8
        self.Finley = [[28,72],[23,2680]]
        self.Finley_b = [[0,0],[51,2752]]

    def tearDown(self):
        super(self.__class__, self).tearDown()

    def test_majorityClass_pos(self):
        '''Test estimation of majority classifier (uses PC) for True majority'''
        ctab = verify.Contingency2x2.fromBoolean(self.predlist, self.obslist)
        self.assertEqual(ctab.majorityClassFraction(), self.MC)

    def test_majorityClass_neg(self):
        '''Test estimation of majority classifier (uses PC) for False majority'''
        neg_obslist = ~np.asarray(self.obslist).astype(bool)
        ctab = verify.Contingency2x2.fromBoolean(self.predlist, neg_obslist)
        self.assertEqual(ctab.majorityClassFraction(), self.MC)

    def test_Finley_n(self):
        ctab = verify.Contingency2x2(self.Finley)
        self.assertEqual(ctab.sum(), 2803)

    def test_Finley_threat(self):
        ctab = verify.Contingency2x2(self.Finley)
        self.assertAlmostEqual(ctab.threat(), 0.228, places=3)

    def test_Finley_bias(self):
        ctab = verify.Contingency2x2(self.Finley)
        self.assertAlmostEqual(ctab.bias(), 1.96, places=2)

    def test_Finley_HSS(self):
        ctab = verify.Contingency2x2(self.Finley)
        self.assertAlmostEqual(ctab.heidke(), 0.355, places=3)

    def test_Finley_PSS(self):
        ctab = verify.Contingency2x2(self.Finley)
        self.assertAlmostEqual(ctab.peirce(), 0.523, places=3)

    def test_Finley_GSS(self):
        ctab = verify.Contingency2x2(self.Finley)
        self.assertAlmostEqual(ctab.equitableThreat(), 0.216, places=3)

    def test_Finley_const_no_PSS(self):
        ctab = verify.Contingency2x2(self.Finley_b)
        self.assertEqual(ctab.peirce(), 0)

    def test_Finley_MC(self):
        '''Ensure that PC for the constant Finley forecast is the same as the MC for the standard Finley'''
        ctab = verify.Contingency2x2(self.Finley)
        ctab_const_no = verify.Contingency2x2(self.Finley_b)
        self.assertEqual(ctab_const_no.PC(), ctab.majorityClassFraction())
        self.assertAlmostEqual(ctab.majorityClassFraction(), 0.982, places=3)
        
        

if __name__ == '__main__':
    unittest.main()

