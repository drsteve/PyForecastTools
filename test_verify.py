#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import copy
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

    def test_MatthewsCC_perf(self):
        '''Matthews correlation coefficient should be 1 for perfect forecast'''
        perfpred = copy.copy(self.obslist)
        ctab = verify.Contingency2x2.fromBoolean(perfpred, self.obslist)
        self.assertEqual(ctab.MatthewsCC(), 1)

    def test_MatthewsCC_known1(self):
        '''Matthews correlation coefficient known value test'''
        ctab = verify.Contingency2x2([[0, 0], [24, 327]])
        self.assertEqual(ctab.MatthewsCC(), 0)

    def test_MatthewsCC_known2(self):
        '''Matthews correlation coefficient known value test'''
        ctab = verify.Contingency2x2([[0, 24], [327, 0]])
        self.assertEqual(ctab.MatthewsCC(), -1)

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


class contingencyNxN_tests(unittest.TestCase):
    def test_NxNto2x2(self):
        '''Goldsmith's 3 category non-probabilistic forecasts [see Wilks, 1995, p273]'''
        tt = verify.ContingencyNxN([[50,91,71],[47,2364,170],[54,205,3288]]) 
        tt2 = tt.get2x2(0)
        cat0 = [[50, 162], [101, 6027]]
        npt.assert_array_equal(tt2, cat0)


class perfectContinuous(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        self.predvec = [8, 9, 11, 10, 10, 11, 15, 19, 17, 16, 12, 8, 6, 7, 9, 15, 16, 17, 24, 28, 32, 28, 20]
        self.obsvec = copy.copy(self.predvec)

    def tearDown(self):
        super(self.__class__, self).tearDown()

    def test_meanSquaredError(self):
        '''Test estimation of MSE for perfect forecast'''
        self.assertEqual(verify.meanSquaredError(self.predvec, self.obsvec), 0)

    def test_RMSE(self):
        '''Test estimation of RMSE for perfect forecast'''
        self.assertEqual(verify.RMSE(self.predvec, self.obsvec), 0)

    def test_skill(self):
        '''Test estimation of skill score for perfect forecast'''
        Aref = verify.meanSquaredError(self.obsvec, np.mean(self.obsvec))
        Apred = verify.meanSquaredError(self.predvec, self.obsvec)
        self.assertEqual(verify.skill(Apred, Aref, A_perf=0), 100)

    def test_bias(self):
        '''Bias of perfect prediction is zero'''
        self.assertEqual(verify.bias(self.predvec, self.obsvec), 0)

    def test_percBetter(self):
        '''Percent Better of perfect prediction, relative to all-incorrect is 100'''
        badmodel = verify._maskSeries(self.predvec) - 1 
        self.assertEqual(verify.percBetter(self.predvec, badmodel, self.obsvec), 100)


class relativeContinuous(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        self.predvec = [170]*10
        self.obsvec = [100]*10

    def tearDown(self):
        super(self.__class__, self).tearDown()

    def test_meanPercentageError(self):
        '''Test estimation of mean percentage error'''
        self.assertEqual(verify.meanPercentageError(self.predvec, self.obsvec), 70)

    def test_meanAPE1(self):
        '''Test estimation of mean absolute percentage error'''
        self.assertEqual(verify.meanAPE(self.predvec, self.obsvec), 70)

    def test_medSymAccuracy1(self):
        '''Test estimation of median symmetric accuracy'''
        self.assertEqual(verify.meanPercentageError(self.predvec, self.obsvec), 70)

    def test_medSymAccuracy2(self):
        '''Test estimation of median symmetric accuracy (reverse ordering)'''
        self.assertEqual(verify.medSymAccuracy(self.obsvec, self.predvec), 70)

    def test_medSymAccuracy3(self):
        '''Test estimation of median symmetric accuracy (UPE method)'''
        self.assertEqual(verify.medSymAccuracy(self.obsvec, self.predvec, method='UPE'), 70)


class otherContinuous(unittest.TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        self.predalt = [-2, 2, -2, 2, -2, 2, -2, 2]
        self.obsalt = [0]*len(self.predalt)

    def tearDown(self):
        super(self.__class__, self).tearDown()

    def test_bias_alt0(self):
        '''Test that bias is zero for prediction alternating evenly about observation '''
        self.assertEqual(verify.bias(self.predalt, self.obsalt), 0)

    def test_bias_offset(self):
        '''Test that bias is correct for constant prediction offset from constant observation '''
        val, diff = 5, 2
        n_elements = 12
        predvec = [val]*n_elements
        obsvec = [val-diff]*n_elements
        self.assertEqual(verify.bias(predvec, obsvec), diff)
        
    def test_meanAbsError_alt0(self):
        '''Test that bias is zero for prediction alternating evenly about observation '''
        self.assertEqual(verify.meanAbsError(self.predalt, self.obsalt), 2)
        
    def test_RMSE_alt0(self):
        '''Test that RMSE is correct for prediction alternating evenly about observation '''
        self.assertEqual(verify.RMSE(self.predalt, self.obsalt), 2)

    def test_meanSquaredError_alt0(self):
        '''Test that MSE is correct for prediction alternating evenly about observation '''
        self.assertEqual(verify.meanSquaredError(self.predalt, self.obsalt), 4)

    def test_RMSE_offset(self):
        '''Test that RMSE is correct for constant prediction offset from constant observation '''
        val, diff = 5, 2
        n_elements = 12
        predvec = [val]*n_elements
        obsvec = [val-diff]*n_elements
        self.assertEqual(verify.RMSE(predvec, obsvec), diff)

    def test_RMSE_offset_neg(self):
        '''Test that RMSE is correct for constant prediction offset from constant observation (with negatives)'''
        val, diff = -5, -2
        n_elements = 12
        predvec = [val]*n_elements
        obsvec = [val-diff]*n_elements
        self.assertEqual(verify.RMSE(predvec, obsvec), np.abs(diff))

    def test_medAbsError_equal_meanAbsError_const(self):
        '''median absolute error is same as mean absolute error for constant pred/obs vectors'''
        val, diff = 5, 2
        n_elements = 12
        predvec = [val]*n_elements
        obsvec = [val-diff]*n_elements
        npt.assert_array_equal(verify.medAbsError(predvec, obsvec), verify.meanAbsError(predvec, obsvec))


class individualContinuous(unittest.TestCase):
    def test_forecastError_spanzero_list(self):
        pred, obs = [-1], [1]
        self.assertEqual(verify.forecastError(pred, obs, full=False), -2)

    def test_forecastError_spanzero_scalar(self):
        pred, obs = -1, 1
        self.assertEqual(verify.forecastError(pred, obs, full=False), -2)

    def test_forecastError_full(self):
        pred, obs = [4.1], np.array([3.7])
        err, newpred, newobs = verify.forecastError(pred, obs)
        npt.assert_array_equal(newpred, pred)
        npt.assert_array_equal(newobs, obs)


if __name__ == '__main__':
    unittest.main()

