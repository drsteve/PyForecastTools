#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import copy
import warnings
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

    def test_direct_array_floats(self):
        '''check construction from direct provision of rates as array, force floating point'''
        ctab = verify.Contingency2x2(np.asarray(self.res), dtype=np.float)
        npt.assert_array_equal(np.asarray(self.res, dtype=np.float), ctab)

    def test_direct_array3D_raises(self):
        '''check construction raises when given rates from non-2D array'''
        self.assertRaises(ValueError, verify.Contingency2x2, np.ones([2,2,2]))

    def test_direct_array5x2_raises(self):
        '''check construction raises when given rates from wrong shape array'''
        self.assertRaises(ValueError, verify.Contingency2x2, [self.predlist,self.obslist])

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

    def test_oddsRatio_known(self):
        '''Test Odds ratio calculation with example given by Thornes and Stephenson'''
        ctab = verify.Contingency2x2([[29, 6], [4, 38]])
        self.assertAlmostEqual(ctab.oddsRatio(), 45.9, places=1)

    def test_YuleQ_known(self):
        '''Test Yule's Q (ORSS) calculation with example given by Thornes and Stephenson'''
        ctab = verify.Contingency2x2([[29, 6], [4, 38]])
        self.assertAlmostEqual(ctab.yuleQ(), 0.96, places=2)

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

    def test_Finley_PSS_CIWald_regress(self):
        ctab = verify.Contingency2x2(self.Finley)
        npt.assert_array_almost_equal(ctab.peirce(ci='Wald'), [0.5228568171454651, 0.13669651496274715], decimal=5)

    def test_Finley_PSS_CIAgresti_regress(self):
        ctab = verify.Contingency2x2(self.Finley)
        npt.assert_array_almost_equal(ctab.peirce(ci='AC'), [0.5228568171454651, 0.13187940039376933], decimal=5)

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

    def test_BootstrapCI_regress(self):
        '''Test bootstrap CI calculation gives expected answer for given input'''
        ctab = verify.Contingency2x2.fromBoolean(self.predlist*30, self.obslist*30)
        gotval, gotci = ctab.PC(ci='bootstrap')
        npt.assert_array_almost_equal([gotval, gotci[0], gotci[1]], [0.4, 0.32, 0.48])

class contingencyNxN_tests(unittest.TestCase):
    def test_NxNto2x2(self):
        '''Goldsmith's 3 category non-probabilistic forecasts [see Wilks, 1995, p273]'''
        tt = verify.ContingencyNxN([[50,91,71],[47,2364,170],[54,205,3288]]) 
        tt2 = tt.get2x2(0)
        cat0 = [[50, 162], [101, 6027]]
        npt.assert_array_equal(tt2, cat0)


class perfectContinuous(unittest.TestCase):
    '''Tests of continuous metrics using perfects forecasts'''
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
        badmodel = verify.metrics._maskSeries(self.predvec) - 1 
        self.assertEqual(verify.percBetter(self.predvec, badmodel, self.obsvec), 100)

    def test_logAccuracy(self):
        '''Log accuracy of perfect prediction is zero'''
        ans = np.zeros_like(self.obsvec)
        npt.assert_array_equal(verify.logAccuracy(self.predvec, self.obsvec, mask=False), ans)

    def test_APE(self):
        '''Absolute percentage error of perfect prediction is zero'''
        ans = np.zeros_like(self.obsvec)
        npt.assert_array_equal(verify.absPercError(self.predvec, self.obsvec), ans)

    def test_symSignedBias(self):
        '''Symmetric signed percentage bias of perfect prediction is zero'''
        self.assertEqual(verify.symmetricSignedBias(self.predvec, self.obsvec), 0)

class relativeContinuous(unittest.TestCase):
    '''Tests of relative error metrics for known relative error'''
    def setUp(self):
        super(self.__class__, self).setUp()
        self.predvec = [170]*10
        self.obsvec = [100]*10

    def tearDown(self):
        super(self.__class__, self).tearDown()

    def test_APE1(self):
        '''Test estimation of absolute percentage error (positive)'''
        ans = np.ones_like(self.obsvec)
        npt.assert_array_equal(verify.absPercError(self.predvec, self.obsvec), ans*70)

    def test_APE2(self):
        '''Test estimation of absolute percentage error (negative)'''
        ans = np.ones_like(self.obsvec)
        npt.assert_array_equal(verify.absPercError(np.asarray(self.predvec)*-1, np.asarray(self.obsvec)*-1), ans*70)

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
    '''Other tests of continuous error metrics'''
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

    def test_medSymAcc_AllMethodsEqual1(self):
        '''median symmetric accuracy is within precision for different methods'''
        np.random.seed(24601)
        predvec = np.arange(1,40)
        obsvec = predvec+np.random.random(len(predvec))
        npt.assert_array_almost_equal(verify.medSymAccuracy(predvec, obsvec), verify.medSymAccuracy(predvec, obsvec, method='log'))

    def test_medSymAcc_AllMethodsEqual2(self):
        '''median symmetric accuracy is within precision for different methods'''
        np.random.seed(24601)
        predvec = np.arange(1,40)
        obsvec = predvec+np.random.random(len(predvec))
        npt.assert_array_almost_equal(verify.medSymAccuracy(predvec, obsvec), verify.medSymAccuracy(predvec, obsvec, method='UPE'))

class badInputs(unittest.TestCase):
    '''Tests of that should raise errors (or related cases)'''
    def setUp(self):
        super(self.__class__, self).setUp()
        self.predvec = np.arange(1,40)
        self.obsvec = self.predvec+1
        self.Finley = [[28,72],[23,2680]]

    def tearDown(self):
        super(self.__class__, self).tearDown()

    def test_medSymAcc_raises(self):
        '''median symmetric accuracy raises an error when an invalid method is selected'''
        self.assertRaises(NotImplementedError, verify.medSymAccuracy, self.predvec, self.obsvec, method='NotAMethod')

    def test_logAcc_base_raises(self):
        '''log accuracy raises an error when an invalid base is selected'''
        self.assertRaises(NotImplementedError, verify.logAccuracy, self.predvec, self.obsvec, base=7.4)
        self.assertRaises(NotImplementedError, verify.logAccuracy, self.predvec, self.obsvec, base='SevenPoint4')
        self.assertRaises(NotImplementedError, verify.logAccuracy, self.predvec, self.obsvec, base=np.e)

    def test_logAcc_negative_raises(self):
        '''log accuracy raises an error when negative values are given'''
        self.assertRaises(ValueError, verify.logAccuracy, self.predvec*-1, self.obsvec, mask=False)

    def test_logAcc_negative_noRaiseWhenMasked(self):
        '''log accuracy masks all values when predicted array is all negative'''
        with warnings.catch_warnings(): #squelch RuntimeWarnings as this will raise w/ msg "invalid value encountered in log10"
            warnings.simplefilter("ignore")
            self.assertTrue(verify.logAccuracy(self.predvec*-1, self.obsvec, mask=True).mask.all())

    def test_BootstrapCI_failsWithoutBooleanCreate(self):
        '''Test bootstrap CI calculation raises an error when called on a table without input booleans'''
        ctab = verify.Contingency2x2(self.Finley)
        self.assertRaises(AttributeError, ctab.POD, ci='bootstrap')

class individualContinuous(unittest.TestCase):
    '''Tests for continuous metrics using single element prediction/observation vectors'''
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

class precisionMeasures(unittest.TestCase):
    '''Tests for measures of spread'''
    def test_Sn_expected_Gaussian_even(self):
        '''Test that Sn estimator has a mean estimated value consistent with expectation'''
        np.random.seed(24601)
        n_elements = 1500
        n_repeats = 5
        sns = [verify.Sn(np.random.randn(n_elements)) for i in range(n_repeats)]
        npt.assert_almost_equal(np.mean(sns), 1, decimal=2)

    def test_Sn_expected_Gaussian_odd(self):
        '''Test that Sn estimator has a mean estimated value consistent with expectation'''
        np.random.seed(24601)
        n_elements = 1499
        n_repeats = 5
        sns = [verify.Sn(np.random.randn(n_elements)) for i in range(n_repeats)]
        npt.assert_almost_equal(np.mean(sns), 1, decimal=2)

if __name__ == '__main__':
    unittest.main()

