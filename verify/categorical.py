"""Module containing contingency table classes and related metrics

Contingency tables are widely used in analysis of categorical data.
The simplest, and most common, is binary event analysis, where
two categories are given (event/non-event). This can be generalized
to N categories that constitute different types of event.

ContingencyNxN and Contingency2x2 classes are currently provided.
Each class can be constructed with the numbers of true positives,
false positives, false negatives, and true negatives. The
Contingency2x2 class can also be constructed using the fromBoolean
class method by providing arrays of True/False.

Author: Steve Morley
Institution: Los Alamos National Laboratory
Contact: smorley@lanl.gov
Los Alamos National Laboratory

Copyright (c) 2017, Los Alamos National Security, LLC
All rights reserved.
"""

from __future__ import division
import numpy as np

try:
    from spacepy import datamodel as dm
except:
    from . import datamodel as dm

#======= Contingency tables =======#
#import xarray ##TODO: do I want to port this to use xarray instead of dmarray??

class ContingencyNxN(dm.dmarray):
    """Class to work with NxN contingency tables for forecast verification

    Examples
    ========

    >>> import verify
    >>> tt = verify.ContingencyNxN([[28,72],[23,2680]])
    >>> tt.sum()
    2803
    >>> tt.threat()
    0.22764227642276422
    >>> tt.heidke()
    0.35532486145845693
    >>> tt.peirce()
    0.52285681714546284

    """
    def __new__(cls, input_array, attrs=None, dtype=None):
        if not dtype:
            obj = np.asarray(input_array).view(cls)
        else:
            obj = np.asarray(input_array).view(cls).astype(dtype)
        if obj.ndim != 2:
            raise ValueError('NxN contingency tables must be 2-dimensional')
        if obj.shape[0] != obj.shape[1]:
            raise ValueError('NxN contingency tables must be square')
        if attrs != None:
            obj.attrs = attrs
        else:
            obj.attrs = {}
        return obj

    def summary(self, verbose=False, subtables=True):
        dum = self.PC()
        dum = self.heidke()
        dum = self.peirce()

        if verbose:
            stats = ['PC']
            skill = ['HeidkeScore', 'PeirceScore']
            print("Summary Statistics")
            print("==================")
            for key in stats:
                print("{0}: {1}".format(key, self.attrs[key]))
            print("\nSkill Scores")
            print("============")
            for key in skill:
                print("{0}: {1}".format(key, self.attrs[key]))

            #print summary stats for subtables
            if subtables:
                for cat in range(self.shape[0]):
                    print('\nSubtable, category {0:1d}'.format(cat))
                    print('-------------------')
                    self.get2x2(cat).summary(verbose=verbose)

    def heidke(self):
        """Calculate the generalized Heidke Skill Score for the NxN contingency
        table

        Returns
        =======
        hss : float
            The Heidke skill score of the contingency table data
            This is also added to the attrs attribute of the table object

        Examples
        ========
        Goldsmith's non-probabilistic forecasts for freezing rain (cat 0), snow
        (cat 1), and rain (cat 2). [see Wilks, 1995, p273-274]

        >>> import verify
        >>> tt = verify.ContingencyNxN([[50,91,71],[47,2364,170],[54,205,3288])
        >>> tt.heidke()
        0.80535269033647217

        """
        N = self.sum()
        Pyo = 0
        # term 1 in numerator
        Pyo = self.PC()
        # term 2 in numerator including term 1 in denominator (only in square
        # table)
        PyPo = 0
        for i in range(self.shape[0]):
            Py, Po = 0, 0
            for j in range(self.shape[0]):
                Py += self[i, j]
                Po += self[j, i]
            Py /= N
            Po /= N
            PyPo += Py*Po
        # put it together
        hss = (Pyo - PyPo)/(1.0 - PyPo)
        self.attrs['HeidkeScore'] = hss
        return hss

    def peirce(self):
        """Calculate the generalized Peirce Skill Score for the NxN contingency
        table

        Returns
        =======
        pss : float
            The Peirce skill score of the contingency table data
            This is also added to the attrs attribute of the table object

        Examples
        ========
        Goldsmith's non-probabilistic forecasts for freezing rain (cat 0), snow
        (cat 1), and rain (cat 2). [see Wilks, 1995, p273-274]

        >>> import verify
        >>> tt = verify.ContingencyNxN([[50,91,71],[47,2364,170],[54,205,3288])
        >>> tt.peirce()
        0.81071330546125309

        """
        N = self.sum()
        Pyo = 0
        # term 1 in numerator
        Pyo = self.PC()
        # term 2 in numerator including term 1 in denominator (only in square
        # table)
        Po2, PyPo = 0, 0
        for i in range(self.shape[0]):
            Py, Po = 0, 0
            for j in range(self.shape[0]):
                Py += self[i, j]
                Po += self[j, i]
            Py /= N
            Po /= N
            Po2 += Po*Po
            PyPo += Py*Po
        # put it together
        pss = (Pyo - PyPo)/(1.0 - Po2)
        self.attrs['PeirceScore'] = pss
        return pss

    def PC(self):
        """Returns the Proportion Correct (PC) for the NxN contingency table
        """
        self.attrs['PC'] = self.trace()/self.sum()
        return self.attrs['PC']

    def get2x2(self, category):
        """Get 2x2 sub-table from multicategory contingency table

        Examples
        ========
        Goldsmith's non-probabilistic forecasts for freezing rain (cat 0), snow
        (cat 1), and rain (cat 2). [see Wilks, 1995, p273]

        >>> import verify
        >>> tt = verify.ContingencyNxN([[50,91,71],[47,2364,170],[54,205,3288])
        >>> tt2 = tt.get2x2(0)
        >>> print(tt2)
        [[  50  162]
         [ 101 6027]]
        >>> tt2.bias()
        1.4039735099337749
        >>> tt2.summary()
        >>> tt2.attrs()
        {'Bias': 1.4039735099337749,
         'FAR': 0.76415094339622647,
         'HeidkeScore': 0.25474971797571822,
         'POD': 0.33112582781456956,
         'POFD': 0.026175472612699952,
         'PeirceScore': 0.30495035520187008,
         'ThreatScore': 0.15974440894568689}

        """
        a = self[category, category]
        b = self[category, :].sum() - a
        c = self[:, category].sum() - a
        d = self.sum()-self[category, :].sum() - self[:, category].sum() + a
        return Contingency2x2([[a, b], [c, d]])


class Contingency2x2(ContingencyNxN):
    """Class to work with 2x2 contingency tables for forecast verification

    The table is defined following the standard presentation in works such
    as Wilks [2006], where the columns are observations and the rows are
    predictions. For a binary forecast, this gives a table

    +-------------+-------------------------------+
    |             |           Observed            |
    |             +---------------+---------------+
    |             |      Y        |      N        |
    +---------+---+---------------+---------------+
    |         | Y | True Positive | False Positive|
    |Predicted+---+---------------+---------------+
    |         | N | False Negative| True Negative |
    +---------+---+---------------+---------------+


    Note that in many machine learning applications this table is called a
    ``confusion matrix'' and the columns and rows are often transposed.

    Wilks, D.S. (2006), Statistical Methods in the Atmospheric Sciences, 2nd Ed.
    Academic Press, Elsevier, Burlington, MA.

    Examples
    ========
    Duplicating the Finley[1884] tornado forecasts [Wilks, 2006, pp267-268]

    >>> import verify
    >>> tt = verify.Contingency2x2([[28,72],[23,2680]])
    >>> tt.sum()
    2803
    >>> tt.threat()
    0.22764227642276422
    >>> tt.heidke()
    0.35532486145845693
    >>> tt.peirce()
    0.52285681714546284

    """
    def __new__(cls, input_array, attrs=None, dtype=None):
        if not dtype:
            obj = np.asarray(input_array).view(cls)
        else:
            obj = np.asarray(input_array).view(cls).astype(dtype)
        if obj.ndim != 2:
            raise ValueError('2x2 contingency tables must be 2-dimensional')
        if obj.shape != (2, 2):
            raise ValueError('2x2 contingency tables must be have shape (2,2)')
        if attrs != None:
            obj.attrs = attrs
        else:
            obj.attrs = {}
        return obj

    def _abcd(self):
        """Returns the 4 elements of the contingency table

        a = True positives, b = False positives, c = False negatives,
        d = True Negatives

        """
        a, b = self[0, 0], self[0, 1]
        c, d = self[1, 0], self[1, 1]
        return a, b, c, d

    @classmethod
    def fromBoolean(cls, predicted, observed):
        """Construct a 2x2 contingency table from two boolean input arrays
        """
        pred = np.asarray(predicted).astype(bool)
        obse = np.asarray(observed).astype(bool)
        fToT = np.logical_and(pred, np.logical_and(pred, obse)).sum()
        fToF = np.logical_and(pred, np.logical_and(pred, ~obse)).sum()
        fFoT = np.logical_and(~pred, np.logical_and(~pred, obse)).sum()
        fFoF = np.logical_and(~pred, np.logical_and(~pred, ~obse)).sum()
        return cls([[fToT, fToF], [fFoT, fFoF]], attrs={'pred': pred, 'obs': obse})

    def majorityClassFraction(self):
        """Proportion Correct (a.k.a. "accuracy" in machine learning) for
        majority classifier

        """
        a, b, c, d = self._abcd()
        nmc = [0.0, 0.0]
        mc = self.sum(axis=0)
        argorder = [nmc, mc] if a < d else [mc, nmc]
        dum = self.__class__(argorder)
        self.attrs['MajorityClassFraction'] = dum.PC()
        return self.attrs['MajorityClassFraction']

    def MatthewsCC(self, ci=None):
        """Matthews Correlation Coefficient

        Examples
        ========

        >>> event_series = [ True,  True,  True, False]
        >>> pred_series  = [ True, False,  True,  True]
        >>> ct = verify.Contingency2x2.fromBoolean(pred_series, event_series)
        >>> ct.MatthewsCC()
        -0.333...
        """
        TP, FP, FN, TN = self._abcd()
        numer = np.float64((TP*TN) - (FP*FN))
        sum_ac, sum_ab = TP+FN, TP+FP
        sum_cd, sum_bd = FN+TN, FP+TN
        if (sum_ac == 0) or (sum_ab == 0) or (sum_cd == 0) or (sum_bd == 0):
            denom = np.float64(1.0)
        else:
            denom = np.sqrt(np.float64(sum_ac*sum_ab*sum_cd*sum_bd))
        self.attrs['MatthewsCC'] = numer/denom
        if ci is not None and ci == 'bootstrap':
            self.attrs['MatthewsCCCI95'] = \
                self._bootstrapCI(func='MatthewsCC')
            return (self.attrs['MatthewsCC'],
                    self.attrs['MatthewsCCCI95'])
        return self.attrs['MatthewsCC']

    def summary(self, verbose=False, ci=None):
        """ Summary table

        Parameters
        ==========
        verbose : boolean
            Print output to stdout (default=False)
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        """

        dum = self.POFD(ci=ci)
        dum = self.POD(ci=ci)
        dum = self.PC(ci=ci)
        dum = self.FAR(ci=ci)
        dum = self.heidke(ci=ci)
        dum = self.threat(ci=ci)
        dum = self.equitableThreat(ci=ci)
        dum = self.peirce(ci=ci)
        dum = self.bias(ci=ci)
        dum = self.majorityClassFraction()
        dum = self.MatthewsCC(ci=ci)
        dum = self.oddsRatio()
        dum = self.yuleQ()
        a, b, c, d = self._abcd()
        if verbose:
            print('Contingency2x2([\n  ' +
                  '[{:6g},{:6g}],\n  [{:6g},{:6g}])\n'.format(a, b, c, d))
            qual = ['MajorityClassFraction', 'MatthewsCC']
            stats = ['Bias', 'FAR', 'PC', 'POFD', 'POD', 'ThreatScore',
                     'OddsRatio']
            skill = ['HeidkeScore', 'PeirceScore', 'EquitableThreatScore',
                     'YuleQ']
            print("Summary Statistics")
            print("==================")
            for key in stats:
                if key+'CI95' in self.attrs:
                    if self.attrs[key+'CI95'].shape:
                        # bootstrapped confidence intervals, which may be
                        # asymmetric
                        print("{0}: {1:0.4f} [{2:0.4f}, {3:0.4f}]".format(key, \
                                self.attrs[key], self.attrs[key+'CI95'][0], \
                                self.attrs[key+'CI95'][1]))
                    else:
                        print("{0}: {1:0.4f} +/- {2:0.4f}".format(key, \
                                    self.attrs[key], self.attrs[key+'CI95']))
                else:
                    print("{0}: {1:0.4f}".format(key, self.attrs[key]))
            print("\nSkill Scores")
            print("============")
            for key in skill:
                if key+'CI95' in self.attrs:
                    if self.attrs[key+'CI95'].shape:
                        # bootstrapped confidence intervals, which may be
                        # asymmetric
                        print("{0}: {1:0.4f} [{2:0.4f}, {3:0.4f}]".format(key, \
                                                    self.attrs[key], \
                                                    self.attrs[key+'CI95'][0], \
                                                    self.attrs[key+'CI95'][1]))
                    else:
                        print("{0}: {1:0.4f} +/- {2:0.4f}".format(key, \
                                    self.attrs[key], self.attrs[key+'CI95']))
                else:
                    print("{0}: {1:0.4f}".format(key, self.attrs[key]))
            print("\nClassification Quality Metrics")
            print("==============================")
            for key in qual:
                if key+'CI95' in self.attrs:
                    if self.attrs[key+'CI95'].shape:
                        print("{0}: {1:0.4f} [{2:0.4f}, {3:0.4f}]".format(key, \
                                                    self.attrs[key], \
                                                    self.attrs[key+'CI95'][0], \
                                                    self.attrs[key+'CI95'][1]))
                    else:
                        print("{0}: {1:0.4f} +/- {2:0.4f}".format(key, \
                                    self.attrs[key], self.attrs[key+'CI95']))
                else:
                    print("{0}: {1:0.4f}".format(key, self.attrs[key]))

    def _bootstrapCI(self, interval=0.95, nsamp=2000, func=None, seed=1406):
        """ bootstrap confidence interval

        Parameters
        ==========
        interval : float
            Interval (default=0.95)
        nsamp : int
            Number of samples (default=2000)
        func : function
            method to calcuate quantity of interest
        seed : int
            Seed for random number generator (default=1406)

        Returns
        -------
        ps : float
            percentile
        """
        if func is None:
            raise ValueError('_bootstrapCI: a method name must be provided ' +
                             'that returns the quantity of interest')
        if seed is not None:
            np.random.seed(seed)

        bootval = np.empty(nsamp)
        if 'obs' in self.attrs and 'pred' in self.attrs:
            inds = np.random.randint(0, len(self.attrs['obs']),
                                     size=(nsamp, len(self.attrs['obs'])))
            for outidx, draw in enumerate(inds):
                tmppred = self.attrs['pred'][draw]
                tmpobs = self.attrs['obs'][draw]
                tmpct = Contingency2x2.fromBoolean(tmppred, tmpobs)
                bootval[outidx] = eval('tmpct.{}()'.format(func))
            btm = 100*(1-interval)/2.0
            tp = btm + interval*100
            ps = np.nanpercentile(bootval, [btm, tp])
            ci95 = ps[1]-ps[0]
        else:
            raise AttributeError('_bootstrapCI: Contingency2x2 must have ' +
                                 'predicted and observed boolean arrays ' +
                                 '(create with fromBoolean)')
        return ps

    def _WaldCI(self, prob, nSamp, mult=1.96):
        """ Wald confidence interval

        Parameters
        ==========
        prob : float
            Probability
        nSamp : float
            number of samples
        mult : float
            Multiplier (default=1.96)

        Returns
        =======
        ci95 : float
            confidence interval

        """
        stdErr = np.sqrt((prob*(1-prob))/nSamp) #std. error of binomial
        ci95 = mult*stdErr
        return ci95

    def _AgrestiCI(self, prob, n, mult=1.96):
        """Agresti-Coull interval. See eqn. 7.67 in Wilks [2006] (p.327)

        Parameters
        ==========
        prob : float
            Probability
        n : float
            number of samples
        mult : float
            Multiplier (default=1.96)

        Returns
        =======
        new_p : float
            new normalized probability
        ci95 : float
            confidence interval

        """
        z, zz = 1.96, 3.84
        norm = 1 + zz/n
        new_p = (prob + zz/(2*n))/norm
        pmod = (prob*(1-prob))/n
        ifac = zz/(4*n*n)
        ci95 = mult*np.sqrt(pmod + ifac)/norm
        return new_p, ci95

    def POFD(self, ci=None):
        """Calculate the Probability of False Detection (POFD), a.k.a. False
        Alarm Rate

        Parameters
        ==========
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        Returns
        =======
        pofd : float
            The probability of false detection of the contingency table data
            This is also added to the attrs attribute of the table object
        """
        a, b, c, d = self._abcd()
        n = b+d
        self.attrs['POFD'] = b/n
        if ci is not None:
            citype = 'Wald' if ci is True else ci #default to Wald CI
            if citype == 'AC':
                # note that the Agresti-Coull method also modifies the estimated
                # param
                #
                # method 2 - Agresti-Coull
                (self.attrs['POFD'],
                 self.attrs['POFDCI95']) = self._AgrestiCI(self.attrs['POFD'], n)
            elif citype == 'Wald':
                #default method - Wald interval
                self.attrs['POFDCI95'] = self._WaldCI(self.attrs['POFD'], n)
            elif citype == 'bootstrap':
                self.attrs['POFDCI95'] = self._bootstrapCI(func='POFD')
            return (self.attrs['POFD'], self.attrs['POFDCI95'])

        return self.attrs['POFD']

    def POD(self, ci=None):
        """Calculate the Probability of Detection, a.k.a. hit rate (ratio of
        correct forecasts to number of event occurrences)

        Parameters
        ==========
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        Returns
        =======
        hitrate : float
            The hit rate of the contingency table data
            This is also added to the attrs attribute of the table object
        """
        a, b, c, d = self._abcd()
        n = a+c
        self.attrs['POD'] = a/n
        if ci is not None:
            # default to Wald CI, as A-C can modify param
            citype = 'Wald' if ci is True else ci
            if citype == 'AC':
                (self.attrs['POD'],
                 self.attrs['PODCI95']) = self._AgrestiCI(self.attrs['POD'], n)
            elif citype == 'Wald':
                self.attrs['PODCI95'] = self._WaldCI(self.attrs['POD'], n)
            elif citype == 'bootstrap':
                self.attrs['PODCI95'] = self._bootstrapCI(func='POD')
            return (self.attrs['POD'], self.attrs['PODCI95'])
        return self.attrs['POD']

    def FAR(self, ci=None):
        """False Alarm Ratio, the fraction of incorrect "yes" forecasts

        Parameters
        ==========
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        Returns
        =======
        far : float
            The false alarm ratio of the contingency table data
            This is also added to the attrs attribute of the table object
        """
        a, b, c, d = self._abcd()
        n = a+b
        self.attrs['FAR'] = b/n
        if ci:
            # default to Wald CI, as A-C can modify param
            citype = 'Wald' if ci is True else ci
            if citype == 'AC':
                # note that the Agresti-Coull method also modifies the
                # estimated param
                (self.attrs['FAR'],
                 self.attrs['FARCI95']) = self._AgrestiCI(self.attrs['FAR'], n)
            elif citype == 'Wald':
                self.attrs['FARCI95'] = self._WaldCI(self.attrs['FAR'], n)
            elif citype == 'bootstrap':
                self.attrs['FARCI95'] = self._bootstrapCI(func='FAR')
            return (self.attrs['FAR'], self.attrs['FARCI95'])
        return self.attrs['FAR']

    def threat(self, ci=None):
        """Calculate the Threat Score (a.k.a. critical success index)

        Parameters
        ==========
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        Returns
        =======
        thr : float
            The threat score of the contingency table data
            This is also added to the attrs attribute of the table object

        Notes
        =====
        This is a ratio of verification, i.e., the proportion of correct
        forecasts after removing correct "no" forecasts (or 'true negatives').

        """
        a, b, c, d = self._abcd()
        self.attrs['ThreatScore'] = a/(a+b+c)
        if ci is not None and ci == 'bootstrap':
            self.attrs['ThreatScoreCI95'] = self._bootstrapCI(func='threat')
            return (self.attrs['ThreatScore'], self.attrs['ThreatScoreCI95'])
        return self.attrs['ThreatScore']

    def equitableThreat(self, ci=None):
        """Calculate the Equitable Threat Score (a.k.a. Gilbert Skill Score)

        Parameters
        ==========
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        Returns
        =======
        thr : float
            The threat score of the contingency table data
            This is also added to the attrs attribute of the table object

        Notes
        =====
        This is a ratio of verification, i.e., the proportion of correct
        forecasts after removing correct "no" forecasts (or 'true negatives').

        """
        a, b, c, d = self._abcd()
        aref = (a+b)*(a+c)/self.sum()
        self.attrs['EquitableThreatScore'] = (a-aref)/(a-aref+b+c)
        if ci is not None and ci == 'bootstrap':
            self.attrs['EquitableThreatScoreCI95'] = \
                self._bootstrapCI(func='equitableThreat')
            return (self.attrs['EquitableThreatScore'],
                    self.attrs['EquitableThreatScoreCI95'])
        return self.attrs['EquitableThreatScore']

    def heidke(self, ci=None):
        """Calculate the Heidke Skill Score for the 2x2 contingency table

        Parameters
        ==========
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        Returns
        =======
        hss : float
            The Heidke skill score of the contingency table data
            This is also added to the attrs attribute of the table object

        Notes
        ======
        This is a skill score based on the proportion of correct forecasts
        referred to the proportion expected correct by chance.

        """
        super(Contingency2x2, self).heidke()
        if ci is not None and ci == 'bootstrap':
            self.attrs['HeidkeScoreCI95'] = self._bootstrapCI(func='heidke')
            return (self.attrs['HeidkeScore'], self.attrs['HeidkeScoreCI95'])
        return self.attrs['HeidkeScore']

    def peirce(self, ci=None):
        """Calculate the Peirce Skill Score for the 2x2 contingency table

        Parameters
        ==========
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        Returns
        =======
        pss : float
            The Peirce skill score of the contingency table data
            This is also added to the attrs attribute of the table object

        """
        super(Contingency2x2, self).peirce()
        if ci is not None:
            POFD = self.POFD()
            POD = self.POD()
            a, b, c, d = self._abcd()
            nFD = b+d
            nD = a+c
            # default to Wald CI, as A-C can modify param
            citype = 'Wald' if ci is True else ci
            if ci == 'AC':
                POFD, stderrPOFD = self._AgrestiCI(self.attrs['POFD'], nFD,
                                                   mult=1)
                POD, stderrPOD = self._AgrestiCI(self.attrs['POD'], nD, mult=1)
                ci95 = 1.96*np.sqrt(stderrPOD**2+stderrPOFD**2)
            elif citype == 'Wald':
                varPOFD = (POFD*(1-POFD))/(b+d)
                varPOD = (POD*(1-POD))/(a+c)
                ci95 = 1.96*np.sqrt(varPOD+varPOFD)
            elif citype == 'bootstrap':
                ci95 = self._bootstrapCI(func='peirce')
            self.attrs['PeirceScoreCI95'] = ci95
            return (self.attrs['PeirceScore'], self.attrs['PeirceScoreCI95'])
        return self.attrs['PeirceScore']

    def PC(self, ci=None):
        """Returns the Proportion Correct (PC) for the 2x2 contingency table

        Parameters
        ==========
        ci : NoneType, str, boolean
            confidence interval (options None, 'bootstrap', True (for 'Wald'),
            or 'AC') (default=None)

        Returns
        =======
        pc: float
            Returns and updates 'PC' attribute

        """
        super(Contingency2x2, self).PC()
        if ci is not None and ci == 'bootstrap':
            self.attrs['PCCI95'] = self._bootstrapCI(func='PC')
            return (self.attrs['PC'], self.attrs['PCCI95'])
        return self.attrs['PC']

    def oddsRatio(self):
        """Calculate the odds ratio for the 2x2 contingency table

        Returns
        =======
        odds : float
             The odds ratio for the contingency table data
             This is also added to the attrs attribute of the table object
        """
        a, b, c, d = self._abcd()
        numer = a*d
        denom = b*c
        odds = numer/denom

        self.attrs['OddsRatio'] = odds
        return self.attrs['OddsRatio']

    def yuleQ(self):
        """Calculate Yule's Q (odds ratio skill score) for the 2x2 contingency
        table

        Returns
        =======
        yule : float
             Yule's Q for the contingency table data
             This is also added to the attrs attribute of the table object
        """
        odds = self.oddsRatio()
        yule = (odds-1)/(odds+1)

        self.attrs['YuleQ'] = yule
        return self.attrs['YuleQ']


    def bias(self, ci=None):
        """The frequency bias of the forecast calculated as the ratio of yes
        forecasts to number of yes events

        Returns
        =======
        bias : float
            The bias of the contingency table data
            This is also added to the attrs attribute of the table object

        Notes
        =====
        An unbiased forecast will have bias=1, showing that the number of
        forecasts is the same as the number of events. Bias>1 means that more
        events were forecast than observed (overforecast).

        """
        a, b, c, d = self._abcd()
        bias_num = a + b
        bias_den = a + c
        bias = bias_num/bias_den

        self.attrs['Bias'] = bias
        if ci is not None and ci == 'bootstrap':
            self.attrs['BiasCI95'] = self._bootstrapCI(func='bias')
            return (self.attrs['Bias'], self.attrs['BiasCI95'])
        return bias
