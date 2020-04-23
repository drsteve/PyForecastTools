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

from __future__ import division
import functools
import numpy as np

# ======= Performance metrics ======= #


def skill(A_data, A_ref, A_perf=0):
    """Generic forecast skill score for quantifying forecast improvement


    Parameters
    ==========
    A_data : float
        Accuracy measure of data set
    A_ref : float
        Accuracy measure for reference forecast
    A_perf : float, optional
        Accuracy measure for "perfect forecast" (Default = 0)

    Returns
    =======
    ss_ref : float
        Forecast skill for the given forecast, relative to the reference, using
        the chosen accuracy measure

    Notes
    =====
    See section 7.1.4 of Wilks [2006] (Statistical methods in the atmospheric
    sciences) for details.

    """
    dif1 = A_data - A_ref
    dif2 = A_perf - A_ref
    ss_ref = dif1/dif2 * 100

    return ss_ref


def percBetter(predict1, predict2, observed):
    """The percentage of cases when method A was closer to actual than method B

    Parameters
    ==========
    predict1 : array-like
        Array-like (list, numpy array, etc.) of predictions from model A
    predict2 : array-like
        Array-like (list, numpy array, etc.) of predictions from model B
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity

    Returns
    =======
    percBetter : float
        The percentage of observations where method A was closer to observation
        than method B

    Notes
    =====
    For example, if we want to know whether a new forecast performs better than
    a reference forecast...

    Examples
    ========
    >>> import verify
    >>> data = [3,4,5,6,7,8]
    >>> p_ref = [5.5]*6 #mean prediction
    >>> p_good = [4,5,4,7,7,8] #"good" model prediction
    >>> verify.percBetter(p_good, p_ref, data)
    66.66666666666666

    That is, two-thirds (66.67%) of the predictions have a lower absolute error
    in p_good than in p_ref.

    """
    # set up inputs
    methA = _maskSeries(predict1)
    methB = _maskSeries(predict2)
    data = _maskSeries(observed)
    # get forecast errors
    errA = forecastError(methA, data, full=False)
    errB = forecastError(methB, data, full=False)
    # exclude ties & count cases where A beats B (smaller error)
    countBetter = (np.abs(errA) < np.abs(errB)).sum()
    numCases = len(methA)
    fracBetter = countBetter/numCases

    return 100*fracBetter

# ======= Bias measures ======= #


def bias(predicted, observed):
    """ Scale-dependent bias as measured by the mean error

    Parameters
    ==========
    predicted : array-like
        Array-like (list, numpy array, etc.) of predictions
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity

    Returns
    =======
    bias : float
        Mean error of prediction

    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)

    return pred.mean()-obse.mean()


def meanPercentageError(predicted, observed):
    """Order-dependent bias as measured by the mean percentage error

    Parameters
    ==========
    predicted : array-like
        Array-like (list, numpy array, etc.) of predictions
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity

    Returns
    =======
    mpe : float
        Mean percentage error of prediction

    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)
    pe = percError(pred, obse)
    mpe = pe.mean()
    return mpe


def medianLogAccuracy(predicted, observed, mfunc=np.median, base=10):
    """Order-dependent bias as measured by the median of the log accuracy ratio

    Parameters
    ==========
    predicted : array-like
        Array-like (list, numpy array, etc.) of predictions
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity
    mfunc : function, optional
        Function to use for central tendency (default: numpy.median)
    base : number, optional
        Base to use for logarithmic transform (default: 10)

    Returns
    =======
    mla : float
        Median log accuracy of prediction

    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)
    la = logAccuracy(pred, obse, base=base)
    mla = mfunc(la.compressed())

    return mla


def symmetricSignedBias(predicted, observed):
    """Symmetric signed bias, expressed as a percentage

    Parameters
    ==========
    predicted : array-like
        List of predicted values
    observed : array-like
        List of observed values

    Returns
    =======
    bias : float
        symmetric signed bias, as a precentage
    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)
    mla = medianLogAccuracy(pred, obse, base='e')
    biasmag = np.exp(np.abs(mla))-1
    # apply sign of mla to symmetric bias magnitude
    ssb = np.copysign(biasmag, mla)
    return 100*ssb

# ======= Accuracy measures ======= #


def accuracy(data, climate=None):
    """Convenience function to calculate a selection of unscaled accuracy
    measures

    Parameters
    ==========
    data : array-like
        Array-like (list, numpy array, etc.) of predictions
    climate : array-like or float, optional
        Array-like (list, numpy array, etc.) or float of observed values of
        scalar quantity. If climate is None (default) then the accuracy is
        assessed relative to persistence.

    Returns
    =======
    out : dict
        Dictionary containing unscaled accuracy measures
        MSE - mean squared error
        RMSE - root mean squared error
        MAE - mean absolute error
        MdAE - median absolute error

    See Also
    ========
    meanSquaredError, RMSE, meanAbsError, medAbsError

    """

    data = _maskSeries(data)
    if climate is not None:
        climate = _maskSeries(climate)
    metrics = {'MSE': meanSquaredError, 'RMSE': RMSE,
               'MAE': meanAbsError, 'MdAE': medAbsError}
    out = dict()
    for met in metrics:
        out[met] = metrics[met](data, climate)

    return out


def meanSquaredError(data, climate=None):
    """Mean squared error of a data set relative to a reference value

    Parameters
    ==========
    data : array-like
        data to calculate mean squared error, default reference is persistence
    climate : array-like or float, optional
        Array-like (list, numpy array, etc.) or float of observed values of
        scalar quantity.  If climate is None (default) then the accuracy is
        assessed relative to persistence.

    Returns
    =======
    out : float
        the mean-squared-error of the data set relative to the chosen reference

    See Also
    ========
    RMSE, meanAbsError

    Notes
    =====
    The chosen reference can be persistence, a provided climatological mean
    (scalar), or a provided climatology (observation vector).

    """
    dat = _maskSeries(data)
    dif = _diff(dat, climate=climate)

    dif2 = dif**2.0

    return dif2.mean()


def RMSE(data, climate=None):
    """Calcualte the root mean squared error of a data set relative to a
    reference value

    Parameters
    ==========
    data : array-like
        data to calculate mean squared error, default reference is persistence
    climate : array-like or float, optional
        Array-like (list, numpy array, etc.) or float of observed values of
        scalar quantity. If climate is None (default) then the accuracy is
        assessed relative to persistence.

    Returns
    =======
    out : float
        the root-mean-squared error of the data set relative to the chosen
        reference

    See Also
    ========
    meanSquaredError, meanAbsError

    Notes
    =====
    The chosen reference can be persistence, a provided climatological mean
    (scalar) or a provided climatology (observation vector).

    """
    data = _maskSeries(data)
    msqerr = meanSquaredError(data, climate=climate)

    return np.sqrt(msqerr)


def meanAbsError(data, climate=None):
    """mean absolute error of a data set relative to some reference value

    Parameters
    ==========
    data : array-like
        data to calculate mean squared error, default reference is persistence
    climate : array-like or float, optional
        Array-like (list, numpy array, etc.) or float of observed values of
        scalar quantity.  If climate is None (default) then the accuracy is
        assessed relative to persistence.

    Returns
    =======
    out : float
        the mean absolute error of the data set relative to the chosen reference

    See Also
    ========
    medAbsError, meanSquaredError, RMSE

    Notes
    =====
    The chosen reference can be persistence, a provided climatological mean
    (scalar) or a provided climatology (observation vector).

    """
    data = _maskSeries(data)
    adif = np.abs(_diff(data, climate=climate))

    return adif.mean()


def medAbsError(data, climate=None):
    """median absolute error of a data set relative to some reference value

    Parameters
    ==========
    data : array-like
        data to calculate median absolute error, default reference is
        persistence
    climate : array-like or float, optional
        Array-like (list, numpy array, etc.) or float of observed values of
        scalar quantity.  If climate is None (default) then the accuracy is
        assessed relative to persistence.

    Returns
    =======
    out : float
        the median absolute error of the data set relative to the chosen
        reference

    See Also
    ========
    meanAbsError, meanSquaredError, RMSE

    Notes
    =====
    The chosen reference can be persistence, a provided climatological mean
    (scalar) or a provided climatology (observation vector).

    """
    dat = _maskSeries(data)
    dif = _diff(dat, climate=climate)
    MdAE = np.median(np.abs(dif))

    return MdAE


# ======= Scaled/Relative Accuracy measures ======= #
def scaledAccuracy(predicted, observed):
    """ Calculate scaled and relative accuracy measures

    Parameters
    ==========
    predicted: array-like
        Array-like (list, numpy array, etc.) of predictions
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity

    Returns
    =======
    out : dict
        Dictionary containing scaled or relative accuracy measures
        nRMSE - normalized root mean squared error
        MASE - mean absolute scaled error
        MAPE - mean absolute percentage error
        MdAPE - median absolute percentage error
        MdSymAcc - median symmetric accuracy

    See Also
    ========
    medSymAccuracy, meanAPE, MASE, nRMSE

    """
    metrics = {'nRMSE': nRMSE, 'MASE': MASE, 'MAPE': meanAPE,
               'MdAPE': functools.partial(meanAPE, mfunc=np.median),
               'MdSymAcc': medSymAccuracy}
    out = dict()
    for met in metrics:
        out[met] = metrics[met](predicted, observed)

    return out


def nRMSE(predicted, observed):
    """normalized root mean squared error of a data set relative to a reference
    value

    Parameters
    ==========
    predicted: array-like
        predicted data for which to calculate mean squared error
    observed: float
        observation vector (or climatological value (scalar)) to use as
        reference value

    Returns
    =======
    out : float
        the normalized root-mean-squared-error of the data set relative to the
        observations

    See Also
    ========
    RMSE

    Notes
    =====
    The chosen reference can be an observation vector or, a provided
    climatological mean (scalar). This definition is due to Yu and Ridley (2002)

    References:
    Yu, Y., and A. J. Ridley (2008), Validation of the space weather modeling
    framework using ground-based magnetometers, Space Weather, 6, S05002,
    doi:10.1029/2007SW000345.

    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)
    n_pts = len(pred)
    dif = _diff(pred, climate=obse)

    dif2 = dif**2.0
    sig_dif2 = dif2.sum()

    if len(obse) == 1:
        obse = np.asanyarray(obse).repeat(n_pts)

    norm = np.sum(obse**2.0)

    return sig_dif2/norm


def scaledError(predicted, observed):
    """Scaled errors, see Hyndman and Koehler (2006)

    Parameters
    ==========
    predicted: array-like
        predicted data for which to calculate scaled error
    observed: float
        observation vector (or climatological value (scalar)) to use as
        reference value

    Returns
    =======
    q : float
        the scaled error

    Notes
    =====
    References:
    R.J. Hyndman and A.B. Koehler, Another look at measures of forecast
    accuracy, Intl. J. Forecasting, 22, pp. 679-688, 2006.

    See Also
    ========
    MASE

    """
    pred = np.asanyarray(predicted).astype(float)
    obse = np.asanyarray(observed).astype(float)

    n_pts = len(pred.ravel())

    if len(obse) == 1:
        obse = np.asanyarray(observed).repeat(n_pts)

    dif = _diff(pred, obse)
    dsum = np.sum(np.abs(np.diff(obse)))

    q = dif/((1/(n_pts-1))*dsum)

    return q


def MASE(predicted, observed):
    """Mean Absolute Scaled Error

    Parameters
    ==========
    predicted: array-like
        predicted data for which to calculate MASE
    observed: float
        observation vector (or climatological value (scalar)) to use as
        reference value

    Returns
    =======
    out : float
        the mean absolute scaled error of the data set

    See Also
    ========
    scaledError

    Notes
    =====
    References:
    R.J. Hyndman and A.B. Koehler, Another look at measures of forecast
    accuracy, Intl. J. Forecasting, 22, pp. 679-688, 2006.

    """
    pred = np.asanyarray(predicted).astype(float)
    obse = np.asanyarray(observed).astype(float)

    q = scaledError(pred, obse)
    n_pts = len(pred.ravel())
    return np.abs(q).sum()/n_pts


def forecastError(predicted, observed, full=True):
    """forecast error, defined using the sign convention of J&S ch. 5

    Parameters
    ==========
    predicted : array-like
        Array-like (list, numpy array, etc.) of predictions
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity
    full : boolean, optional
        Switch determining nature of return value. When it is True (the
        default) the function returns the errors as well as the predicted
        and observed values as numpy arrays of floats, when False only the
        array of forecast errors is returned.

    Returns
    =======
    err : array
        the forecast error
    pred : array
        Optional return array of predicted values as floats, included if full
        is True
    obse : array
        Optional return array of observed values as floats, included if full
        is True

    Notes
    =====
    J&S: Jolliffe and Stephenson (Ch. 5)

    """
    pred = np.asanyarray(predicted).astype(float)
    obse = np.asanyarray(observed).astype(float)
    err = pred-obse
    if full:
        return err, pred, obse
    else:
        return err


def percError(predicted, observed):
    """Percentage Error

    Parameters
    ==========
    predicted : array-like
        Array-like (list, numpy array, etc.) of predictions
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity

    Returns
    =======
    perc : float
        Array of forecast errors expressed as a percentage

    """
    err, pred, obse = forecastError(predicted, observed, full=True)
    res = err/obse
    return 100*res


def absPercError(predicted, observed):
    """Absolute percentage error

    Parameters
    ==========
    predicted : array-like
        Array-like (list, numpy array, etc.) of predictions
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity

    Returns
    =======
    perc : array
        Array of absolute percentage errors

    """
    err, pred, obse = forecastError(predicted, observed, full=True)
    res = np.abs(err/obse)
    return 100*res


def logAccuracy(predicted, observed, base=10, mask=True):
    """Log Accuracy Ratio, defined as log(predicted/observed) or
    log(predicted)-log(observed)

    Parameters
    ==========
    predicted : array-like
        Array-like (list, numpy array, etc.) of predictions
    observed : array-like
        Array-like (list, numpy array, etc.) of observed values of scalar
        quantity
    base : number, optional
        Base to use for logarithmic transform (allows 10, 2, and 'e')
       (default=10)
    mask : boolean, optional
        Switch to set masking behaviour. If True (default) the function will
        mask out NaN and negative values, and will return a masked array. If
        False, the presence of negative numbers will raise a ValueError and
        NaN will propagate through the calculation.

    Returns
    =======
    logacc : array or masked array
        Array of absolute percentage errors

    Notes
    =====
    Using base 2 is computationally much faster, so unless the base is
    important to interpretation we recommend using that.

    """
    if mask:
        pred = _maskSeries(predicted)
        obse = _maskSeries(observed)
        negs_p = predicted <= 0
        negs_o = observed <= 0
        pred.mask = np.logical_or(pred.mask, negs_p)
        obse.mask = np.logical_or(obse.mask, negs_o)
    else:
        pred = np.asanyarray(predicted)
        obse = np.asanyarray(observed)
    # check for positivity
    if (pred <= 0).any() or (obse <= 0).any():
        raise ValueError('logAccuracy: input data are required to be positive')
    logfuncs = {10: np.log10, 2: np.log2, 'e': np.log}
    if base not in logfuncs:
        supportedbases = '[' + ', '.join([str(key) for key in logfuncs]) + ']'
        raise NotImplementedError('logAccuracy: Selected base ' +
                                  '({0}) for logarithms not'.format(base) +
                                  ' supported. Supported values are ' +
                                  '{0}'.format(supportedbases))
    logacc = logfuncs[base](pred/obse)
    return logacc


def medSymAccuracy(predicted, observed, mfunc=np.median, method=None):
    """Scaled measure of accuracy that is not biased to over- or
    under-predictions.

    Parameters
    ==========
    predicted: array-like
        predicted data for which to calculate mean squared error
    observed: float
        observation vector (or climatological value (scalar)) to use as
        reference value
    mfunc : function
        function for calculating the median (default=np.median)
    method : string, optional
        Method to use for calculating the median symmetric accuracy (MSA).
        Options are 'log' which uses the median of the re-exponentiated
        absolute log accuracy, 'UPE' which calculates MSA using the unsigned
        percentage error, and None (default), in which case the method is
        implemented as described above. The UPE method has reduced accuracy
        compared to the other methods and is included primarily for testing
        purposes.

    Returns
    =======
    msa : float
        Array of median symmetric accuracy

    Notes
    =====
    The accuracy ratio is given by (prediction/observation), to avoid the bias
    inherent in mean/median percentage error metrics we use the log of the
    accuracy ratio (which is symmetric about 0 for changes of the same factor).
    Specifically, the Median Symmetric Accuracy is found by calculating the
    median of the absolute log accuracy, and re-exponentiating:
    g = exp( median( |ln(pred) - ln(obs)| ) )

    This can be expressed as a symmetric percentage error by shifting by one
    unit and multiplying by 100:
    MSA = 100*(g-1)

    It can also be shown that this is identically equivalent to the median
    unsigned percentage error, where the unsigned relative error is given by:
    (y' - x')/x'

    where y' is always the larger of the (observation, prediction) pair, and x'
    is always the smaller.

    Reference:
    Morley, S.K. (2016), Alternatives to accuracy and bias metrics based on
    percentage errors for radiation belt modeling applications, Los Alamos
    National Laboratory Report, LA-UR-15-24592.

    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)
    if method is None:
        # is this different from the method above for large series??
        absLogAcc = np.abs(logAccuracy(pred, obse, base=2))
        symAcc = np.exp2(mfunc(absLogAcc.compressed()))
        msa = 100*(symAcc-1)
    elif method == 'log':
        # median(log(Q)) method
        absLogAcc = np.abs(logAccuracy(pred, obse, base=2))
        symAcc = mfunc(np.exp2(absLogAcc.compressed()))
        msa = 100*(symAcc-1)
    elif method == 'UPE':
        # unsigned percentage error method
        PleO = pred >= obse
        OltP = np.logical_not(PleO)
        unsRelErr = pred.copy()
        unsRelErr[PleO] = (pred[PleO]-obse[PleO])/obse[PleO]
        unsRelErr[OltP] = (obse[OltP]-pred[OltP])/pred[OltP]
        unsPercErr = unsRelErr*100
        msa = mfunc(unsPercErr.compressed())
    else:
        raise NotImplementedError('medSymAccuracy: invalid method {0}. Valid ' +
                                  'options are None, "log" or ' +
                                  '"UPE".'.format(method))

    return msa


def meanAPE(predicted, observed, mfunc=np.mean):
    """ mean absolute percentage error

    Parameters
    ==========
    predicted: array-like
        predicted data for which to calculate mean squared error
    observed: float
        observation vector (or climatological value (scalar)) to use as
        reference value
    mfunc : function
        function to calculate mean (default=np.mean)

    Returns
    =======
    mape : float
        the mean absolute percentage error

    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)
    e_perc = np.abs(percError(pred, obse))
    return mfunc(e_perc.compressed())


# def S1Score():
#     pass


# ======= Precision (scale) Measures ======= #
def medAbsDev(series, scale=False, median=False):
    """ Computes the median absolute deviation from the median

    Parameters
    ==========
    series : array-like
        Input data
    scale : boolean
        Scale so that median absolute deviation is the same as the standard
        deviation for normal distributions (default=False)
    median : boolean
        Return the median of the series as well as the median absolute deviation
        (default=False)

    Returns
    =======
    mad : float
        median absolute deviation
    perc50 : float
        median of series, optional output

    """
    series = _maskSeries(series)
    # get median absolute deviation of unmasked elements
    perc50 = np.median(series.compressed())
    mad = np.median(abs(series.compressed()-perc50))
    if scale:
        mad *= 1.4826  # scale so that MAD is same as SD for normal distr.
    if median:
        return mad, perc50
    else:
        return mad


def rSD(predicted):
    """ robust standard deviation

    Parameters
    ==========
    predicted : array-like
        Predicted input

    Returns
    =======
    rsd : float
        robust standard deviation, the scaled med abs dev

    Notes
    =====
    Computes the "robust standard deviation", i.e. the median absolute
    deviation times a correction factor

    The median absolute deviation (medAbsDev) scaled by a factor of 1.4826
    recovers the standard deviation when applied to a normal distribution.
    However, unlike the standard deviation the medAbsDev has a high breakdown
    point and is therefore considered a robust estimator.

    """
    return medAbsDev(predicted, scale=True)


def rCV(predicted):
    """ robust coefficient of variation

    Parameters
    ==========
    predicted : array-like
        Predicted input

    Returns
    =======
    rcv : float
        robust coefficient of variation (see notes)

    Notes
    =====
    Computes the "robust coefficient of variation", i.e. median absolute
    deviation divided by the median

    By analogy with the coefficient of variation, which is the standard
    deviation divided by the mean, rCV gives the median absolute deviation
    (aka rSD) divided by the median, thereby providing a scaled measure
    of precision/spread.

    """
    mad, perc50 = medAbsDev(predicted, scale=True, median=True)

    return mad/perc50


def Sn(data, scale=True, correct=True):
    """Sn statistic, a robust measure of scale

    Parameters
    ==========
    data : array-like
        data to calculate Sn statistic for
    scale : boolean
        Scale so that output is the same as the standard deviation for if the
        distribution is normal (default=True)
        (default=True)
    correct : boolean
        Set a correction factor (default=True)

    Returns
    =======
    Sn : float
        the Sn statistic

    See Also
    ========
    medAbsDev

    Notes
    =====
    Sn is more efficient than the median absolute deviation, and is not
    constructed with the assumption of a symmetric distribution, because it
    does not measure distance from an assumed central location. To quote RC1993,
    "...Sn looks at a typical distance between observations,  which is still
    valid at asymmetric distributions."

    [RC1993] P.J.Rouseeuw and C.Croux, "Alternatives to the Median Absolute
    Deviation", J. Amer. Stat. Assoc., 88 (424), pp.1273-1283. Equation 2.1,
    but note that they use "low" and "high" medians:
    Sn = c * 1.1926 * LOMED_{i} ( HIMED_{j} (|x_i - x_j|) )

    Note that the implementation of the original formulation is slow for large
    n. As the original formulation is identical to using a true median for
    odd-length series, we do so here automatically to gain a significant
    speedup.

    """
    # Define local utility functions
    def dropPoint(vec, i):
        if i == 0: return vec[1:]
        elif i == len(vec): return vec[:-1]
        else:
            vec1 = vec.tolist()
            vec1[:i].extend(vec1[i+1:])
            return vec1

    def lowmed(vec):
        lenvec = len(vec)
        if len(vec) % 2:  # odd
            ind = int(lenvec//2)
        else:
            ind = int(lenvec/2.0)-1
        q = np.partition(vec, ind)
        return q[ind]

    def highmed(vec):
        if len(vec) % 2:
            ind = int(len(vec)//2)
        else:
            ind = int(np.ceil(len(vec)/2))
        q = np.partition(vec, ind)
        return q[ind]

    series = _maskSeries(data)
    series = series.compressed()
    series.sort()
    n_pts = len(series)
    seriesodd = True if (n_pts % 2 == 1) else False
    # odd number of points, so use true median (shouldn't make a difference...
    # but seems to)
    truemedian = seriesodd
    # truemedian = False
    imd = np.empty(n_pts)
    # for each i, find median of absolute differences
    for i in range(n_pts):
        if truemedian:
            imd[i] = np.median(np.abs(series[i]-series))
        else:
            # tmp_series = np.asarray(dropPoint(series, i))
            # imd[i] = lowmed(np.abs(series[i]-tmp_series))
            imd[i] = highmed(np.abs(series[i]-series))
    # find median of result, then scale to std. normal
    if truemedian:
        Sn_imd = np.median(imd)
    else:
        Sn_imd = lowmed(imd)  # "low median"
    sfac = 1.1926 if scale else 1.0
    Sn_imd *= sfac
    # now set correction factor
    cn = 1.0
    cfac = [1.0, 0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131]
    if correct:
        if n_pts <= 9:
            cn = cfac[n_pts-1]
        elif seriesodd:  # n odd, >= 11
            cn = n_pts/(n_pts - 0.9)
    # else n=1, or n>=10 and even
    return Sn_imd * cn


def normSn(data, **kwargs):
    """ Computes the normalized Sn statistic, a scaled measure of spread.

    Parameters
    ==========
    data : array-like
        data to calculate normSn statistic for
    **kwards : dict
        Optional keyword arguements (see Sn)

    Returns
    =======
    normSn : float
        the normalized Sn statistic

    See Also
    ========
    rCV

    Notes
    =====
    We here scale the Sn estimator by the median, giving a non-symmetric
    alternative to the robust coefficient of variation (rCV).

    """
    series = _maskSeries(data)
    p50 = np.median(series.compressed())
    return Sn(data, **kwargs)/p50

# ======= Other useful functions ======= #


def _diff(data, climate=None):
    """ difference utility function

    Parameters
    ==========
    data : np.array
        Array of data values
    climate : NoneType
        Array or scalar of climatology values, or None  to calcuate difference
        of data using np.diff (default=None)

    Returns
    =======
    dif : float
        Difference between data and climate or between data if climate is None

    """
    fc = np.asarray(data)
    clim = np.asarray(climate)

    try:
        assert fc.ndim <= 2
    except:
        raise ValueError('Input data set must be of rank 2 or less')
    else:
        n_pts = len(fc)

    if climate is not None:
        if clim.ndim == 0:
            clim = clim.tolist()
        elif len(clim) == 1:
            # climate is a scalar
            climate = climate[0]
        else:
            try:
                assert clim.shape == fc.shape
            except:
                AssertionError('If climate is not scalar, it must have the ' +
                               'same shape as data')

        dif = fc - clim  # climate - data
    else:
        dif = np.diff(fc)

    return dif


def _maskSeries(series):
    """ Mask the NaN in array-like input

    Parameters
    ==========
    series : array-like
        Array-like input

    Returns
    =======
    ser : numpy array
        Numpy array with NaN values masked

    """
    # ensure input is numpy array (and make 1-D)
    ser = np.asarray(series, dtype=float).ravel()
    # mask for NaNs
    ser = np.ma.masked_where(np.isnan(ser), ser)
    return ser
