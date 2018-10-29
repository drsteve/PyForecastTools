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

try:
    from spacepy import datamodel as dm
except:
    from . import datamodel as dm


#======= Performance metrics =======#

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
    #set up inputs
    methA = _maskSeries(predict1)
    methB = _maskSeries(predict2)
    data = _maskSeries(observed)
    #get forecast errors
    errA = forecastError(methA, data, full=False)
    errB = forecastError(methB, data, full=False)
    #exclude ties & count cases where A beats B (smaller error)
    countBetter = (np.abs(errA) < np.abs(errB)).sum()
    numCases = len(methA)
    fracBetter = countBetter/numCases
    
    return 100*fracBetter


#======= Bias measures =======#
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
    pred =  _maskSeries(predicted)
    obse =  _maskSeries(observed)
    
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
    pred =  _maskSeries(predicted)
    obse =  _maskSeries(observed)
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
    pred =  _maskSeries(predicted)
    obse =  _maskSeries(observed)
    la = logAccuracy(pred, obse, base=base)
    mla = mfunc(la)

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
    pred =  _maskSeries(predicted)
    obse =  _maskSeries(observed)
    mla = medianLogAccuracy(pred, obse, base='e')
    sign = np.sign(mla)
    biasmag = np.exp(np.abs(mla))-1
    # apply sign of mla to symmetric bias magnitude
    ssb = np.copysign(biasmag, mla)
    return 100*ssb
   

#======= Accuracy measures =======#

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
        clim = _maskSeries(climate)
    metrics = {'MSE': meanSquaredError, 'RMSE': RMSE,
               'MAE':meanAbsError, 'MdAE': medAbsError}
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
    n_pts = len(dat)
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
    dat = _maskSeries(data)
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
    data =  _maskSeries(data)
    n_pts = len(data)
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
    n_pts = len(dat)
    dif = _diff(dat, climate=climate)
    MdAE = np.median(np.abs(dif))

    return MdAE


#======= Scaled/Relative Accuracy measures =======#
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
    pred =  _maskSeries(predicted)
    obse =  _maskSeries(observed)
    n_pts = len(pred)
    dif = _diff(pred, climate=obse)

    dif2 = dif**2.0
    sig_dif2 = dif2.sum()

    if len(obse)==1:
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

    if len(obse)==1:
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
    err : float
        the mean absolute scaled error
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
    #check for positivity
    if (pred<=0).any() or (obse<=0).any():
        raise ValueError('logAccuracy: input data are required to be positive')
    logfuncs = {10: np.log10, 2: np.log2, 'e': np.log}
    if base not in logfuncs:
        supportedbases = '['+ ', '.join([str(key) for key in logfuncs]) + ']'
        raise NotImplementedError('logAccuracy: Selected base ' +
                                  '({0}) for logarithms not'.format(base) +
                                  ' supported. Supported values are ' +
                                  '{0}'.format(supportedbases))
    logacc = logfuncs[base](pred/obse)
    if mask:
        return logfuncs[base](pred/obse)
    else:
        
        return logfuncs[base](pred/obse)

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
        symAcc = np.exp2(mfunc(absLogAcc))
        msa = 100*(symAcc-1)
    elif method=='log':
        ##median(log(Q)) method
        absLogAcc = np.abs(logAccuracy(pred, obse, base=2))
        symAcc = mfunc(np.exp2(absLogAcc))
        msa = 100*(symAcc-1)
    elif method=='UPE':
        ##unsigned percentage error method
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


#======= Precision (scale) Measures =======#
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
    #get median absolute deviation of unmasked elements
    perc50 = np.median(series.compressed())
    mad = np.median(abs(series.compressed()-perc50))
    if scale:
        mad *= 1.4826 #scale so that MAD is same as SD for normal distr.
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
        if i==0: return vec[1:]
        elif i==len(vec): return vec[:-1]
        else:
            vec1 = vec.tolist()
            vec1[:i].extend(vec1[i+1:])
            return vec1

    def lowmed(vec):
        lenvec = len(vec)
        if len(vec)%2: #odd
           ind = int(lenvec//2)
        else:
            ind = int(lenvec/2.0)-1
        q = np.partition(vec, ind)
        return q[ind]

    def highmed(vec):
        if len(vec)%2:
            ind = int(len(vec)//2)
        else:
            ind = int(np.ceil(len(vec)/2))
        q = np.partition(vec, ind)
        return q[ind]

    series = _maskSeries(data)
    series.compressed().sort()
    n_pts = len(series)
    seriesodd = True if (n_pts%2==1) else False
    #odd number of points, so use true median (shouldn't make a difference...
    # but seems to)
    truemedian = seriesodd
    #truemedian = False
    imd = np.empty(n_pts)
    #for each i, find median of absolute differences
    for i in range(n_pts):
        if truemedian:
            imd[i] = np.median(np.abs(series[i]-series))
        else:
            #tmp_series = np.asarray(dropPoint(series, i))
            #imd[i] = lowmed(np.abs(series[i]-tmp_series))
            imd[i] = highmed(np.abs(series[i]-series))
    #find median of result, then scale to std. normal
    if truemedian:
        Sn_imd = np.median(imd)
    else:
        Sn_imd = lowmed(imd) #"low median"
    sfac = 1.1926 if scale else 1.0
    Sn_imd *= sfac
    #now set correction factor
    cn = 1.0
    cfac = [1.0, 0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131]
    if correct:
        if (n_pts <= 9):
            cn = cfac[n_pts-1]
        elif seriesodd: #n odd, >= 11
            cn = n_pts/(n_pts - 0.9)
    #else n=1, or n>=10 and even
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
            skill = ['HeidkeScore','PeirceScore']
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
                Py += self[i,j]
                Po += self[j,i]
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
                Py += self[i,j]
                Po += self[j,i]
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
        b = self[category,:].sum() - a
        c = self[:,category].sum() - a
        d = self.sum()-self[category,:].sum() - self[:,category].sum() + a
        return Contingency2x2([[a,b],[c,d]])
            

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
    def __new__(cls, input_array, attrs=None, dtype=None, stats=False):
        if not dtype:
            obj = np.asarray(input_array).view(cls)
        else:
            obj = np.asarray(input_array).view(cls).astype(dtype)
        if obj.ndim != 2:
            raise ValueError('2x2 contingency tables must be 2-dimensional')
        if obj.shape != (2,2):
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
        a, b = self[0,0], self[0,1]
        c, d = self[1,0], self[1,1]
        return a,b,c,d

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
        return cls([[fToT, fToF],[fFoT, fFoF]], attrs={'pred':pred, 'obs':obse})

    def majorityClassFraction(self):
        """Proportion Correct (a.k.a. "accuracy" in machine learning) for
        majority classifier
        
        """
        a,b,c,d = self._abcd()
        nmc = [0.0, 0.0]
        mc = self.sum(axis=0)
        argorder = [nmc, mc] if a<d else [mc,nmc]
        dum = self.__class__(argorder)
        self.attrs['MajorityClassFraction'] = dum.PC()
        return self.attrs['MajorityClassFraction']

    def MatthewsCC(self):
        """Matthews Correlation Coefficient

        Examples
        ========

        >>> event_series = [ True,  True,  True, False]
        >>> pred_series  = [ True, False,  True,  True]
        >>> ct = verify.Contingency2x2.fromBoolean(pred_series, event_series)
        >>> ct.MatthewsCC()
        -0.333...
        """
        TP,FP,FN,TN = self._abcd()
        numer = np.float64((TP*TN) - (FP*FN))
        sum_ac, sum_ab = TP+FN, TP+FP
        sum_cd, sum_bd = FN+TN, FP+TN
        if (sum_ac==0) or (sum_ab==0) or (sum_cd==0) or (sum_bd==0):
            denom = np.float64(1.0)
        else:
            denom = np.sqrt(np.float64(sum_ac*sum_ab*sum_cd*sum_bd))
        self.attrs['MatthewsCC'] = numer/denom
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
        dum = self.MatthewsCC()
        dum = self.oddsRatio()
        dum = self.yuleQ()
        a,b,c,d = self._abcd()
        if verbose:
            print('Contingency2x2([\n  ' +
                  '[{:6g},{:6g}],\n  [{:6g},{:6g}])\n'.format(a,b,c,d))
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
                                     size=(nsamp,len(self.attrs['obs'])))
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

    def _WaldCI(self, prob, n, mult=1.96):
        """ Wald confidence interval

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
        ci95 : float
            confidence interval

        """
        stdErr = np.sqrt((prob*(1-prob))/n) #std. error of binomial
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
        a,b,c,d = self._abcd()
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
                 self.attrs['POFDCI95']) = self._AgrestiCI(self.attrs['POFD'],n)
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
        a,b,c,d = self._abcd()
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
        a,b,c,d = self._abcd()
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
        a,b,c,d = self._abcd()
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
        a,b,c,d = self._abcd()
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
            a,b,c,d = self._abcd()
            nFD = b+d
            nD  = a+c
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
        a,b,c,d = self._abcd()
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
        a,b,c,d = self._abcd()
        bias_num = a + b
        bias_den = a + c
        bias = bias_num/bias_den
        
        self.attrs['Bias'] = bias
        if ci is not None and ci == 'bootstrap':
            self.attrs['BiasCI95'] = self._bootstrapCI(func='bias')
            return (self.attrs['Bias'], self.attrs['BiasCI95'])
        return bias



#======= Other useful functions =======#

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
        if clim.ndim ==0:
            clim = clim.tolist()
        elif len(clim)==1:
            #climate is a scalar
            climate = climate[0]
        else:
            try:
                assert clim.shape==fc.shape
            except:
                AssertionError('If climate is not scalar, it must have the ' +
                               'same shape as data')
               
        dif = fc - clim #climate - data
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
    #ensure input is numpy array (and make 1-D)
    ser = np.asarray(series, dtype=float).ravel()
    #mask for NaNs
    ser = np.ma.masked_where(np.isnan(ser), ser)
    return ser
