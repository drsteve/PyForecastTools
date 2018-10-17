"""
Sub-module to provide simple plotting for exploratory data analysis
"""
import numpy
import matplotlib.pyplot as plt

from .metrics import _maskSeries
from .categorical import Contingency2x2

def set_target(target, figsize=None, loc=111, polar=False):
    """
    Given a *target* on which to plot a figure, determine if that *target*
    is **None** or a matplotlib figure or axes object.  Based on the type
    of *target*, a figure and/or axes will be either located or generated.
    Both the figure and axes objects are returned to the caller for further
    manipulation.

    Parameters
    ==========
    target : object
        The object on which plotting will happen.

    Other Parameters
    ================
    figsize : tuple
        A two-item tuple/list giving the dimensions of the figure, in inches.  
        Defaults to Matplotlib defaults.
    loc : integer 
        The subplot triple that specifies the location of the axes object.  
        Defaults to 111.
    polar : bool
        Set the axes object to polar coodinates.  Defaults to **False**.

    Returns
    =======
    fig : object
      A matplotlib figure object on which to plot.
    ax : object
      A matplotlib subplot object on which to plot.

    Examples
    ========
    >>> import matplotlib.pyplot as plt
    >>> from verify.plot import set_target
    >>> fig = plt.figure()
    >>> fig, ax = set_target(target=fig, loc=211)

    Notes
    =====
    Implementation from SpacePy's plot module.
    """
    # Is target a figure?  Make a new axes.
    if type(target) == plt.Figure:
        fig = target
        ax  = fig.add_subplot(loc, polar=polar)
    # Is target an axes?  Make no new items.
    elif issubclass(type(target), plt.Axes):
        ax  = target
        fig = ax.figure
    # Is target something else?  Make new everything.
    else:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(loc, polar=polar)
    return fig, ax


def qqplot(predicted, observed, xyline=True, target=None, plot_kwargs={}):
    """Quantile-quantile plot for predictions and observations

    Parameters
    ==========
    predicted :  array-like
        predicted data for which to calculate mean squared error
    observed : array-like
        observation vector (or climatological value (scalar)) to use as
        reference value

    Other Parameters
    ================
    xyline : boolean
        Toggles the display of a line of y=x (perfect model). Default True.
    target : figure, axes, or None
        The object on which plotting will happen. If **None** (default)
        then a figure an axes will be created. If a matplotlib figure is
        supplied a set of axes will be made, and if matplotlib axes are 
        given then the plot will be made on those axes.
    plot_kwargs : dict
        Dictionary containing plot keyword arguments to pass to matplotlib's
        scatter function.

    Returns
    =======
    out_dict : dict
        A dictionary containing the Figure and the Axes

    Example
    =======
    >>> import numpy as np
    >>> import verify
    >>> np.random.seed(46)
    >>> predicted = np.random.randint(0,40,101)
    >>> observed = predicted + np.random.randn(101)
    >>> observed = observed[:71] #QQ plots don't require even smaple lengths
    >>> plot_settings = {'marker': 'X', 'c': np.arange(71), 'cmap':'plasma'}
    >>> verify.plot.qqplot(predicted, observed, plot_kwargs=plot_settings)

    Notes
    =====
    The q-q plot is formed by:
    - Vertical axis: Estimated quantiles from observed data
    - Horizontal axis: Estimated quantiles from predicted data

    Both axes are in units of their respective data sets. That is, the actual
    quantile level is not plotted. For a given point on the q-q plot, we know
    that the quantile level is the same for both points, but not what that 
    quantile level actually is.

    If the data sets have the same size, the q-q plot is essentially a plot of
    sorted data set 1 against sorted data set 2. If the data sets are not of
    equal size, the quantiles are usually picked to correspond to the sorted
    values from the smaller data set and then the quantiles for the larger data
    set are interpolated. Following NIST's Engineering Statistics Handbook.
    """
    #pre-process and sort observed and predicted data
    q_pred = _maskSeries(predicted).compressed()
    q_pred.sort()
    q_obse = _maskSeries(observed).compressed()
    q_obse.sort()

    #are predicted and observed same length?
    q_len = len(q_pred)
    o_len = len(q_obse)

    # calculate quantiles from presorted arrays
    def quantsFromSorted(inarr):
        # quantile given by rank divided by number of elements
        numElem = len(inarr)
        ranks = numpy.arange(1,numElem+1).astype(float)
        quants = ranks/numElem
        return quants

    #get target and plot
    fig, ax = set_target(target=target)
    if q_len>o_len:
        plot_obse = q_obse
        plot_pred = numpy.percentile(q_pred, 100*quantsFromSorted(q_obse))
    elif q_len<o_len:
        plot_obse = numpy.percentile(q_obse, 100*quantsFromSorted(q_pred))
        plot_pred = q_pred
    else:
        plot_pred = q_pred
        plot_obse = q_obse
    ax.scatter(plot_pred, plot_obse, **plot_kwargs)
    ax.set_ylabel('Observed')
    ax.set_xlabel('Predicted')
    ax.set_title('Q-Q plot')

    if xyline:
        #add y=x line
        ax.plot([0,1],[0,1], transform=ax.transAxes, linestyle='--', 
                             linewidth=1.0, color='black')

    out = dict()
    out['Figure'] = fig
    out['Axes'] = ax

    return out


def ROCcurve(predicted, observed, low=None, high=None, nthresh=100, 
             target=None, xyline=True):
    """
    observed is binary
    predicted is predicted probability

    Returns
    =======
    out_dict : dict
        A dictionary containing the Figure, Axes, POD, POFD, and Thresholds

    Example
    =======
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import StratifiedKFold
    
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> X, y = X[y != 2], y[y != 2]
    >>> n_samples, n_features = X.shape
    
    >>> random_state = np.random.RandomState(0)
    >>> X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
    
    >>> cv = StratifiedKFold(n_splits=6)
    >>> classifier = svm.SVC(kernel='linear', probability=True,
    >>>                      random_state=random_state)

    >>> fig, ax = set_target(None)
    >>> output = []
    
    >>> for train, test in cv.split(X, y):
    >>>     probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    >>>     output.append(ROCcurve(probas_[:, 1], y[test], target=ax))
    #compare to http://scikit-learn.org/stable/_images/sphx_glr_plot_roc_crossval_001.png
    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)

    out = dict()

    if low is None:
        low = pred.min()
    if high is None:
        high = pred.max()
    thresholds = numpy.linspace(low, high, num=nthresh)
    
    pods = numpy.zeros(len(thresholds)+2, dtype=float)
    pofds = numpy.zeros(len(thresholds)+2, dtype=float)
    pods[0] = 1
    pofds[0] = 1
    for idx, thr in enumerate(thresholds,1):
        bool_pred = pred >= thr
        ctable = Contingency2x2.fromBoolean(bool_pred, obse)
        pods[idx] = ctable.POD()
        pofds[idx] = ctable.POFD()
    pods[-1] = 0
    pofds[-1] = 0

    #get target and plot
    fig, ax = set_target(target=target)

    ax.plot(pofds, pods, drawstyle='steps-post')
    if xyline:
        #add y=x line
        ax.plot([0,1],[0,1], transform=ax.transAxes, linestyle='--', 
                             linewidth=1.0, color='black')

    ax.set_ylabel('Probability of Detection')
    ax.set_xlabel('Probability of False Detection')

    out['POD'] = pods
    out['POFD'] = pofds
    out['Thresholds'] = thresholds
    out['Figure'] = fig
    out['Axes'] = ax

    return out
