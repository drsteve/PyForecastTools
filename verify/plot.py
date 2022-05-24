"""
Sub-module to provide simple plotting for exploratory data analysis
"""
import numpy
import matplotlib.pyplot as plt

from .metrics import _maskSeries, bias
from .categorical import Contingency2x2

def setTarget(target, figsize=None, loc=111, polar=False):
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
    >>> from verify.plot import setTarget
    >>> fig = plt.figure()
    >>> fig, ax = setTarget(target=fig, loc=211)

    Notes
    =====
    Implementation from SpacePy's plot module.
    SpacePy is available at https://github.com/spacepy/spacepy
    under a Python Software Foudnation license.
    """
    # Is target a figure?  Make a new axes.
    if type(target) == plt.Figure:
        fig = target
        ax = fig.add_subplot(loc, polar=polar)
    # Is target an axes?  Make no new items.
    elif issubclass(type(target), plt.Axes):
        ax = target
        fig = ax.figure
    # Is target something else?  Make new everything.
    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(loc, polar=polar)
    return fig, ax


def qqPlot(predicted, observed, xyline=True, addTo=None, modelName='',
           legend=False, plot_kwargs=None):
    """Quantile-quantile plot for predictions and observations

    Parameters
    ==========
    predicted :  array-like
        predicted data (model output)
    observed : array-like
        observation vector reference value

    Other Parameters
    ================
    xyline : boolean
        Toggles the display of a line of y=x (perfect model). Default True.
    addTo : figure, axes, or None
        The object on which plotting will happen. If **None** (default)
        then a figure an axes will be created. If a matplotlib figure is
        supplied a set of axes will be made, and if matplotlib axes are
        given then the plot will be made on those axes.
    modelName :
        Name of model for legend.
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
    >>> import matplotlib.pyplot as plt
    >>> from verify.plot import qqPlot
    >>> np.random.seed(46)
    >>> model1 = np.random.randint(0,40,101).astype(float)
    >>> model2 = model1*(1 + np.arange(101)/70)
    >>> obs = model1 + np.random.randn(101)
    >>> obs[[2,3,8,9,15,16,30,31]] = obs[[31,30,16,15,9,8,3,2]]
    >>> obs *= 0.25 + (5-(np.arange(101)/30.))/4
    >>> observed = obs[:71] #QQ plots don't require even sample lengths
    >>> plot_settings = {'marker': 'X', 'c': np.arange(71), 'cmap':'cool'}
    >>> out1 = qqPlot(model1, observed, modelName='1',
    >>>               plot_kwargs=plot_settings)
    >>> plot_settings = {'marker': 'o'}
    >>> out2 = qqPlot(model2, observed, modelName='2', legend=True,
    >>>               plot_kwargs=plot_settings, addTo=out1['Axes'])
    >>> plt.show()

    Notes
    =====
    The q-q plot is formed by:
    - Vertical axis: Estimated quantiles from observed data
    - Horizontal axis: Estimated quantiles from predicted data

    Both axes are in units of their respective data sets. That is, the actual
    quantile level is not plotted. For a given point on the q-q plot, we know
    that the quantile level is the same for both points, but not what that
    quantile level actually is.

    For equal length samples, the q-q plot displays sorted(sample1) against
    sorted(sample2). If the samples are not of equal length, the quantiles
    for the smaller sample are calculated and data from the larger sample are
    interpolated to those quantiles. See, e.g., NIST's Engineering Statistics
    Handbook.
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
        ranks = numpy.arange(1, numElem+1).astype(float)
        quants = ranks/numElem
        return quants

    #get target and plot
    fig, ax = setTarget(target=addTo)
    if q_len > o_len:
        plot_obse = q_obse
        plot_pred = numpy.percentile(q_pred, 100*quantsFromSorted(q_obse))
    elif q_len < o_len:
        plot_obse = numpy.percentile(q_obse, 100*quantsFromSorted(q_pred))
        plot_pred = q_pred
    else:
        plot_pred = q_pred
        plot_obse = q_obse
    if plot_kwargs is None: plot_kwargs = dict()
    ax.scatter(plot_pred, plot_obse, label=modelName, **plot_kwargs)
    ax.set_ylabel('Observed')
    ax.set_xlabel('Predicted')
    ax.set_title('Q-Q plot')
    if legend:
        ax.legend(loc=0)

    if xyline:
        #add y=x line, force axes to be equal
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--',
                linewidth=1.0, color='black')
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        newmin = min(xlims[0], ylims[0])
        newmax = max(xlims[1], ylims[1])
        ax.set_xlim([newmin, newmax])
        ax.set_ylim([newmin, newmax])

    out = dict()
    out['Figure'] = fig
    out['Axes'] = ax

    return out


def rocCurve(predicted, observed, low=None, high=None, nthresh=100,
             addTo=None, xyline=True, modelName='', legend=False):
    """Receiver Operating Characteristic curve for assessing model skill

    Parameters
    ==========
    predicted :  array-like
        predicted data, continuous data (e.g. probability)
    observed : array-like
        observation vector of binary events (boolean or 0,1)

    Other Parameters
    ================
    low : float or None
        Set the lowest threshold to use.
    high : float or None
        Set the highest threshold to use
    xyline : boolean
        Toggles the display of a line of y=x (perfect model). Default True.
    addTo : figure, axes, or None
        The object on which plotting will happen. If **None** (default)
        then a figure an axes will be created. If a matplotlib figure is
        supplied a set of axes will be made, and if matplotlib axes are
        given then the plot will be made on those axes.

    Returns
    =======
    out_dict : dict
        A dictionary containing the Figure, Axes, POD, POFD, and Thresholds

    Example
    =======
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn import svm, datasets, linear_model

    >>> from verify.plot import rocCurve, setTarget

    >>> np.random.seed(0)
    >>> classifiers = {'Logistic regression': linear_model.LogisticRegression(),
    >>>                'SVC': svm.SVC(kernel='linear',
    >>>                               decision_function_shape='ovr',
    >>>                               probability=True)}

    >>> X, y = datasets.make_classification(n_samples=10000, n_features=21,
    >>>                                     n_informative=8, n_redundant=2,
    >>>                                     n_classes=2, n_repeated=1,
    >>>                                     n_clusters_per_class=3,
    >>>                                     flip_y=0.2)
    >>> Nsample_train = 100
    >>> X_train = X[:Nsample_train]
    >>> X_test = X[Nsample_train:]
    >>> y_train = y[:Nsample_train]
    >>> y_test = y[Nsample_train:]

    >>> fig, ax = setTarget(None)
    >>> output = []
    >>> for method, model in classifiers.items():
    >>>     model.fit(X_train, y_train)
    >>>     pred = model.predict_proba(X_test)[:,1]
    >>>     output.append(rocCurve(pred, y_test, modelName=method, addTo=ax))
    >>> output[-1]['Axes'].legend()
    >>> plt.show()
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
    for idx, thr in enumerate(thresholds, 1):
        bool_pred = pred >= thr
        ctable = Contingency2x2.fromBoolean(bool_pred, obse)
        pods[idx] = ctable.POD()
        pofds[idx] = ctable.POFD()
    pods[-1] = 0
    pofds[-1] = 0

    #get target and plot
    fig, ax = setTarget(target=addTo)

    ax.plot(pofds, pods, drawstyle='steps-post', label=modelName)
    if xyline:
        #add y=x line
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, linestyle='--',
                linewidth=1.0, color='black')

    ax.set_xlabel('Probability of False Detection')
    ax.set_ylabel('Probability of Detection')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if legend:
        ax.legend(loc=0)

    out['POD'] = pods
    out['POFD'] = pofds
    out['Thresholds'] = thresholds
    out['Figure'] = fig
    out['Axes'] = ax

    return out

def reliabilityDiagram(predicted, observed, norm=False, addTo=None,
                       modelName='', bins=None, xyline=True, legend=False,
                       plotkwargs={}, histkwargs={}):
    """Reliability diagram for a probabilistic forecast model

    Parameters
    ==========
    predicted :  array-like
        predicted data, continuous data (e.g. probability)
    observed : array-like
        observation vector of binary events (boolean or 0,1)

    Other Parameters
    ================
    norm : boolean
        Normalize input scores into [0,1] interval. Default False.
    addTo : figure, axes, or None
        The object on which plotting will happen. If **None** (default)
        then a figure an axes will be created. If a matplotlib figure is
        supplied a set of axes will be made, and if matplotlib axes are
        given then the plot will be made on those axes.
    modelName : string
        Name for model to be supplied to legend. Default empty string.
    bins : nonetype, int, sequence of scalars, or str
        Provides bin information as required by numpy.histogram. An integer
        sets the number of equally-sized bins used for the calibration function
        and refinement distribution. A sequence of scalars will be used to 
        define the bin edges, and a string must be a valid binning method per
        numpy.histogram. The default is to use numpy's 'auto' method, limited
        to a maximum of 100 bins.
    xyline : boolean
        Toggles the display of a line of y=x (perfect model). Default True.
    legend : boolean
        Toggles the display of a legend with the labels defined in 'modelName'
    plotkwargs : dict
        Dictionary of keyword arguments for the calibration function
    histkwargs : dict
        Dictionary of keyword arguments for the refinement distribution

    Returns
    =======
    out_dict : dict
        A dictionary containing the Figure, Axes, ...

    Example
    =======
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn import svm, datasets, linear_model

    >>> from verify.plot import reliabilityDiagram, setTarget

    >>> np.random.seed(0)
    >>> classifiers = {'Logistic regression': linear_model.LogisticRegression(),
    >>>                'SVC': svm.SVC(kernel='linear',
    >>>                               decision_function_shape='ovr',
    >>>                               probability=True)}

    >>> X, y = datasets.make_classification(n_samples=10000, n_features=21,
    >>>                                     n_informative=8, n_redundant=2,
    >>>                                     n_classes=2, n_repeated=1,
    >>>                                     n_clusters_per_class=3,
    >>>                                     flip_y=0.2)
    >>> train_samples = 100  # Samples used for training the models
    >>> X_train = X[:train_samples]
    >>> X_test = X[train_samples:]
    >>> y_train = y[:train_samples]
    >>> y_test = y[train_samples:]

    >>> fig, ax = setTarget(None)
    >>> output = []
    >>> for method, model in classifiers.items():
    >>>     model.fit(X_train, y_train)
    >>>     pred = model.predict_proba(X_test)[:,1]
    >>>     output.append(reliabilityDiagram(pred, y_test, norm=True,
    >>>                                      modelName=method, addTo=fig))
    >>> plt.show()

    Notes
    =====
    Reliability diagrams show whether the predictions from a probabilistic
    binary classifier are well calibrated.
    """
    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)

    out = dict()

    if norm:  #Normalize scores into range [0, 1]
        pred = (pred-pred.min())/(pred.max()-pred.min())
        rmin = 0
        rmax = 1
    else:
        rmin = pred.min()
        rmax = pred.max()

    if bins is None:
        bins='auto'
        limit = True
    else:
        limit = False # if user supplied bins, dispense with the limit...
    bin_edges = numpy.histogram_bin_edges(pred, bins=bins, range=(rmin, rmax))
    nbins = len(bin_edges)-1
    if (limit) and (nbins > 100): #too busy for plot
        bins = numpy.histogram_bin_edges(pred, bins=100, range=(rmin, rmax))
        nbins = len(bins)-1

    pred_binMean = numpy.zeros(nbins)
    obse_binProb = numpy.zeros(nbins)
    pred_binMean.fill(numpy.nan)
    obse_binProb.fill(numpy.nan)

    inds = numpy.digitize(pred, bin_edges)
    #digitize is left-inclusive, so put the observations
    #at p=1 into the top bin
    inds[inds == nbins+1] = nbins

    filledBins = list(set(inds))
    for idx in filledBins:
        # Store mean predicted prob and mean empirical probability
        pred_binMean[idx-1] = pred[inds == idx].mean()
        obse_binProb[idx-1] = obse[inds == idx].mean()

    if addTo is None:
        fig = plt.figure(0, figsize=(8, 8))
        ax_rel = plt.subplot2grid((3, 1), (0, 0), rowspan=2, fig=fig)
        ax_hist = plt.subplot2grid((3, 1), (2, 0), fig=fig)
    else:
        fig = addTo
        axes = fig.axes
        if len(axes) != 2:
            if len(axes) == 1 and len(axes[0].get_children()) == 10:
                #probably an empty figure, so let's nuke it
                ax_rel = plt.subplot2grid((3, 1), (0, 0), rowspan=2, fig=fig)
                ax_hist = plt.subplot2grid((3, 1), (2, 0), fig=fig)
            else:
                raise ValueError('reliabilityDiagram: supplied Figure for'+\
                                 'plotting appears to have been used for a'+\
                                 'different purpose.')
        else:
            ax_rel = axes[0]
            ax_hist = axes[1]

    handles, labels = ax_rel.get_legend_handles_labels()
    if 'y=x' not in labels or xyline is True:
        ax_rel.plot([0.0, 1.0], [0.0, 1.0], 'k--', label='y=x')
    valid = ~numpy.isnan(obse_binProb)
    if 'label' not in plotkwargs:
        plotkwargs['label'] = modelName
    ax_rel.plot(pred_binMean[valid], obse_binProb[valid], **plotkwargs)
    ax_rel.set_ylabel('Empirical probability')
    ax_rel.xaxis.set_major_formatter(plt.NullFormatter())
    if legend:
        ax_rel.legend(loc=0)

    histkwargs['bins'] = bins
    if 'histtype' not in histkwargs: histkwargs['histtype'] = 'step'
    if 'lw' not in histkwargs: histkwargs['lw'] = 2
    if 'density' not in histkwargs: histkwargs['density'] = True
    ax_hist.hist(pred, range=(rmin, rmax), **histkwargs)
    ax_hist.set_xlabel('Predicted Probability')
    ax_hist.set_ylabel('Density')

    ax_rel.set_xlim([rmin, rmax])
    ax_rel.set_ylim([rmin, rmax])
    ax_hist.set_xlim([rmin, rmax])

    out = dict()
    out['Figure'] = fig
    out['Axes'] = fig.axes

    return out


def taylorDiagram(predicted, observed, norm=False, addTo=None, modelName='',
                  isoSTD=True):
    """Taylor diagrams for comparing model performance

    Parameters
    ==========
    predicted :  array-like
        predicted data
    observed : array-like
        observation vector

    Other Parameters
    ================
    norm : boolean or float
        Selects whether the values should be normalized (default is False).
        If a value is given this will be used to normalize the inputs.
    xyline : boolean
        Toggles the display of a line of y=x (perfect model). Default True.
    addTo : axes, or None
        The object on which plotting will happen. If **None** (default)
        then a figure and axes will be created. If matplotlib axes are
        given then the plot will be made on those axes, assuming that the
        point is being added to a previously generated Taylor diagram.
    modelName : string
        Name of model to label the point on the Taylor diagram with.
    isoSTD : boolean
        Toggle for isocontours of standard deviation. Default is True, but
        turning them off can reduce visual clutter or prevent intereference
        with custom plot styles that alter background grid behavior.

    Returns
    =======
    out_dict : dict
        A dictionary containing the Figure, the Axes, and 'Norm' (the value
        used to normalize the inputs/outputs).

    Example
    =======
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from verify.plot import taylorDiagram
    >>> model1 = np.random.randint(0,40,101).astype(float)
    >>> model2 = model1*(1 + np.arange(101)/70)
    >>> obs = model1 + np.random.randn(101)
    >>> obs[[2,3,8,9,15,16,30,31]] = obs[[31,30,16,15,9,8,3,2]]
    >>> obs *= 0.25 + (5-(np.arange(101)/30.))/4
    >>> result1 = taylorDiagram(model1, obs, norm=True, modelName='A')
    >>> result2 = taylorDiagram(model2, obs, norm=result1['Norm'],
    >>>                         modelName='B', addTo=result1['Axes'])
    >>> plt.show()

    Notes
    =====
    Based on ''Summarizing multiple aspects of model performance in a single
    diagram' by K.E. Taylor (Radio Science, 2001; doi: 10.1029/2000JD900719)
    and 'Taylor Diagram Primer' by Taylor (document at
    https://pcmdi.llnl.gov/staff/taylor/CV/Taylor_diagram_primer.pdf)
    With some implementation aspects inspired by the public domain code of
    github user ycopin at https://gist.github.com/ycopin/3342888
    """
    #fancy plotting imports
    from mpl_toolkits.axisartist import floating_axes, angle_helper, grid_finder
    from matplotlib.projections import PolarAxes

    pred = _maskSeries(predicted)
    obse = _maskSeries(observed)
    pstd = pred.std(ddof=1) #unbiased sample std.dev. model
    ostd = obse.std(ddof=1) #unbiased sample std.dev. observed
    pcorr = numpy.corrcoef(obse, pred)[0, 1] # Pearson's r
    pbias = bias(pred, obse) #mean error

    # Normalize on request
    if norm:
        setNormed = True
        normfac = ostd
        ostd = ostd / normfac
        pstd = pstd / normfac
    else:
        setNormed = False
        normfac = 1

    tr = PolarAxes.PolarTransform()
    # Correlation labels
    corrtickvals = numpy.asarray(list(range(10))+[9.5, 9.9])/10.
    corrtickvals_polar = numpy.arccos(corrtickvals)
    # Round labels to nearest N digits
    corrticklabs = []
    for cval in corrtickvals:
        lab = '{0:0.2f}'.format(cval) if str(cval)[2] != '0' else \
              '{0:0.1f}'.format(cval)
        corrticklabs.append('{0:0.2f}'.format(cval))
    valsAndLabs = list(zip(corrtickvals_polar, corrticklabs))
    corrgrid = grid_finder.FixedLocator(corrtickvals_polar)
    corrticks = grid_finder.DictFormatter(dict(valsAndLabs))

    # Std. Dev. tick values and labels
    smin, smax = 0, numpy.ceil(0.8335*ostd)*2
    stdtickvals = numpy.linspace(smin, smax, 9)
    # Round labels to nearest N digits
    stdticklabs = []
    for stdval in stdtickvals:
        stdticklabs.append('{0:0.2f}'.format(stdval))

    valsAndLabs = list(zip(stdtickvals, stdticklabs))
    stdgrid = grid_finder.FixedLocator(stdtickvals)
    stdticks = grid_finder.DictFormatter(dict(valsAndLabs))

    gh_curvegrid = floating_axes.GridHelperCurveLinear(tr,
                                                       extremes=(0, numpy.pi/2, smin, smax),
                                                       grid_locator1=corrgrid,
                                                       grid_locator2=stdgrid,
                                                       tick_formatter1=corrticks,
                                                       tick_formatter2=stdticks
                                                      )

    artists = []
    #if addTo isn't None then assume we've been given a previously returned
    #axes object
    if addTo is None:
        fig = plt.figure()
        ax = floating_axes.FloatingSubplot(fig, 111,
                                           grid_helper=gh_curvegrid)
        fig.add_subplot(ax)

        #Adjust axes following matplotlib gallery example
        #Correlation is angular coord
        ax.axis['top'].set_axis_direction('bottom')
        ax.axis['top'].toggle(ticklabels=True, label=True)
        ax.axis['top'].major_ticklabels.set_axis_direction('top')
        ax.axis['top'].major_ticklabels.set_color('royalblue')
        ax.axis['top'].label.set_axis_direction('top')
        ax.axis['top'].label.set_color('royalblue')
        ax.axis['top'].label.set_text("Pearson's r")
        #X-axis
        ax.axis['left'].set_axis_direction('bottom')
        ax.axis['left'].toggle(ticklabels=True)
        #Y-axis
        ax.axis['right'].set_axis_direction('top')
        ax.axis['right'].toggle(ticklabels=True, label=True)
        ax.axis['right'].major_ticklabels.set_axis_direction('left')
        if setNormed:
            xylabel = 'Normalized standard deviation'
        else:
            xylabel = 'Standard deviation'
        ax.axis['right'].label.set_text(xylabel)
        ax.axis['left'].label.set_text(xylabel)
        ax.axis['bottom'].set_visible(False)
        _ax = ax                   # Display axes

        #Figure set up, done we work with the transformed axes
        ax = ax.get_aux_axes(tr)   # Axes referenced in polar coords
        #Add reference point, ref stddev contour and other stddev contours
        artists.append(ax.scatter(0, ostd, marker='o', linewidths=2,
                                  edgecolors='black', facecolors='None',
                                  label='Observation', zorder=99))
        azim = numpy.linspace(0, numpy.pi/2)
        rad = numpy.zeros_like(azim)
        rad.fill(ostd)
        ax.plot(azim, rad, 'k--', alpha=0.75)
        if isoSTD:
            for sd in stdtickvals[::2]:
                rad.fill(sd)
                ax.plot(azim, rad, linestyle=':', color='dimgrey', alpha=0.75,
                        zorder=3)

        #Add radial markers at correlation ticks
        for i in corrtickvals[1:]:
            ax.plot([numpy.arccos(i), numpy.arccos(i)], [0, smax],
                    c='royalblue', alpha=0.5, zorder=40)

        #Add contours of centered RMS error
        rs, ts = numpy.meshgrid(numpy.linspace(smin, smax),
                                numpy.linspace(0, numpy.pi/2))
        rms = numpy.sqrt(ostd**2 + rs**2 - 2*ostd*rs*numpy.cos(ts))
        contours = ax.contour(ts, rs, rms, 4, alpha=0.75, zorder=30,
                              colors='dimgray')
        plt.clabel(contours, inline=True, fontsize='smaller', fmt='%1.2f')
    else:
        #TODO: add some testing/error handling here
        ax = addTo
        fig = ax.figure

    #add present model
    stdextent = smax-smin
    twopercent = stdextent/50.0
    artists.append(ax.scatter(numpy.arccos(pcorr), pstd, marker='o',
                              label=modelName, zorder=99))
    dum = ax.text(numpy.arccos(pcorr), pstd+twopercent, modelName,
                  fontsize='larger')

    out = dict()
    out['Figure'] = fig
    out['Axes'] = ax
    out['Norm'] = normfac
    out['Artists'] = artists

    return out
