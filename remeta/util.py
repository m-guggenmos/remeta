import sys
import warnings

import numpy as np
from scipy.stats import rankdata


TAB = '    '
maxfloat = np.float128 if hasattr(np, 'float128') else np.longdouble
_slsqp_epsilon = np.sqrt(np.finfo(float).eps)  # scipy's default value for the SLSQP epsilon parameter


class ReprMixin:
    def __repr__(self):
        return f'{self.__class__.__name__}\n' + '\n'.join([f'\t{k}: {v}' for k, v in self.__dict__.items()])


def _check_param(x):
    if hasattr(x, '__len__'):
        if len(x) == 2:
            return x
        elif len(x) == 1:
            return [x[0], x[0]]
        else:
            print(f'Something went wrong, parameter array has length {len(x)}')
    else:
        return [x, x]


def _check_criteria(x):
    if hasattr(x[0], '__len__'):
        return x
    else:
        return [x, x]


def pearson2d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    mx, my = np.nanmean(x, axis=-1), np.nanmean(y, axis=-1)
    xm, ym = x - mx[..., None], y - my[..., None]
    r_num = np.nansum(xm * ym, axis=-1)
    r_den = np.sqrt(np.nansum(xm ** 2, axis=-1) * np.nansum(ym ** 2, axis=-1))
    r = r_num / r_den
    return r


def spearman2d(x, y, axis=0):
    x, y = np.asarray(x), np.asarray(y)
    xr, yr = rankdata(x, axis=axis), rankdata(y, axis=axis)
    mxr, myr = np.nanmean(xr, axis=-1), np.nanmean(yr, axis=-1)
    xmr, ymr = xr - mxr[..., None], yr - myr[..., None]
    r_num = np.nansum(xmr * ymr, axis=-1)
    r_den = np.sqrt(np.nansum(xmr ** 2, axis=-1) * np.nansum(ymr ** 2, axis=-1))
    r = r_num / r_den
    return r


def print_warnings(w):
    for el in set([w_.message.args[0] for w_ in w]):
        if 'delta_grad == 0.0' not in el:
            print('\tWarning: ' + el)


def raise_warning_in_catch_block(msg, category, w):
    warnings.warn(msg, category=category)
    if len(w):
        sys.stderr.write(warnings.formatwarning(
            w[-1].message, w[-1].category, w[-1].filename, w[-1].lineno, line=w[-1].line
        ))


def type2roc(correct, conf, nbins=5):
    # Calculate area under type 2 ROC
    #
    # correct - vector of 1 x ntrials, 0 for error, 1 for correct
    # conf - vector of continuous confidence ratings between 0 and 1
    # nbins - how many bins to use for discretization

    bs = 1 / nbins
    h2, fa2 = np.full(nbins, np.nan), np.full(nbins, np.nan)
    for c in range(nbins):
        if c:
            h2[nbins - c - 1] = np.sum((conf > c*bs) & (conf <= (c+1)*bs) & correct.astype(bool)) + 0.5
            fa2[nbins - c - 1] = np.sum((conf > c*bs) & (conf <= (c+1)*bs) & ~correct.astype(bool)) + 0.5
        else:
            h2[nbins - c - 1] = np.sum((conf >= c * bs) & (conf <= (c + 1) * bs) & correct.astype(bool)) + 0.5
            fa2[nbins - c - 1] = np.sum((conf >= c * bs) & (conf <= (c + 1) * bs) & ~correct.astype(bool)) + 0.5

    h2 /= np.sum(h2)
    fa2 /= np.sum(fa2)
    cum_h2 = np.hstack((0, np.cumsum(h2)))
    cum_fa2 = np.hstack((0, np.cumsum(fa2)))

    k = np.full(nbins, np.nan)
    for c in range(nbins):
        k[c] = (cum_h2[c+1] - cum_fa2[c])**2 - (cum_h2[c] - cum_fa2[c+1])**2

    auroc2 = 0.5 + 0.25*np.sum(k)

    return auroc2
