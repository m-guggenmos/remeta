import warnings

import numpy as np
from scipy.special import erf, erfinv
from scipy.special import factorial as fac
from scipy.stats import norm

from .util import maxfloat
from .util import _check_param, _check_criteria


def warp(stimuli, warping, warping_function='power'):
    """
    Nonlinear transducing.

    Parameters
    ----------
    stimuli : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity. Must be normalized to [-1; 1].
    warping : float
        Warping factor. Negative (positive) values lead to logarithmic (exponential) transformations of the stimuli.
    warping_function : str
        Either 'power' or 'exponential'. Pass 'identity' to leave stimuli unchanged.

    Returns
    ----------
    stimuli_warped : array-liked
        Warped stimulus array.
    """
    warping_ = _check_param(warping)
    if warping_function == 'identity':
        return stimuli
    elif warping_function == 'power':
        stimuli_warped = np.sign(stimuli)
        stimuli_warped[stimuli < 0] *= np.abs(stimuli[stimuli < 0]) ** warping_[0]
        stimuli_warped[stimuli >= 0] *= np.abs(stimuli[stimuli >= 0]) ** warping_[1]
    elif warping_function == 'exponential':
        stimuli_warped = np.sign(stimuli)
        if np.abs(warping_[0]) > 1e-10:
            stimuli_warped[stimuli < 0] *= np.abs(np.exp(warping_[0] * np.abs(stimuli[stimuli < 0])) - 1)
        else:
            stimuli_warped[stimuli < 0] = stimuli[stimuli < 0]
        if np.abs(warping_[1]) > 1e-10:
            stimuli_warped[stimuli >= 0] *= np.abs(np.exp(warping_[1] * np.abs(stimuli[stimuli >= 0])) - 1)
        else:
            stimuli_warped[stimuli >= 0] = stimuli[stimuli >= 0]
    else:
        raise ValueError(f"'{warping_function}' is not a valid warping function")

    stimuli_warped /= np.max(np.abs(stimuli_warped))

    return stimuli_warped


def logistic(x, noise_sens):
    """
    Logistic function

    Parameters
    ----------
    x : array-like
        This is typically a stimulus array or a transformed stimulus array (e.g. sensory decision values).
    noise_sens : float
        Sensory noise parameter.

    Returns
    ----------
    posterior : array-like
        Posterior probability under a logistic model.
    """
    posterior = 1 / (1 + np.exp(-((np.pi / np.sqrt(3)) * x / np.maximum(noise_sens, 1.6e-4)).squeeze(),
                                dtype=maxfloat))
    return posterior.astype(np.float64)


def logistic_inv(posterior, noise_sens):
    """
    Inverse logistic function

    Parameters
    ----------
    posterior : array-like
        Posterior probability.
    noise_sens : float
        Sensory noise parameter.

    Returns
    ----------
    x : array-like
        See docstring of the logistic method.
    """
    x = -(np.sqrt(3) * noise_sens / np.pi) * np.log(1 / posterior - 1)
    return x


def noise_sens_transform(stimuli, noise_sens=None, noise_transform_sens=None, thresh_sens=None,
                         function_noise_transform_sens='multiplicative', **kwargs):  # noqa
    """
    Sensory noise transformation.

    Parameters
    ----------
    stimuli : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity. Must be normalized to [-1; 1].
    noise_sens : float or array-like
        Sensory noise parameter.
    noise_transform_sens : float or array-like
        Signal-dependent sensory noise parameter.
    thresh_sens: float or array-like
        Sensory threshold.
    function_noise_transform_sens : str
        Define the signal dependency of sensory noise. One of 'multiplicative', 'power', 'exponential', 'logarithm'.
    kwargs : dict
        Conveniance parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    noise_sens_transformed : array-like
        Sensory noise array of shape stimuli.shape. Each stimulus is assigned a sensory noise parameter.
    """
    noise_sens_ = _check_param(noise_sens)
    noise_transform_sens_ = _check_param(noise_transform_sens)
    thresh_sens_ = (0, 0) if thresh_sens is None else _check_param(thresh_sens)
    noise_sens_transformed = np.ones(stimuli.shape)
    neg, pos = stimuli < 0, stimuli >= 0
    if noise_transform_sens is None:
        noise_sens_transformed[neg] *= noise_sens_[0]
        noise_sens_transformed[pos] *= noise_sens_[1]
    elif function_noise_transform_sens == 'multiplicative':
        noise_sens_transformed[neg] = np.sqrt(noise_sens_[0]**2 +
            ((np.abs(stimuli[neg]) - thresh_sens_[0]) * noise_transform_sens_[0])**2)  # noqa
        noise_sens_transformed[pos] = np.sqrt(noise_sens_[1]**2 +
            ((np.abs(stimuli[pos]) - thresh_sens_[1]) * noise_transform_sens_[1])**2)  # noqa
    elif function_noise_transform_sens == 'power':
        noise_sens_transformed[neg] = np.sqrt(noise_sens_[0]**2 +
            (np.abs(stimuli[neg]) - thresh_sens_[0]) ** (2 * noise_transform_sens_[0]))  # noqa
        noise_sens_transformed[pos] = np.sqrt(noise_sens_[1]**2 +
            (np.abs(stimuli[pos]) - thresh_sens_[1]) ** (2 * noise_transform_sens_[1]))  # noqa
    elif function_noise_transform_sens == 'exponential':
        noise_sens_transformed[neg] = np.sqrt(noise_sens_[0]**2 +
            (np.exp(noise_transform_sens_[0] * (np.abs(stimuli[neg]) - thresh_sens_[0])) - 1)**2)  # noqa
        noise_sens_transformed[pos] = np.sqrt(noise_sens_[1]**2 +
            (np.exp(noise_transform_sens_[1] * (np.abs(stimuli[pos]) - thresh_sens_[1])) - 1)**2)  # noqa
    elif function_noise_transform_sens == 'logarithm':
        noise_sens_transformed[neg] = np.sqrt(noise_sens_[0]**2 +
            np.log(noise_transform_sens_[0] * (np.abs(stimuli[neg]) - thresh_sens_[0]) + 1)**2)  # noqa
        noise_sens_transformed[pos] = np.sqrt(noise_sens_[1]**2 +
            np.log(noise_transform_sens_[1] * (np.abs(stimuli[pos]) - thresh_sens_[1]) + 1)**2)  # noqa
    else:
        raise ValueError(f'{function_noise_transform_sens} is not a valid transform function for noise_sens')

    return noise_sens_transformed


def _noise_sens_transform_pc(stimuli, dv_sens, evidence_bias_mult_postnoise_meta=None, noise_sens=None,
                             noise_transform_sens=None, function_noise_transform_sens='multiplicative', **kwargs):  # noqa
    """
    Signal-dependent sensory noise transformation.
    Helper function for the signal-dependent sensory noise transformation under a probability-correct (pc) link
    function. In this case, sensory noise might be subject to a multiplicative metacognitive bias defined by
    evidence_bias_mult_postnoise_meta.
    """
    noise_sens_ = _check_param(noise_sens)
    evidence_bias_mult_postnoise_meta_ = _check_param(evidence_bias_mult_postnoise_meta)
    noise_sens_transformed = np.full(dv_sens.shape, np.nan)
    if (len(dv_sens.shape) > len(stimuli.shape)) or (stimuli.shape[-1] == 1):
        stimuli = np.tile(stimuli, dv_sens.shape[-1])

    noise_sens_neg = [noise_sens_[0] / evidence_bias_mult_postnoise_meta_[0],
                      noise_sens_[1] / evidence_bias_mult_postnoise_meta_[0]]
    noise_sens_transformed[(dv_sens < 0)] = \
        noise_sens_transform(stimuli[dv_sens < 0], noise_sens_neg, noise_transform_sens=noise_transform_sens,
                             function_noise_transform_sens=function_noise_transform_sens)
    noise_sens_pos = [noise_sens_[0] / evidence_bias_mult_postnoise_meta_[1],
                      noise_sens_[1] / evidence_bias_mult_postnoise_meta_[1]]
    noise_sens_transformed[(dv_sens >= 0)] = \
        noise_sens_transform(stimuli[dv_sens >= 0], noise_sens_pos, noise_transform_sens=noise_transform_sens,
                             function_noise_transform_sens=function_noise_transform_sens)
    return noise_sens_transformed


def noise_meta_transform(confidence_or_dv_meta, dv_sens=None, noise_meta=None, noise_transform_meta=None,
                         function_noise_transform_meta='multiplicative',
                         ignore_warnings=False, **kwargs):  # noqa
    """
    Metacognitive noise transformation.

    Parameters
    ----------
    confidence_or_dv_meta : array-like
        Model-based absolute sensory decision values ("dv_meta") in case of a noisy-readout model or model-predicted
        confidence in case of a noisy-report model.
    dv_sens : array-like
        Sensory decision values
    noise_meta : float or array-like
        Metacognitive noise parameter.
    noise_transform_meta : float or array-like
        Signal-dependent metacognitive noise parameter.
    function_noise_transform_meta : array-like
        Signal-dependent noise type. One of 'multiplicative', 'power', 'exponential', 'logarithm'.
    ignore_warnings : bool
        If True, ignore warnings within the method (currently not used).
    kwargs : dict
        Conveniance parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    noise_meta_transformed : array-like
        Metacognitive noise array of shape confidence_or_dv_meta.shape.
    """
    noise_meta_ = _check_param(noise_meta)
    noise_meta_transformed = np.full(confidence_or_dv_meta.shape, np.nan)
    neg, pos = dv_sens < 0, dv_sens >= 0
    if noise_transform_meta is None:
        noise_meta_transformed[neg] = noise_meta_[0]
        noise_meta_transformed[pos] = noise_meta_[1]
    else:
        noise_transform_meta_ = _check_param(noise_transform_meta)
        if function_noise_transform_meta == 'multiplicative':
            noise_meta_transformed[neg] = np.sqrt(noise_meta_[0]**2 +
                                                  (confidence_or_dv_meta[neg] * noise_transform_meta_[0])**2)
            noise_meta_transformed[pos] = np.sqrt(noise_meta_[1]**2 +
                                                  (confidence_or_dv_meta[pos] * noise_transform_meta_[1])**2)
        elif function_noise_transform_meta == 'test':
            noise_meta_transformed[neg] = noise_meta_[0] + noise_transform_meta_[0] * np.abs(confidence_or_dv_meta[neg])
            noise_meta_transformed[pos] = noise_meta_[1] + noise_transform_meta_[1] * np.abs(confidence_or_dv_meta[pos])
        elif function_noise_transform_meta == 'power':
            noise_meta_transformed[neg] = np.sqrt(noise_meta_[0]**2 +
                                                  confidence_or_dv_meta[neg] ** (2 * noise_transform_meta_[0]))
            noise_meta_transformed[pos] = np.sqrt(noise_meta_[1]**2 +
                                                  confidence_or_dv_meta[pos] ** (2 * noise_transform_meta_[1]))
        elif function_noise_transform_meta == 'exponential':
            noise_meta_transformed[neg] = np.sqrt(noise_meta_[0]**2 +
                (np.exp(noise_transform_meta_[0] * confidence_or_dv_meta[neg]) - 1)**2)  # noqa
            noise_meta_transformed[pos] = np.sqrt(noise_meta_[1]**2 +
                (np.exp(noise_transform_meta_[1] * confidence_or_dv_meta[pos]) - 1)**2)  # noqa
        elif function_noise_transform_meta == 'logarithm':
            noise_meta_transformed[neg] = np.sqrt(noise_meta_[0]**2 +
                                                  np.log(noise_transform_meta_[0] * confidence_or_dv_meta[neg] + 1)**2)
            noise_meta_transformed[pos] = np.sqrt(noise_meta_[1]**2 +
                                                  np.log(noise_transform_meta_[1] * confidence_or_dv_meta[pos] + 1)**2)
        else:
            raise ValueError(f'{function_noise_transform_meta} is not a valid transform function for '
                             f'noise_intercept_meta.')

    # if np.any(noise_meta_transformed < 0.001):
    #     noise_meta_transformed = np.maximum(noise_meta_transformed, 0.001)
    # if not ignore_warning:
    #     warnings.warn('Minimal noise_meta below 0.001. This should not happen. '
    #                   'Setting minimal noise_meta = 0.001')
    return noise_meta_transformed

def link_function(dv_meta, link_fun, evidence_bias_mult_postnoise_meta=1, confidence_bias_mult_meta=1,
                  confidence_bias_add_meta=0, confidence_bias_exp_meta=1, criteria_meta=None, levels_meta=None,
                  noise_sens=None, noise_transform_sens=None, function_noise_transform_sens='linear',
                  dv_sens=None, stimuli=None, constraint_mode=False, nchannels=10,
                  **kwargs):  # noqa
    """
    Link function.

    Parameters
    ----------
    dv_meta : array-like
        Model-based absolute sensory decision values ("dv_meta").
    link_fun : str
        Metacognitive link function. In case of criterion-based link functions {x} refers to the number of criteria.
        Possible values: 'probability_correct', 'tanh', 'normcdf', 'erf', 'alg', 'guder', 'linear', 'identity',
                         'detection_model_linear', 'detection_model_mean', 'detection_model_mode',
                         'detection_model_full', 'detection_model_ideal'
                         '{x}_criteria', '{x}_criteria_linear', '{x}_criteria_linear_tanh'
    evidence_bias_mult_postnoise_meta : float or array-like
        Multiplicative metacognitive bias parameter loading on evidence, but after application of readout noise.
    confidence_bias_mult_meta : float or array-like
        Multiplicative metacognitive bias parameter loading on confidence.
    confidence_bias_add_meta : float or array-like
        Additive metacognitive bias parameter loading on confidence.
    confidence_bias_exp_meta : float or array-like
        Exponential metacognitive bias parameter loading on confidence.
    criteria_meta : array-like
        Confidence criteria in case of a criterion-based link function.
    levels_meta : array-like
        Confidence levels in case of a criterion-based link function.
    noise_sens : float or array-like
        Sensory noise parameter.
    noise_transform_sens : float or array-like
        Signal-dependent sensory noise parameter.
    function_noise_transform_sens : float or array-like
        Signal-dependent sensory noise type. One of 'linear', 'power', 'exponential', 'logarithm'.
    dv_sens : array-like
        Sensory decision values.
    stimuli : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity.
    constraint_mode : bool
        If True, method runs during scipy optimize constraint testing and certain warnings are ignored.
    nchannels : int
        Number of channels (relevant only for detection-type models).
    kwargs : dict
        Conveniance parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    confidence_pred : array-like
        Model-predicted confidence.
    """
    dv_meta = np.atleast_1d(dv_meta)
    if dv_sens is None:
        if hasattr(evidence_bias_mult_postnoise_meta, '__len__') or hasattr(confidence_bias_add_meta, '__len__') or \
                hasattr(confidence_bias_mult_meta, '__len__'):
            raise ValueError('Parameters evidence_bias_mult_postnoise_meta, confidence_bias_add_meta or '
                             'confidence_bias_mult_meta appear to be sign-dependent (they have been passed as '
                             'array-like), but dv_sens has not been provided.')
        else:
            dv_sens = dv_meta
    evidence_bias_mult_postnoise_meta_ = _check_param(evidence_bias_mult_postnoise_meta)
    confidence_bias_mult_meta_ = _check_param(confidence_bias_mult_meta)
    confidence_bias_add_meta_ = _check_param(confidence_bias_add_meta)
    confidence_bias_exp_meta_ = _check_param(confidence_bias_exp_meta)
    if criteria_meta is not None:
        criteria_meta_ = _check_criteria(criteria_meta)
    if levels_meta is not None:
        levels_meta_ = _check_criteria(levels_meta)
    else:
        levels_meta_ = None
    if link_fun.startswith('detection_model') and link_fun.endswith('_scaled'):
        dv_meta = np.minimum(nchannels, evidence_bias_mult_postnoise_meta * dv_meta)

    if link_fun in ['tanh', 'normcdf', 'erf', 'alg', 'guder', 'linear', 'identity']:
        lf = dict(
            tanh=lambda x, xsign, slope:
            (xsign < 0) * np.tanh(slope[0] * np.abs(x)) +
            (xsign >= 0) * np.tanh(slope[1] * np.abs(x)),
            normcdf=lambda x, xsign, slope:
            (xsign < 0) * 2 * (norm(scale=slope[0]).cdf(np.abs(x)) - 0.5) +
            (xsign >= 0) * 2 * (norm(scale=slope[1]).cdf(np.abs(x)) - 0.5),
            erf=lambda x, xsign, slope:
            (xsign < 0) * erf(slope[0] * np.abs(x)) +
            (xsign >= 0) * erf(slope[1] * np.abs(x)),
            alg=lambda x, xsign, slope:
            (xsign < 0) * slope[0] * np.abs(x) / np.sqrt(1 + (slope[0] * np.abs(x)) ** 2) +
            (xsign >= 0) * slope[1] * np.abs(x) / np.sqrt(1 + (slope[1] * np.abs(x)) ** 2),
            guder=lambda x, xsign, slope:
            (xsign < 0) * (4 / np.pi) * np.arctan(np.tanh(slope[0] * np.abs(x))) +
            (xsign >= 0) * (4 / np.pi) * np.arctan(np.tanh(slope[1] * np.abs(x))),
            linear=lambda x, xsign, slope:
            (xsign < 0) * np.minimum(1, slope[0] * np.abs(x)) +
            (xsign >= 0) * np.minimum(1, slope[1] * np.abs(x)),
            identity=lambda x, xsign, slope: np.abs(x)
        )[link_fun]
        confidence_pred = lf(dv_meta, dv_sens, evidence_bias_mult_postnoise_meta_)
    elif link_fun == 'probability_correct':
        if stimuli is None:
            if noise_transform_sens is not None or hasattr(noise_sens, '__len__'):
                raise ValueError('Sensory noise is sign- or intensity-dependent, but stimuli have not been '
                                 'provided.')
            else:
                stimuli = dv_sens
        noise_sens = _noise_sens_transform_pc(
            stimuli, dv_sens, evidence_bias_mult_postnoise_meta=evidence_bias_mult_postnoise_meta_,
            noise_sens=noise_sens, noise_transform_sens=noise_transform_sens,
            noise_sens_function=function_noise_transform_sens
        )
        confidence_pred = np.tanh(np.pi * dv_meta / (2 * np.sqrt(3) * noise_sens))
    elif 'crit' in link_fun:
        if 'linear_tanh' in link_fun:
            def lf(x, xsign, crit, levels=None):
                slope_tanh_neg = crit[0][-1]  # by definition, the last criterion is the slope of the tanh component
                slope_tanh_pos = crit[1][-1]  # by definition, the last criterion is the slope of the tanh component
                critt_neg = np.array(crit[0][:-1])
                critt_pos = np.array(crit[1][:-1])
                if levels is None:
                    level_neg = np.linspace(0, 1, len(critt_neg) + 2)  # confidence at the criteria
                    level_pos = np.linspace(0, 1, len(critt_pos) + 2)  # confidence at the criteria
                else:
                    level_neg, level_pos = np.hstack((0, levels[0])), np.hstack((0, levels[1]))
                crit0inf_neg = np.hstack((0, critt_neg, np.inf))  # add the edge cases of 0 and infinity
                crit0inf_pos = np.hstack((0, critt_pos, np.inf))  # add the edge cases of 0 and infinity

                # the tanh function covers the space between the last (real) criterion and infinity (x-axis) and
                # between the second-last and last criterion (y-axis):
                def func_tanh_neg(x_):
                    return (level_neg[-1] - level_neg[-2]) * \
                           np.tanh(slope_tanh_neg * (x_ - crit0inf_neg[-2])) + level_neg[-2]

                def func_tanh_pos(x_):
                    return (level_neg[-1] - level_pos[-2]) * \
                           np.tanh(slope_tanh_pos * (x_ - crit0inf_pos[-2])) + level_pos[-2]

                with warnings.catch_warnings():
                    # when constraints are tested, equal-value criterions may be tested, leading to a 'divide-by-zero'
                    # warning; we ignore it, as affected criterions will be invalidated by a subsequent constraint
                    warnings.simplefilter('ignore' if constraint_mode else 'default')  # noqa
                    # compute slopes up until the last (real) criterion:
                    slopes_neg = [(level_neg[i] - level_neg[i - 1]) / (crit0inf_neg[i] - crit0inf_neg[i - 1])
                                  for i in range(1, len(critt_neg) + 1)]
                    slopes_pos = [(level_pos[i] - level_pos[i - 1]) / (crit0inf_pos[i] - crit0inf_pos[i - 1])
                                  for i in range(1, len(critt_pos) + 1)]
                    # compute intercepts up until the last (real) criterion:
                    intercepts_neg = [
                        level_neg[i - 1] - crit0inf_neg[i - 1] * (level_neg[i] - level_neg[i - 1]) / (
                                crit0inf_neg[i] - crit0inf_neg[i - 1]) for i in range(1, len(critt_neg) + 1)]
                    intercepts_pos = [
                        level_pos[i - 1] - crit0inf_pos[i - 1] * (level_pos[i] - level_pos[i - 1]) / (
                                crit0inf_pos[i] - crit0inf_pos[i - 1]) for i in range(1, len(critt_pos) + 1)]
                # conditions of the piecewise function:
                cond_neg = [(x >= crit0inf_neg[i]) & (x < crit0inf_neg[i + 1]) for i in range(len(crit0inf_neg) - 1)]
                cond_pos = [(x >= crit0inf_pos[i]) & (x < crit0inf_pos[i + 1]) for i in range(len(crit0inf_pos) - 1)]
                # piecewise functions - linear until the last (real) criterion, then tanh:
                func_neg = [(lambda x_, i=i: slopes_neg[i] * x_ + intercepts_neg[i])
                            for i in range(len(critt_neg))] + [func_tanh_neg]
                func_pos = [(lambda x_, i=i: slopes_pos[i] * x_ + intercepts_pos[i])
                            for i in range(len(critt_pos))] + [func_tanh_pos]
                return (xsign < 0) * np.piecewise(x, cond_neg, func_neg) + \
                       (xsign >= 0) * np.piecewise(x, cond_pos, func_pos)
        elif 'linear' in link_fun:
            def lf(x, xsign, crit, levels=None):
                critt_neg = np.array(crit[0])
                critt_pos = np.array(crit[1])
                if levels is None:
                    level_neg = np.linspace(0, 1, len(critt_neg) + 1)  # confidence at the criteria
                    level_pos = np.linspace(0, 1, len(critt_pos) + 1)  # confidence at the criteria
                else:
                    level_neg, level_pos = np.hstack((0, levels[0])), np.hstack((0, levels[1]))
                crit0inf_neg = np.hstack((0, critt_neg, np.inf))  # add the edge cases of 0 and infinity
                crit0inf_pos = np.hstack((0, critt_pos, np.inf))  # add the edge cases of 0 and infinity
                with warnings.catch_warnings():
                    # when constraints are tested equal-value criterions may be tested, leading to a 'divide-by-zero'
                    # warning; we ignore it, as affected criterions will be invalidated by a subsequent constraint
                    warnings.simplefilter('ignore' if constraint_mode else 'default')  # noqa
                    slopes_neg = [(level_neg[i] - level_neg[i - 1]) /
                                  (crit0inf_neg[i] - crit0inf_neg[i - 1]) for i in range(1, len(critt_neg) + 1)]
                    slopes_pos = [(level_pos[i] - level_pos[i - 1]) /
                                  (crit0inf_pos[i] - crit0inf_pos[i - 1]) for i in range(1, len(critt_pos) + 1)]
                    intercepts_neg = \
                        [level_neg[i - 1] - crit0inf_neg[i - 1] * ((level_neg[i] - level_neg[i - 1]) /
                                                                   (crit0inf_neg[i] - crit0inf_neg[i - 1])) for i in
                         range(1, len(critt_neg) + 1)]
                    intercepts_pos = \
                        [level_pos[i - 1] - crit0inf_pos[i - 1] * (level_pos[i] - level_pos[i - 1]) /
                         (crit0inf_pos[i] - crit0inf_pos[i - 1]) for i in range(1, len(critt_pos) + 1)]
                cond_neg = [(x >= crit0inf_neg[i]) & (x < crit0inf_neg[i + 1]) for i in range(len(crit0inf_neg) - 1)]
                cond_pos = [(x >= crit0inf_pos[i]) & (x < crit0inf_pos[i + 1]) for i in range(len(crit0inf_pos) - 1)]
                func_neg = [(lambda x_, i=i: slopes_neg[i] * x_ + intercepts_neg[i]) for i in range(len(critt_neg))] + \
                           [lambda x_, i=None: level_neg[-1]]
                func_pos = [(lambda x_, i=i: slopes_pos[i] * x_ + intercepts_pos[i]) for i in range(len(critt_pos))] + \
                           [lambda x_, i=None: level_pos[-1]]
                return (xsign < 0) * np.piecewise(x, cond_neg, func_neg) + \
                       (xsign >= 0) * np.piecewise(x, cond_pos, func_pos)
        else:
            def lf(x, xsign, crit, levels=None):
                conf = np.full(x.shape, np.nan)
                levels_neg = np.linspace(0, 1, len(crit[0]) + 1) if levels is None else np.hstack(
                    (0, levels[0]))  # noqa
                levels_pos = np.linspace(0, 1, len(crit[1]) + 1) if levels is None else np.hstack(
                    (0, levels[1]))  # noqa
                conf[xsign < 0] = levels_neg[np.searchsorted(np.array(crit[0]), x[xsign < 0])]
                conf[xsign >= 0] = levels_pos[np.searchsorted(np.array(crit[1]), x[xsign >= 0])]
                return conf
        confidence_pred = lf(dv_meta, dv_sens, criteria_meta_, levels_meta_)  # noqa
    elif link_fun.startswith('detection_model_linear'):
        # heuristic model: simply report the proportion of active channels
        confidence_pred = dv_meta / nchannels
    elif link_fun.startswith('detection_model_mean'):
        # given your current observation, compute the probability of making a correct choice when faced with the
        # identical experiment again *using the mean of the posterior distribution of p_active*
        confidence_pred = 1 - (1 - (dv_meta + 1) / (nchannels + 2)) ** nchannels
    elif link_fun.startswith('detection_model_mode'):
        # given your current observation, compute the probability of making a correct choice when faced with the
        # identical experiment again *using the mode of the posterior distribution of p_active*
        confidence_pred = 1 - (1 - dv_meta / nchannels) ** nchannels
        # noise_transform_sens = noise_sens_transform(stimuli, noise_sens, noise_transform_sens,
        #                                             function_noise_transform_sens)
        # confidence_pred = 1 - (1 - np.tanh(dv_meta / noise_transform_sens))**nchannels
    elif link_fun.startswith('detection_model_full'):
        # given your current observation, compute the probability of making a correct choice when faced with the
        # identical experiment again *by evaluating the extended posterior distribution of p_active*
        confidence_pred = 1 - ((fac(2 * nchannels - dv_meta) * fac(nchannels + 1)) / (
                fac(2 * nchannels + 1) * fac(nchannels - dv_meta)))
    elif link_fun.startswith('detection_model_ideal'):
        confidence_pred = np.full(dv_meta.shape, np.nan)
        confidence_pred[dv_meta == 0] = 0
        confidence_pred[dv_meta > 0] = 1
    else:
        raise ValueError(f'{link_fun} is not a valid link function for the metacognitive type noisy-report')

    confidence_pred[dv_sens < 0] = (confidence_pred[dv_sens < 0] ** confidence_bias_exp_meta_[0]) * \
        confidence_bias_mult_meta_[0] + confidence_bias_add_meta_[0]
    confidence_pred[dv_sens >= 0] = (confidence_pred[dv_sens >= 0] ** confidence_bias_exp_meta_[1]) * \
        confidence_bias_mult_meta_[1] + confidence_bias_add_meta_[1]
    confidence_pred = np.maximum(0, np.minimum(1, confidence_pred))

    return confidence_pred


def link_function_inv(confidence, link_fun, evidence_bias_mult_postnoise_meta=1, confidence_bias_mult_meta=1,
                      confidence_bias_add_meta=0, confidence_bias_exp_meta=1, criteria_meta=None,
                      levels_meta=None, noise_sens=None, noise_transform_sens=None,
                      function_noise_transform_sens='linear', dv_sens=None, stimuli=None,
                      **kwargs):  ## noqa
    """
    Inverse link function.

    Parameters
    ----------
    confidence : array-like
        Confidence ratings (from behavioral or simulated data).
    link_fun : str
        Metacognitive link function. In case of criterion-based link functions {x} refers to the number of criteria.
        Possible values: 'probability_correct', 'tanh', 'normcdf', 'erf', 'alg', 'guder', 'linear', 'identity',
                         'detection_model_linear', 'detection_model_mean', 'detection_model_mode',
                         'detection_model_full', 'detection_model_ideal'
                         '{x}_criteria', '{x}_criteria_linear', '{x}_criteria_linear_tanh'
    evidence_bias_mult_postnoise_meta : float or array-like
        Multiplicative metacognitive bias parameter loading on evidence, but after application of readout noise.
    confidence_bias_mult_meta : float or array-like
        Multiplicative metacognitive bias parameter loading on confidence.
    confidence_bias_add_meta : float or array-like
        Additive metacognitive bias parameter loading on confidence.
    confidence_bias_exp_meta : float or array-like
        Exponential metacognitive bias parameter loading on confidence.
    criteria_meta : array-like
        Confidence criteria in case of a criterion-based link function.
    levels_meta : array-like
        Confidence levels in case of a criterion-based link function.
    noise_sens : float or array-like
        Sensory noise parameter.
    noise_transform_sens : float or array-like
        Signal-dependent sensory noise parameter.
    function_noise_transform_sens : float or array-like
        Signal-dependent sensory noise type. One of 'linear', 'power', 'exponential', 'logarithm'.
    dv_sens : array-like
        Sensory decision values.
    stimuli : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity.
    kwargs : dict
        Conveniance parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    dv_meta : array-like
        Absolute sensory decision values ('dv_meta').
    """
    if dv_sens is None:
        if hasattr(evidence_bias_mult_postnoise_meta, '__len__') or hasattr(confidence_bias_add_meta, '__len__') or \
                hasattr(confidence_bias_mult_meta, '__len__'):
            raise ValueError('Parameters evidence_bias_mult_postnoise_meta, confidence_bias_mult_meta or '
                             'confidence_bias_add_meta appear to be sign-dependent (they have been passed as '
                             'array-like), but dv_sens has not been provided.')
        else:
            dv_sens = confidence
    else:
        confidence = np.tile(confidence, dv_sens.shape[-1])
    evidence_bias_mult_postnoise_meta_ = _check_param(evidence_bias_mult_postnoise_meta)
    confidence_bias_add_meta_ = _check_param(confidence_bias_add_meta)
    confidence_bias_mult_meta_ = _check_param(confidence_bias_mult_meta)
    confidence_bias_exp_meta_ = _check_param(confidence_bias_exp_meta)

    confidence[dv_sens < 0] = ((confidence[dv_sens < 0] - confidence_bias_add_meta_[0]) / confidence_bias_mult_meta_[0]) ** (1 / confidence_bias_exp_meta_[0])
    confidence[dv_sens >= 0] = ((confidence[dv_sens >= 0] - confidence_bias_add_meta_[1]) / confidence_bias_mult_meta_[1]) ** (1 / confidence_bias_exp_meta_[1])
    confidence = np.minimum(1, confidence)

    if link_fun in ['tanh', 'erf', 'alg', 'guder', 'linear', 'logistic3']:
        if link_fun != 'linear':
            confidence = np.minimum(1 - 1e-8, confidence)  # confidence=1 not invertible
        invlf = dict(
            tanh=lambda x, xsign, slope: (xsign < 0) * (np.arctanh(x) / slope[0]) +
                                         (xsign >= 0) * (np.arctanh(x) / slope[1]),
            erf=lambda x, xsign, slope: (xsign < 0) * (erfinv(x) / slope[0]) +
                                        (xsign >= 0) * (erfinv(x) / slope[1]),
            alg=lambda x, xsign, slope: (xsign < 0) * ((x / np.sqrt(1 - x ** 2)) / slope[0]) +
                                        (xsign >= 0) * ((x / np.sqrt(1 - x ** 2)) / slope[1]),
            guder=lambda x, xsign, slope: (xsign < 0) * (np.arctanh(np.tan(np.pi * x / 4)) / slope[0]) +
                                          (xsign >= 0) * (np.arctanh(np.tan(np.pi * x / 4)) / slope[1]),
            linear=lambda x, xsign, slope: (xsign < 0) * (x / slope[0]) +
                                           (xsign >= 0) * (x / slope[1]),
        )[link_fun]
        dv_meta = invlf(confidence, dv_sens, evidence_bias_mult_postnoise_meta_)
    else:
        if link_fun == 'probability_correct':
            confidence = np.minimum(1 - 1e-8, confidence)
            if stimuli is None:
                if noise_transform_sens is not None or hasattr(noise_sens, '__len__'):
                    raise ValueError('Sensory noise is sign- or intensity-dependent, but stimuli have not been '
                                     'provided.')
            else:
                noise_sens = _noise_sens_transform_pc(
                    stimuli, dv_sens, evidence_bias_mult_postnoise_meta=evidence_bias_mult_postnoise_meta_,
                    noise_sens=noise_sens, noise_transform_sens=noise_transform_sens,
                    function_noise_transform_sens=function_noise_transform_sens,
                )
            dv_meta = (2 * np.sqrt(3) * noise_sens / np.pi) * np.arctanh(confidence)
        elif 'criteria_linear_tanh' in link_fun:
            criteria_meta_ = _check_criteria(criteria_meta)
            confidence = np.minimum(1 - 1e-8, confidence)  # confidence=1 not invertible
            slope_tanh_neg = criteria_meta_[0][-1]  # the last criterion is the slope of tanh
            slope_tanh_pos = criteria_meta_[1][-1]  # the last criterion is the slope of tanh
            critt_neg = np.array(criteria_meta_[0][:-1])
            critt_pos = np.array(criteria_meta_[1][:-1])
            if levels_meta is None:
                level_neg = np.linspace(0, 1, len(critt_neg) + 2)  # confidence at the criteria
                level_pos = np.linspace(0, 1, len(critt_pos) + 2)  # confidence at the criteria
            else:
                levels_meta_ = _check_criteria(levels_meta)
                level_neg, level_pos = np.hstack((0, levels_meta_[0])), np.hstack((0, levels_meta_[1]))
            crit0inf_neg = np.hstack((0, critt_neg, np.inf))  # add the edge cases of 0 and infinity
            crit0inf_pos = np.hstack((0, critt_pos, np.inf))  # add the edge cases of 0 and infinity
            y = np.full(dv_sens.shape, np.nan)
            for i in range(1, len(critt_neg) + 2):
                m_neg = (crit0inf_neg[i] - crit0inf_neg[i - 1]) / (level_neg[i] - level_neg[i - 1])
                m_pos = (crit0inf_pos[i] - crit0inf_pos[i - 1]) / (level_pos[i] - level_pos[i - 1])
                cond_neg = (dv_sens < 0) & (confidence >= level_neg[i - 1]) & (confidence < level_neg[i])
                cond_pos = (dv_sens >= 0) & (confidence >= level_pos[i - 1]) & (confidence < level_pos[i])
                y[cond_neg] = m_neg * (confidence[cond_neg] - level_neg[i - 1]) + crit0inf_neg[i - 1]
                y[cond_pos] = m_pos * (confidence[cond_pos] - level_pos[i - 1]) + crit0inf_pos[i - 1]
            cond_neg = (dv_sens < 0) & (confidence >= level_neg[-2])
            cond_pos = (dv_sens >= 0) & (confidence >= level_pos[-2])
            y[cond_neg] = (np.arctanh(
                np.minimum(1 - 1e-16, (confidence[cond_neg] - level_neg[-2]) / (level_neg[-1] - level_neg[-2]))) /
                           slope_tanh_neg) + crit0inf_neg[-2]
            y[cond_pos] = (np.arctanh(
                np.minimum(1 - 1e-16, (confidence[cond_pos] - level_pos[-2]) / (level_pos[-1] - level_pos[-2]))) /
                           slope_tanh_pos) + crit0inf_pos[-2]
            dv_meta = y
        elif 'criteria_linear' in link_fun:
            raise ValueError("This model is not invertible.")
        else:
            raise ValueError(f'{link_fun} is not a valid link function for the metacognitive type noisy-readout')

    return dv_meta
