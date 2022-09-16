import os
import pathlib
import pickle
import timeit
import warnings
from dataclasses import make_dataclass

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import logistic as logistic_dist, binom

from .configuration import Configuration
from .dist import get_dist
from .fit import fmincon
from .gendata import simu_data
from .modelspec import Model, Data
from .plot import plot_link_function, plot_confidence_dist
from .transform import warp, noise_meta_transform, noise_sens_transform, logistic, link_function, link_function_inv
from .util import _check_param, TAB
from .util import maxfloat

np.set_printoptions(suppress=True)


class ReMeta:

    def __init__(self, cfg=None, **kwargs):
        """
        Main class of the ReMeta toolbox

        Parameters
        ----------
        cfg : util.Configuration
            Configuration object. If None is passed, the default configuration is used (but see kwargs).
        kwargs : dict
            The kwargs dictionary is parsed for keywords that match keywords of util.Configuration; in case of a match,
            the configuration is set.
        """

        if cfg is None:
            # Set configuration attributes that match keyword arguments
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in Configuration.__dict__}
            self.cfg = Configuration(**cfg_kwargs)
        else:
            self.cfg = cfg
        self.cfg.setup()

        if self.cfg.meta_noise_dist.startswith('truncated_') and self.cfg.meta_noise_dist.endswith('_lookup'):
            try:
                self.lookup_table = np.load(f"lookup_{self.cfg.meta_noise_dist}_{self.cfg.meta_noise_type}.npz")
            except FileNotFoundError:
                raise FileNotFoundError('Lookup table not found. Lookup tables are not deployed via pip. You can '
                                        'download them from Github and put in a directory named "lookup"')
        else:
            self.lookup_table = None

        self.model = Model(cfg=self.cfg)
        self.data = None

        # self._punish_message = False

        self.sens_is_fitted = False
        self.meta_is_fitted = False

        # Define the function that is minimized for metacognitive parameter fitting
        self.fun_meta = dict(noisy_report=self._negll_meta_noisyreport,
                             noisy_readout=self._negll_meta_noisyreadout)[self.cfg.meta_noise_type]
        self.fun_meta_helper = dict(noisy_report=self._helper_negll_meta_noisyreport,
                                    noisy_readout=self._helper_negll_meta_noisyreadout)[self.cfg.meta_noise_type]

    def fit(self, stimuli, choices, confidence, precomputed_parameters=None, guess_meta=None, verbose=True,
            ignore_warnings=False):
        """
        Fit sensory and metacognitive parameters

        Parameters
        ----------
        stimuli : array-like of shape (n_samples)
            Array of signed stimulus intensity values, where the sign codes the stimulus category (cat 1: -, cat2: +)
            and the absolut value codes the intensity. Must be normalized to [-1; 1], or set
            `normalize_stimuli_by_max=True`.
        choices : array-like of shape (n_samples)
            Array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive
            stimulus category.
        confidence : array-like of shape (n_samples)
            Confidence ratings; must be normalized to the range [0;1].
        precomputed_parameters : dict
            Provide pre-computed parameters. A dictionary with all parameters defined by the model must be passed. This
            can sometimes be useful to obtain information from the model without having to fit the model.
            [ToDO: which information?]
        guess_meta : array-like of shape (n_params_meta)
            For testing: provide an initial guess for the optimization of the metacognitive level
        verbose : bool
            If True, information of the model fitting procedure is printed.
        ignore_warnings : bool
            If True, warnings during model fitting are supressed.
        """

        # Instantiate util.Data object and perform preprocessing of the data
        self.data = Data(self.cfg, stimuli, choices, confidence)
        self.data.preproc()

        if verbose:
            print('\n+++ Sensory level +++')
        # with warnings.catch_warnings(record=True) as w:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize',
                                    message='delta_grad == 0.0. Check if the approximated function is linear. If the '
                                            'function is linear better results can be obtained by defining the Hessian '
                                            'as zero instead of using quasi-Newton approximations.')
            if ignore_warnings:
                warnings.filterwarnings('ignore')
            if isinstance(precomputed_parameters, dict):
                if not np.all([p in precomputed_parameters for p in self.cfg.paramset_sens.names]):
                    raise ValueError('Set of precomputed sensory parameters is incomplete.')
                self.model.fit.fit_sens = OptimizeResult(
                    x=[precomputed_parameters[p] for p in self.cfg.paramset_sens.names],
                    fun=self._negll_sens([precomputed_parameters[p] for p in self.cfg.paramset_sens.names])
                )
            else:
                if self.cfg.paramset_sens.nparams > 0:
                    self.cfg.paramset_sens.constraints = self.cfg.constraints_sens_callable(self)
                    if verbose:
                        negll_initial_guess = self._negll_sens(self.cfg.paramset_sens.guess)
                        print(f'Initial guess (neg. LL: {negll_initial_guess:.2f})')
                        for i, p in enumerate(self.cfg.paramset_sens.names):
                            print(f'{TAB}[guess] {p}: {self.cfg.paramset_sens.guess[i]:.4g}')
                        print('Performing local optimization')
                    t0 = timeit.default_timer()
                    self.model.fit.fit_sens = minimize(
                        self._negll_sens, self.cfg.paramset_sens.guess, bounds=self.cfg.paramset_sens.bounds,
                        constraints=self.cfg.paramset_sens.constraints, method='trust-constr'
                    )
                    if self.cfg.enable_thresh_sens:
                        fit_powell = minimize(
                            self._negll_sens, self.cfg.paramset_sens.guess, bounds=self.cfg.paramset_sens.bounds,
                            constraints=self.cfg.paramset_sens.constraints, method='Powell'
                        )
                        if fit_powell.fun < self.model.fit.fit_sens.fun:
                            self.model.fit.fit_sens = fit_powell
                    self.model.fit.fit_sens.execution_time = timeit.default_timer() - t0

                else:
                    self.model.fit.fit_sens = OptimizeResult(x=None)
            if isinstance(self.cfg.true_params, dict):
                if not np.all([p in self.cfg.true_params for p in self.cfg.paramset_sens.base_names]):
                    raise ValueError('Set of provided true sensory parameters is incomplete.')
                psens_true = sum([[self.cfg.true_params[p]] if n == 1 else self.cfg.true_params[p] for n, p in
                                  zip(self.cfg.paramset_sens.base_len, self.cfg.paramset_sens.base_names)], [])
                self.model.fit.fit_sens.negll_true = self._negll_sens(psens_true)

            # call once again with final=True to save the model fit
            negll, params_sens, choiceprob, posterior, stimuli_final = \
                self._negll_sens(self.model.fit.fit_sens.x, final=True)
            if 'thresh_sens' in params_sens and params_sens['thresh_sens'] < self.data.stimuli_min:
                warnings.warn('Fitted threshold is below the minimal stimulus intensity; consider disabling '
                              'the sensory threshold by setting enable_thresh_sens to 0', category=UserWarning)
            self.model.store_sens(negll, params_sens, choiceprob, posterior, stimuli_final,
                                  stimulus_norm_coefficent=self.data.stimuli_unnorm_max)
            self.model.report_fit_sens(verbose)
            self.sens_is_fitted = True

        # if not ignore_warnings and verbose:
        #     print_warnings(w)

        if not self.cfg.skip_meta:

            # compute decision values
            self._compute_dv_sens()

            if verbose:
                print('\n+++ Metacognitive level +++')

            args_meta = [ignore_warnings, None]
            if precomputed_parameters is not None:
                if not np.all([p in precomputed_parameters for p in self.cfg.paramset_meta.names]):
                    raise ValueError('Set of precomputed metacognitive parameters is incomplete.')
                self.model.params_meta = {p: precomputed_parameters[p] for p in self.cfg.paramset_meta.names}
                self.model.fit.fit_meta = OptimizeResult(
                    x=[precomputed_parameters[p] for p in self.cfg.paramset_meta.names],
                    fun=self.fun_meta([precomputed_parameters[p] for p in self.cfg.paramset_meta.names])
                )
                fitinfo_meta = self.fun_meta(list(self.model.params_meta.values()), *args_meta, final=True)  # noqa
                self.model.store_meta(*fitinfo_meta)
            else:
                with warnings.catch_warnings(record=True) as w:  # noqa
                    warnings.filterwarnings('ignore', module='scipy.optimize')
                    # prepare constraints for metacognitive parameters (which may depend on decision values)
                    if self.cfg.paramset_meta.nparams > 0:
                        self.cfg.paramset_meta.constraints = self.cfg.constraints_meta_callable(self)
                        self.model.fit.fit_meta = fmincon(
                            self.fun_meta, self.cfg.paramset_meta, args_meta, gradient_free=self.cfg.gradient_free,
                            gridsearch=self.cfg.gridsearch, grid_multiproc=self.cfg.grid_multiproc,
                            global_minimization=self.cfg.global_minimization,
                            fine_gridsearch=self.cfg.fine_gridsearch,
                            gradient_method=self.cfg.gradient_method, slsqp_epsilon=self.cfg.slsqp_epsilon,
                            init_nelder_mead=self.cfg.init_nelder_mead,
                            guess=guess_meta,
                            verbose=verbose
                        )
                    else:
                        self.model.fit.fit_meta = OptimizeResult(x=None)

                # call once again with final=True to save the model fit
                fitinfo_meta = self.fun_meta(self.model.fit.fit_meta.x, *args_meta, final=True)
                self.model.store_meta(*fitinfo_meta)

            if self.cfg.true_params is not None:
                params_true_meta = sum([[self.cfg.true_params[p]] if n == 1 else self.cfg.true_params[p] for n, p in
                                        zip(self.cfg.paramset_meta.base_len, self.cfg.paramset_meta.base_names)],
                                       [])
                self.model.fit.fit_meta.negll_true = self.fun_meta(params_true_meta)

            self.model.report_fit_meta(verbose)

            self.model.params = {**self.model.params_sens, **self.model.params_meta}

            self.meta_is_fitted = True

            # if not ignore_warnings:
            #     print_warnings(w)

    def summary(self, extended=False, generative=True, generative_nsamples=1000):
        """
        Provides information about the model fit.

        Parameters
        ----------
        extended : bool
            If True, store various model variables in the summary object.
        generative : bool
            If True, compare model predictions of confidence with empirical confidence by repeatedly sampling from
            the generative model.
        generative_nsamples : int
            Number of samples used for the generative model (higher = more accurate).

        Returns
        ----------
        summary : dataclass
            Information about model fit.
        """

        if not self.cfg.skip_meta and generative:
            gen = simu_data(generative_nsamples, self.data.nsamples, self.model.params,
                            cfg=self.cfg, stimuli_ext=self.data.stimuli, verbose=False)
        else:
            gen = None
        summary_model = self.model.summary(extended=extended, fun_meta=self.fun_meta,
                                           confidence_gen=None if gen is None else gen.confidence,  # noqa
                                           confidence_emp=self.data.confidence if generative else None)
        desc = dict(data=self.data.summary(extended), model=summary_model, cfg=self.cfg)

        summary_ = make_dataclass('Summary', desc.keys())

        def repr_(self_):
            txt = f'***{self_.__class__.__name__}***\n'
            for k, v in self_.__dict__.items():
                if k == 'cfg':
                    txt += f"\n{k}: {type(desc['cfg'])} <not displayed>"
                else:
                    txt += f"\n{k}: {v}"
            return txt

        summary_.__repr__ = repr_
        summary_.__module__ = '__main__'
        summary = summary_(**desc)
        return summary

    def _negll_sens(self, params, final=False, return_noise=False):
        """
        Minimization function for the sensory level

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the sensory level.
        final : bool
            If True, store latent variables and parameters.
        return_noise : bool
            If True, only return the sensory noise array (auxiliarly option to define constraints on the noise array).

        Returns:
        --------
        negll: float
            Negative (summed) log likelihood.
        """

        bl = self.cfg.paramset_sens.base_len
        params_sens = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                       for i, (p, n) in enumerate(zip(self.cfg.paramset_sens.base_names, bl))}
        thresh_sens = _check_param(params_sens['thresh_sens'] if self.cfg.enable_thresh_sens else 0)
        bias_sens = _check_param(params_sens['bias_sens'] if self.cfg.enable_bias_sens else 0)

        if self.cfg.enable_warping_sens:
            stimuli_final = warp(self.data.stimuli, params_sens['warping_sens'], self.cfg.function_warping_sens)
        else:
            stimuli_final = self.data.stimuli

        cond_neg, cond_pos = stimuli_final < 0, stimuli_final >= 0
        dv_sens = np.full(stimuli_final.shape, np.nan)
        # dv_sens[cond_neg] = (np.abs(stimuli_final[cond_neg]) > thresh_sens[0]) * \
        #                   (stimuli_final[cond_neg] - np.sign(stimuli_final[cond_neg]) * thresh_sens[0]) - bias_sens[0]
        # dv_sens[cond_pos] = (np.abs(stimuli_final[cond_pos]) > thresh_sens[1]) * \
        #                   (stimuli_final[cond_pos] - np.sign(stimuli_final[cond_pos]) * thresh_sens[1]) - bias_sens[1]
        dv_sens[cond_neg] = (np.abs(stimuli_final[cond_neg]) > thresh_sens[0]) * stimuli_final[cond_neg] + bias_sens[0]
        dv_sens[cond_pos] = (np.abs(stimuli_final[cond_pos]) > thresh_sens[1]) * stimuli_final[cond_pos] + bias_sens[1]

        noise_sens = self._noise_sens_transform(stimuli_final, params_sens)
        if return_noise:
            return noise_sens

        if self.cfg.detection_model:
            p_active = np.tanh(np.abs(dv_sens) / noise_sens)  # probability that a channel is active
            dv_congruent = np.sign(dv_sens) == np.sign(stimuli_final)
            p_correct = (dv_congruent * (1 - 0.5 * (1 - p_active) ** self.cfg.detection_model_nchannels) +
                         ~dv_congruent * 0.5 * (1 - p_active) ** self.cfg.detection_model_nchannels)
            posterior = (self.data.stimulus_ids == 1) * p_correct + (self.data.stimulus_ids == 0) * (1 - p_correct)
        else:
            posterior = logistic(dv_sens, noise_sens)
        choiceprob = (self.data.choices == 1) * posterior + (self.data.choices == 0) * (1 - posterior)
        negll = np.sum(-np.log(np.maximum(choiceprob, self.cfg.min_likelihood_sens)))

        return (negll, params_sens, choiceprob, posterior, stimuli_final) if final else negll

    def _negll_meta_noisyreadout(self, params, ignore_warnings=False, mock_binsize=None, final=False,
                                 return_noise=False, return_criteria=False, return_levels=False,
                                 constraint_mode=False):
        """
        Minimization function for the noisy-report model

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the metacognitive level.
        ignore_warnings : bool
            If True, suppress warnings during minimization.
        mock_binsize : float
            Binsize around empirical confidence ratings to evaluate the likelihood. If not None, also returns the
            likelihood variable.
        final : bool
            If True, return latent variables and parameters.
        return_noise : bool
            If True, only return the metacognitive noise array (auxiliarly option to define constraints on the noise
            array).
        return_criteria : bool
            If True, only return confidence criteria (auxiliarly option to define constraints on confidence criteria).
        return_levels : bool
            If True, only return confidence levels (auxiliarly option to define constraints on confidence levels).
        constraint_mode : bool
            If True, method runs during scipy optimize constraint testing.

        Returns:
        --------
        By default, the method returns the negative log likelihood. However, depending on the arguments various
        combinations of variables are returned (see Parameters).
        negll: float
            Negative (summed) log likelihood.
        """

        params_meta, noise_meta, dist, dv_meta_considered, likelihood, likelihood_pdf, criteria_meta, levels_meta, \
            punishment_factor = \
            self._helper_negll_meta_noisyreadout(params, ignore_warnings, mock_binsize, final, return_noise,
                                                 return_criteria, return_levels, constraint_mode)

        # compute log likelihood
        likelihood_weighted_cum = np.nansum(self.model.dv_sens_pmf * likelihood, axis=1)
        if self.cfg.experimental_likelihood:
            # use an upper bound for the negative log likelihood based on a uniform 'guessing' model
            negll = min(self.model.max_negll, -np.sum(np.log(np.maximum(likelihood_weighted_cum, 1e-200))))
        else:
            negll = -np.sum(np.log(np.maximum(likelihood_weighted_cum, self.cfg.min_likelihood_meta)))
        negll *= punishment_factor

        if final:
            self.model.confidence = self._link_function(dv_meta_considered, params_meta,
                                                        criteria_meta=criteria_meta, levels_meta=levels_meta)
            return negll, params_meta, noise_meta, likelihood, dv_meta_considered, likelihood_weighted_cum, \
                   likelihood_pdf  # noqa
        elif mock_binsize is not None:
            return negll, likelihood
        else:
            return negll

    def _helper_negll_meta_noisyreadout(self, params, ignore_warnings=False, mock_binsize=None, final=False,
                                        return_noise=False, return_criteria=False, return_levels=False,
                                        constraint_mode=False):  # noqa
        """
        Minimization function for the noisy-readout model

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the metacognitive level.
        ignore_warnings : bool
            If True, suppress warnings during minimization.
        mock_binsize : float
            Binsize around empirical confidence ratings to evaluate the likelihood. If not None, also returns the
            likelihood variable.
        final : bool
            If True, return latent variables and parameters.
        return_noise : bool
            If True, only return the metacognitive noise array (auxiliarly option to define constraints on the noise
            array).
        return_criteria : bool
            If True, only return confidence criteria (auxiliarly option to define constraints on confidence criteria).
        return_levels : bool
            If True, only return confidence levels (auxiliarly option to define constraints on confidence levels).
        constraint_mode : bool
            If True, method runs during scipy optimize constraint testing.

        Returns:
        --------
        By default, the method returns the negative log likelihood. However, depending on the arguments various
        combinations of variables are returned (see Parameters).
        negll: float
            Negative (summed) log likelihood.
        """

        bl = self.cfg.paramset_meta.base_len
        params_meta = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                       for i, (p, n) in enumerate(zip(self.cfg.paramset_meta.base_names, bl))}

        punishment_factor = 1
        if '_criteria' in self.cfg.meta_link_function:
            criteria_meta, levels_meta = self._prepare_criteria(params_meta)
            if return_criteria:
                return criteria_meta
            if return_levels:
                return levels_meta
            if not final:
                bds = [(params[i] >= b[0]) & (params[i] <= b[1]) for i, b in enumerate(self.cfg.paramset_meta.bounds)]
                cns = [con['fun'](params) >= 0 for con in self.cfg.paramset_meta.constraints]
                if not np.all(cns) or not np.all(bds):
                    # if not self._punish_message:
                    #     print('punish!')
                    #     self._punish_message = True
                    punishment_factor = 10
        else:
            criteria_meta, levels_meta = None, None

        dv_meta_considered = self._compute_dv_meta(params_meta)

        noise_meta = self._noise_meta_transform(dv_meta_considered, params_meta, ignore_warnings=ignore_warnings)
        if return_noise:
            return noise_meta

        binsize = self.cfg.binsize_meta if mock_binsize is None else mock_binsize
        if self.cfg.experimental_wrap_binsize_meta:
            wrap_neg = binsize - np.abs(np.minimum(1, self.data.confidence_2d + binsize) - self.data.confidence_2d)  # noqa
            wrap_pos = binsize - np.abs(np.maximum(0, self.data.confidence_2d - binsize) - self.data.confidence_2d)  # noqa
            binsize_neg, binsize_pos = binsize + wrap_neg, binsize + wrap_pos
        else:
            binsize_neg, binsize_pos = binsize, binsize
        dv_meta_from_conf_lb = self._link_function_inv(
            np.maximum(0, self.data.confidence_2d - binsize_neg), self.cfg.meta_link_function, params_meta,
            criteria_meta=criteria_meta, levels_meta=levels_meta)
        dv_meta_from_conf_ub = self._link_function_inv(
            np.minimum(1, self.data.confidence_2d + binsize_pos), self.cfg.meta_link_function, params_meta,
            criteria_meta=criteria_meta, levels_meta=levels_meta)
        dv_meta_from_conf = self._link_function_inv(
            self.data.confidence_2d, self.cfg.meta_link_function, params_meta,
            criteria_meta=criteria_meta, levels_meta=levels_meta)

        dist = get_dist(self.cfg.meta_noise_dist, mode=dv_meta_considered, scale=noise_meta,
                        meta_noise_type='noisy_readout', lookup_table=self.lookup_table)
        if self.cfg.meta_noise_dist.startswith('censored_'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                window = dist.cdf(dv_meta_from_conf_ub) - dist.cdf(dv_meta_from_conf_lb)
                likelihood = (dv_meta_from_conf > 1e-8) * window + \
                             (dv_meta_from_conf <= 1e-8) * dist.cdf(dv_meta_from_conf + binsize_pos)
            if final:
                try:
                    cdf2 = dist.cdf(dv_meta_from_conf.astype(maxfloat) + binsize_pos).astype(np.float64)
                except TypeError:
                    # may currently fail for the normal distribution because np.float128 is not supported
                    cdf2 = dist.cdf(dv_meta_from_conf + binsize_pos)
                likelihood_pdf = (dv_meta_from_conf > 1e-8) * \
                    dist.pdf(dv_meta_from_conf.astype(maxfloat)).astype(np.float64) + (dv_meta_from_conf <= 1e-8) * cdf2
            else:
                likelihood_pdf = None
        else:
            likelihood = dist.cdf(dv_meta_from_conf_ub) - dist.cdf(dv_meta_from_conf_lb)
            if final:
                likelihood_pdf = dist.pdf(dv_meta_from_conf)
            else:
                likelihood_pdf = None

        if not self.cfg.detection_model and not self.cfg.experimental_include_incongruent_dv:
            likelihood[self.model.dv_sens_considered_invalid] = np.nan

        return params_meta, noise_meta, dist, dv_meta_considered, likelihood, \
               likelihood_pdf, criteria_meta, levels_meta, punishment_factor  # noqa

    def _negll_meta_noisyreport(self, params, ignore_warnings=False, mock_binsize=None, final=False,
                                return_noise=False, return_criteria=False, return_levels=False,
                                constraint_mode=False):
        """
        Minimization function for the noisy-report model

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the metacognitive level.
        ignore_warnings : bool
            If True, suppress warnings during minimization.
        mock_binsize : float
            Binsize around empirical confidence ratings to evaluate the likelihood. If not None, also returns the
            likelihood variable.
        final : bool
            If True, return latent variables and parameters.
        return_noise : bool
            If True, only return the metacognitive noise array (auxiliarly option to define constraints on the noise
            array).
        return_criteria : bool
            If True, only return confidence criteria (auxiliarly option to define constraints on confidence criteria).
        return_levels : bool
            If True, only return confidence levels (auxiliarly option to define constraints on confidence levels).
        constraint_mode : bool
            If True, method runs during scipy optimize constraint testing.

        Returns:
        --------
        By default, the method returns the negative log likelihood. However, depending on the arguments various
        combinations of variables are returned (see Parameters).
        negll: float
            Negative (summed) log likelihood.
        """

        params_meta, noise_meta, model_confidence, dist, dv_meta_considered, likelihood, likelihood_pdf, \
            punishment_factor = \
            self._helper_negll_meta_noisyreport(params, ignore_warnings, mock_binsize, final, return_noise,
                                                return_criteria, return_levels, constraint_mode)

        # compute weighted cumulative negative log likelihood
        likelihood_weighted_cum = np.nansum(likelihood * self.model.dv_sens_pmf, axis=1)
        if self.cfg.experimental_likelihood:
            # use an upper bound for the negative log likelihood based on a uniform 'guessing' model
            negll = min(self.model.max_negll, -np.sum(np.log(np.maximum(likelihood_weighted_cum, 1e-200))))
        else:
            negll = -np.sum(np.log(np.maximum(likelihood_weighted_cum, self.cfg.min_likelihood_meta)))
        negll *= punishment_factor

        if final:
            return negll, params_meta, noise_meta, likelihood, dv_meta_considered, likelihood_weighted_cum, \
                   likelihood_pdf  # noqa
        elif mock_binsize is not None:
            return negll, likelihood
        else:
            return negll

    def _helper_negll_meta_noisyreport(self, params, ignore_warnings=False, mock_binsize=None, final=False,
                                       return_noise=False, return_criteria=False, return_levels=False,
                                       constraint_mode=False):

        bl = self.cfg.paramset_meta.base_len
        params_meta = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                       for i, (p, n) in enumerate(zip(self.cfg.paramset_meta.base_names, bl))}

        punishment_factor = 1

        if '_criteria' in self.cfg.meta_link_function:
            criteria_meta, levels_meta = self._prepare_criteria(params_meta)
            if return_criteria:
                return criteria_meta
            if return_levels:
                return levels_meta
            if not final:
                bds = [(params[i] >= b[0]) & (params[i] <= b[1]) for i, b in enumerate(self.cfg.paramset_meta.bounds)]
                cns = [con['fun'](params) >= 0 for con in self.cfg.paramset_meta.constraints]
                if not np.all(cns) or not np.all(bds):
                    # if not self._punish_message:
                    #     print('punish!')
                    #     self._punish_message = True
                    punishment_factor = 10
        else:
            criteria_meta, levels_meta = None, None

        dv_meta_considered = self._compute_dv_meta(params_meta)

        self.model.confidence = self._link_function(dv_meta_considered, params_meta, criteria_meta=criteria_meta,
                                                    levels_meta=levels_meta, constraint_mode=constraint_mode)

        noise_meta = self._noise_meta_transform(self.model.confidence, params_meta, ignore_warnings=ignore_warnings)
        if return_noise:
            return noise_meta
        if (self.cfg.meta_noise_dist == 'beta') and np.any(noise_meta > 0.5):
            if np.max(noise_meta) < 0.5 + 1e-5:
                noise_meta = np.minimum(0.5, noise_meta)
            else:
                warnings.warn(f'max(noise_intercept_meta) = {np.max(noise_meta):.2f}, but maximum allowed '
                              f'value for noise_intercept_meta is 0.5 for metacognitive type '
                              f'{self.cfg.meta_noise_type} and noise model {self.cfg.meta_noise_dist}')
                punishment_factor = np.max(noise_meta) / 0.5
                noise_meta = np.minimum(0.5, noise_meta)

        binsize = self.cfg.binsize_meta if mock_binsize is None else mock_binsize
        if self.cfg.experimental_wrap_binsize_meta:
            wrap_neg = binsize - np.abs(np.minimum(1, self.data.confidence_2d + binsize) - self.data.confidence_2d)
            wrap_pos = binsize - np.abs(np.maximum(0, self.data.confidence_2d - binsize) - self.data.confidence_2d)
            binsize_neg, binsize_pos = binsize + wrap_neg, binsize + wrap_pos
        else:
            binsize_neg, binsize_pos = binsize, binsize
        dist = get_dist(self.cfg.meta_noise_dist, mode=self.model.confidence, scale=noise_meta,
                        meta_noise_type='noisy_report', lookup_table=self.lookup_table)
        # compute the probability of the actual confidence ratings given the pred confidence
        if self.cfg.meta_noise_dist.startswith('censored_'):
            if self.cfg.meta_noise_dist.endswith('gumbel'):
                binsize_neg = np.array(binsize_neg).astype(maxfloat)
                binsize_pos = np.array(binsize_pos).astype(maxfloat)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                window = (dist.cdf(np.minimum(1, self.data.confidence_2d + binsize_pos)) -
                          dist.cdf(np.maximum(0, self.data.confidence_2d - binsize_neg)))
            likelihood = (((self.data.confidence_2d > 1e-8) & (self.data.confidence_2d < 1 - 1e-8)) * window +
                          (self.data.confidence_2d <= 1e-8) * dist.cdf(binsize_pos).astype(np.float64) +
                          (self.data.confidence_2d >= 1 - 1e-8) * (1 - dist.cdf(1 - binsize_neg))).astype(np.float64)
            if final:
                likelihood_pdf = \
                    (((self.data.confidence_2d > 1e-8) & (self.data.confidence_2d < 1 - 1e-8)) *
                        dist.pdf(self.data.confidence_2d.astype(maxfloat)) +
                     (self.data.confidence_2d <= 1e-8) * dist.cdf(binsize_pos) +
                        (self.data.confidence_2d >= 1 - 1e-8) * (1 - dist.cdf(1 - binsize_neg))).astype(np.float64)
            else:
                likelihood_pdf = None
        else:
            with warnings.catch_warnings():
                # catch this warning, which doesn't make any sense (output is valid if this happens)
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats',
                                        message='divide by zero encountered in _beta_cdf')
                likelihood = dist.cdf(np.minimum(1, self.data.confidence_2d + binsize_pos)) - \
                             dist.cdf(np.maximum(0, self.data.confidence_2d - binsize_neg))  # noqa
            if final:
                likelihood_pdf = dist.pdf(self.data.confidence_2d)
            else:
                likelihood_pdf = None

        if not self.cfg.detection_model and not self.cfg.experimental_include_incongruent_dv:
            likelihood[self.model.dv_sens_considered_invalid] = np.nan
            self.model.confidence[self.model.dv_sens_considered_invalid] = np.nan

        return params_meta, noise_meta, self.model.confidence, dist, dv_meta_considered, likelihood, likelihood_pdf, \
            punishment_factor

    def _noise_sens_transform(self, stimuli=None, params_sens=None):
        """
        Helper function to call the sensory noise transformation function.
        """
        if stimuli is None:
            stimuli = self.model.stimuli_final
        if params_sens is None:
            params_sens = self.model.params_sens
        if self.cfg.enable_noise_sens:
            return noise_sens_transform(
                stimuli=stimuli, function_noise_transform_sens=self.cfg.function_noise_transform_sens, **params_sens
            )
        else:
            params_sens = {} if params_sens is None else params_sens
            return noise_sens_transform(
                stimuli=stimuli, function_noise_transform_sens=self.cfg.function_noise_transform_sens,
                **{**params_sens, 'noise_sens': self.cfg.noise_sens_default}
            )

    def _noise_meta_transform(self, confidence_or_dv_meta, params_meta, ignore_warnings=False):
        """
        Helper function to call the metacognitive noise transformation function.
        """
        if self.cfg.enable_noise_meta:
            noise_meta_transformed = noise_meta_transform(
                confidence_or_dv_meta=confidence_or_dv_meta, dv_sens=self.model.dv_sens_considered,
                function_noise_transform_meta=self.cfg.function_noise_transform_meta,
                ignore_warning=ignore_warnings,  # noqa
                **self.model.params_sens, **params_meta
            )
            return np.maximum(self.cfg.noise_meta_min, noise_meta_transformed)
        else:
            return self.cfg.noise_meta_default

    def _link_function(self, dv_meta, params_meta, criteria_meta=None, levels_meta=None, constraint_mode=False):
        """
        Helper function to call the link function.
        """
        return link_function(
            dv_meta=dv_meta, link_fun=self.cfg.meta_link_function, dv_sens=self.model.dv_sens_considered,
            stimuli=self.model.stimuli_final, function_noise_transform_sens=self.cfg.function_noise_transform_sens,
            nchannels=self.cfg.detection_model_nchannels, criteria_meta=criteria_meta, levels_meta=levels_meta,
            constraint_mode=constraint_mode,
            **self.model.params_sens, **params_meta
        )

    def _link_function_inv(self, confidence, link_fun, params_meta, criteria_meta=None, levels_meta=None):
        """
        Helper function to call the inverse link function.
        """
        return link_function_inv(
            confidence=confidence, link_fun=link_fun, stimuli=self.model.stimuli_final,
            dv_sens=self.model.dv_sens_considered, function_noise_transform_sens=self.cfg.function_noise_transform_sens,
            nchannels=self.cfg.detection_model_nchannels, criteria_meta=criteria_meta, levels_meta=levels_meta,
            **self.model.params_sens, **params_meta
        )

    def _compute_dv_sens(self):
        """
        Compute sensory decision values
        """

        noise_sens = self._noise_sens_transform()
        thresh_sens = _check_param(self.model.params_sens['thresh_sens'] if self.cfg.enable_thresh_sens else 0)
        bias_sens = _check_param(self.model.params_sens['bias_sens'] if self.cfg.enable_bias_sens else 0)

        cond_neg, cond_pos = (self.model.stimuli_final < 0).squeeze(), (self.model.stimuli_final >= 0).squeeze()
        self.model.stimuli_final_thresh = np.full(self.model.stimuli_final.shape, np.nan)
        # self.model.stimuli_final_thresh[cond_neg] = self.model.stimuli_final[cond_neg] - \
        #     np.sign(self.model.stimuli_final[cond_neg]) * thresh_sens[0]
        # self.model.stimuli_final_thresh[cond_pos] = self.model.stimuli_final[cond_pos] - \
        #     np.sign(self.model.stimuli_final[cond_pos]) * thresh_sens[1]
        self.model.stimuli_final_thresh[cond_neg] = self.model.stimuli_final[cond_neg]
        self.model.stimuli_final_thresh[cond_pos] = self.model.stimuli_final[cond_pos]
        dv_sens_mode = np.full(self.model.stimuli_final.shape, np.nan)
        dv_sens_mode[cond_neg] = (np.abs(self.model.stimuli_final[cond_neg]) >= thresh_sens[0]) * \
            self.model.stimuli_final_thresh[cond_neg] + bias_sens[0]
        dv_sens_mode[cond_pos] = (np.abs(self.model.stimuli_final[cond_pos]) >= thresh_sens[1]) * \
            self.model.stimuli_final_thresh[cond_pos] + bias_sens[1]

        # self.model.stimuli_final_thresh = self.model.stimuli_final - np.sign(self.model.stimuli_final) * thresh_sens
        # dv_sens_mode = (np.abs(self.model.stimuli_final) >= thresh_sens) * self.model.stimuli_final_thresh - bias_sens

        if self.cfg.detection_model:
            nchannels = self.cfg.detection_model_nchannels  # number of sensory channels
            p_active = np.full(dv_sens_mode.shape, np.nan)  # prob. channel active
            p_active[cond_neg] = np.tanh(np.abs(dv_sens_mode[cond_neg]) / noise_sens[cond_neg])
            p_active[cond_pos] = np.tanh(np.abs(dv_sens_mode[cond_pos]) / noise_sens[cond_pos])

            signs = np.sign(dv_sens_mode)
            dv_sens_mode = signs * p_active * nchannels
            self.model.dv_sens_considered = signs * np.tile(np.arange(0, nchannels + 1), (len(dv_sens_mode), 1))
            self.model.dv_sens_pmf = binom(nchannels, p_active).pmf(np.abs(self.model.dv_sens_considered))

            # this doesn't work, possibly because the beta distribution is zero at p=0?
            # (in contrast to the binomial distribution)
            # nactive = p_active * nchannels
            # ninactive = nchannels - nactive
            # p_active_range = np.linspace(1e-5, 1-1e-5, self.cfg.nbins_dv)
            # self.model.dv_sens_considered = signs * np.tile(p_active_range*nchannels, (len(dv_sens_mode), 1))
            # self.model.dv_sens_pmf = beta(nactive+1, ninactive+1).pdf(p_active_range)

            # signs = np.sign(dv_sens_mode)
            # dv_sens_range = np.linspace(0, np.abs(dv_sens_mode.flatten()) +
            #                             self.cfg.max_dv_deviation, int((self.cfg.nbins_dv))).T
            # self.model.dv_sens_considered = signs * dv_sens_range * noise_transform_sens
            # p_active_considered = np.tanh(np.abs(self.model.dv_sens_considered) / noise_transform_sens)
            # nactive_considered = p_active_considered * nchannels
            # self.model.dv_sens_pmf = binom(nchannels, p_active).pmf(np.round(nactive_considered).astype(int))

            # signs = np.sign(dv_sens_mode)
            # dv_sens_mode = signs * p_active * nchannels
            # nactive_considered = np.tile(np.arange(0, nchannels+1), (len(dv_sens_mode), 1))
            # self.model.dv_sens_considered = signs * noise_transform_sens * \
            #                                 np.arctanh(np.minimum(nchannels-1e-5, nactive_considered) / nchannels)
            # self.model.dv_sens_pmf = binom(nchannels, p_active).pmf(nactive_considered)

        else:
            range_ = np.linspace(0, self.cfg.max_dv_deviation, int((self.cfg.nbins_dv + 1) / 2))[1:]
            dv_sens_range = np.hstack((-range_[::-1], 0, range_))
            self.model.dv_sens_considered = np.full((dv_sens_mode.shape[0], dv_sens_range.shape[0]), np.nan)
            self.model.dv_sens_considered[cond_neg] = dv_sens_mode[cond_neg] + dv_sens_range * noise_sens[cond_neg]
            self.model.dv_sens_considered[cond_pos] = dv_sens_mode[cond_pos] + dv_sens_range * noise_sens[cond_pos]

            logistic_neg = logistic_dist(loc=dv_sens_mode[cond_neg], scale=noise_sens[cond_neg] * np.sqrt(3) / np.pi)
            logistic_pos = logistic_dist(loc=dv_sens_mode[cond_pos], scale=noise_sens[cond_pos] * np.sqrt(3) / np.pi)
            margin_neg = noise_sens[cond_neg] * self.cfg.max_dv_deviation / self.cfg.nbins_dv
            margin_pos = noise_sens[cond_pos] * self.cfg.max_dv_deviation / self.cfg.nbins_dv
            self.model.dv_sens_pmf = np.full(self.model.dv_sens_considered.shape, np.nan)
            self.model.dv_sens_pmf[cond_neg] = (logistic_neg.cdf(self.model.dv_sens_considered[cond_neg] + margin_neg) -
                                                logistic_neg.cdf(self.model.dv_sens_considered[cond_neg] - margin_neg))
            self.model.dv_sens_pmf[cond_pos] = (logistic_pos.cdf(self.model.dv_sens_considered[cond_pos] + margin_pos) -
                                                logistic_pos.cdf(self.model.dv_sens_considered[cond_pos] - margin_pos))
            # normalize PMF
            self.model.dv_sens_pmf = self.model.dv_sens_pmf / self.model.dv_sens_pmf.sum(axis=1).reshape(-1, 1)
            # invalidate invalid decision values
            if not self.cfg.experimental_include_incongruent_dv:
                self.model.dv_sens_considered_invalid = np.sign(self.model.dv_sens_considered) != \
                                                        np.sign(self.data.choices_2d - 0.5)
                self.model.dv_sens_pmf[self.model.dv_sens_considered_invalid] = np.nan

            if self.cfg.experimental_likelihood:
                # self.cfg.binsize_meta*2 is the probability for a given confidence rating assuming a uniform
                # distribution for confidence. This 'confidence guessing model' serves as a upper bound for the
                # negative log likelihood.
                min_likelihood = self.cfg.binsize_meta*2*np.ones(self.model.dv_sens_pmf.shape)
                min_likelihood_weighted_cum = np.nansum(min_likelihood * self.model.dv_sens_pmf, axis=1)
                self.model.max_negll = -np.log(min_likelihood_weighted_cum).sum()

        self.model.dv_sens_considered_abs = np.abs(self.model.dv_sens_considered)
        self.model.dv_sens_mode = dv_sens_mode.flatten()

    def _compute_dv_meta(self, params_meta):
        """
        Compute metacognitive decision values
        """
        if self.cfg.enable_evidence_bias_mult_meta == 1:
            dv_meta_considered = params_meta['evidence_bias_mult_meta'] * self.model.dv_sens_considered_abs
        elif self.cfg.enable_evidence_bias_mult_meta == 2:
            dv_meta_considered = np.full(self.model.dv_sens_considered.shape, np.nan)
            neg, pos = self.model.dv_sens_considered < 0, self.model.dv_sens_considered >= 0
            dv_meta_considered[neg] = params_meta['evidence_bias_mult_meta'][0] * self.model.dv_sens_considered_abs[neg]
            dv_meta_considered[pos] = params_meta['evidence_bias_mult_meta'][1] * self.model.dv_sens_considered_abs[pos]
        else:
            dv_meta_considered = self.model.dv_sens_considered_abs

        if self.cfg.enable_evidence_bias_add_meta == 1:
            dv_meta_considered = np.maximum(0, dv_meta_considered + params_meta['evidence_bias_add_meta'])
        elif self.cfg.enable_evidence_bias_add_meta == 2:
            dv_meta_considered = np.full(self.model.dv_sens_considered.shape, np.nan)
            neg, pos = self.model.dv_sens_considered < 0, self.model.dv_sens_considered >= 0
            dv_meta_considered[neg] = np.maximum(0, dv_meta_considered[neg] + params_meta['evidence_bias_add_meta'][0])
            dv_meta_considered[pos] = np.maximum(0, dv_meta_considered[pos] + params_meta['evidence_bias_add_meta'][1])
        return dv_meta_considered

    def _prepare_criteria(self, params_meta):
        ncrit_meta = int(self.cfg.meta_link_function.split('_')[0])

        if self.cfg.enable_criteria_meta == 2:
            criteria_meta = [[params_meta[f'criterion{i}_meta'][j] for i in range(ncrit_meta)] for j in range(2)]
        else:
            criteria_meta = [params_meta[f'criterion{i}_meta'] for i in range(ncrit_meta)]

        if self.cfg.enable_levels_meta == 2:
            levels_meta = [[params_meta[f'level{i}_meta'][j] for i in range(ncrit_meta)] for j in range(2)]
        elif self.cfg.enable_levels_meta == 1:
            levels_meta = [params_meta[f'level{i}_meta'] for i in range(ncrit_meta)]
        else:
            levels_meta = None

        return criteria_meta, levels_meta

    def _check_fit(self):
        if not self.sens_is_fitted and not self.meta_is_fitted:
            raise RuntimeError('Please fit the model before plotting.')
        elif self.sens_is_fitted and not self.meta_is_fitted:
            raise RuntimeError('Only the sensory level was fitted. Please also fit the metacognitive level to plot'
                               'a link function.')

    def plot_link_function(self, **kwargs):
        self._check_fit()
        plot_link_function(
            self.data.stimuli, self.data.confidence, self.model.dv_sens_mode, self.model.params, cfg=self.cfg, **kwargs
        )

    def plot_confidence_dist(self, **kwargs):
        self._check_fit()
        varlik = self.model.dv_meta_considered if self.cfg.meta_noise_type == 'noisy_readout' else self.model.confidence
        plot_confidence_dist(
            self.cfg, self.data.stimuli, self.data.confidence, self.model.params, var_likelihood=varlik,
            noise_meta_transformed=self.model.noise_meta, dv_sens=self.model.dv_sens_considered,
            likelihood_weighting=self.model.dv_sens_pmf, **kwargs
        )


def load_dataset(name, verbose=True, return_params=False, return_dv_sens=False, return_cfg=False):
    import gzip
    path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'demo_data', f'example_data_{name}.pkl.gz')
    if os.path.exists(path):
        with gzip.open(path, 'rb') as f:
            stimuli, choices, confidence, params, cfg, dv_sens, stats = pickle.load(f)
    else:
        raise FileNotFoundError(f'[Dataset does not exist!] No such file: {path}')

    if verbose:
        print(f"Loading dataset '{name}' which was generated as follows:")
        print('..Generative model:')
        print(f'{TAB}Metatacognitive noise type: {cfg.meta_noise_type}')
        print(f'{TAB}Metatacognitive noise distribution: {cfg.meta_noise_dist}')
        print(f'{TAB}Link function: {cfg.meta_link_function}')
        print('..Generative parameters:')
        for i, (k, v) in enumerate(params.items()):
            if hasattr(v, '__len__'):
                print(f"{TAB}{k}: {[float(f'{x:.4g}') for x in v]}")
            else:
                print(f'{TAB}{k}: {v:.4g}')
        print('..Characteristics:')
        print(f'{TAB}No. subjects: {1 if stimuli.ndim == 1 else len(stimuli)}')
        print(f'{TAB}No. samples: {stimuli.shape[0] if stimuli.ndim == 1 else stimuli.shape[1]}')
        print(f"{TAB}Type 1 performance: {100*stats['performance']:.1f}%")
        if not cfg.skip_meta:
            print(f"{TAB}Avg. confidence: {stats['confidence']:.3f}")
            print(f"{TAB}M-Ratio: {stats['mratio']:.3f}")

    return_list = [stimuli, choices, confidence]
    if return_params:
        return_list += [params]
    if return_cfg:
        return_list += [cfg]
    if return_dv_sens:
        return_list += [dv_sens]
    return tuple(return_list)
