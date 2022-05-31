from dataclasses import make_dataclass

import numpy as np
from scipy.optimize import OptimizeResult

from .util import TAB, ReprMixin, spearman2d, pearson2d


class Parameter(ReprMixin):
    def __init__(self, guess, bounds, grid_range=None):
        """
        Class that defines the fitting characteristics of a Parameter.

        Parameters
        ----------
        guess : float
            Initial guess for the parameter value.
        bounds: array-like of length 2
            Parameter bounds. The first and second element indicate the lower and upper bound of the parameter.
        grid_range: array-like
            Points to visit in the initial parameter gridsearch search. It makes sense to restrict this to a range of
            _likely_ values, rather than minimal and maximal bounds.
        """
        self.guess = guess
        self.bounds = bounds
        self.grid_range = bounds if grid_range is None else grid_range

    def copy(self):
        return Parameter(self.guess, self.bounds, self.grid_range)


class ParameterSet(ReprMixin):
    def __init__(self, parameters, names, constraints=None):
        """
        Container class for all Parameters of a model.

        Parameters
        ----------
        parameters : dict[str, Parameter]
            The dictionary must have the form {parameter_name1: Parameter(..), parameter_name2: Parameter(..), ..}
        names: List[str]
            List of parameter names of a model.
        constraints: List[Dict]
            List of scipy minimize constraints (each constraint is a dictionary with keys 'type' and 'fun'}
        """
        self.base_names = names
        self.base_is_multiple = [isinstance(parameters[p], list) for p in names]
        self.base_len = [len(parameters[p]) if isinstance(parameters[p], list) else 1 for p in names]  # noqa
        self.names = sum([[f'{p}_{i}' for i in range(len(parameters[p]))] if isinstance(parameters[p], list)  # noqa
                          else [p] for p in names], [])
        self.guess = np.array(sum([[parameters[p][i].guess for i in range(len(parameters[p]))] if  # noqa
                                   isinstance(parameters[p], list) else [parameters[p].guess] for p in names], []))
        self.bounds = np.array(sum([[parameters[p][i].bounds for i in range(len(parameters[p]))] if  # noqa
                                   isinstance(parameters[p], list) else [parameters[p].bounds] for p in names], []))
        self.grid_range = np.array(sum([[parameters[p][i].grid_range for i in range(len(parameters[p]))] if  # noqa
                                   isinstance(parameters[p], list) else [parameters[p].grid_range] for p in names], []),
                                   dtype=object)
        self.constraints = [] if constraints is None else constraints
        self.nparams = len(names)


class Data(ReprMixin):

    def __init__(self, cfg, stimuli, choices, confidence):
        """
        Container class for behavioral data.

        Parameters
        ----------
        cfg : configuration.Configuration
            Settings
        stimuli : array-like of shape (n_samples)
            Array of signed stimulus intensity values, where the sign codes the stimulus category (+: cat1; -: cat2) and
            the absolut value codes the intensity. The scale of the data is not relevant, as a normalisation to [-1; 1]
            is applied.
        choices : array-like of shape (n_samples)
            Array of choices coded as 0 (cat1) and 1 (cat2) for the two stimulus categories. See parameter 'stimuli'
            for the definition of cat1 and cat2.
        confidence : array-like of shape (n_samples)
            Confidence ratings; must be normalized to the range [0;1].
        """

        self.cfg = cfg
        self.stimuli_unnorm = np.array(stimuli)
        self.choices = np.array(choices)
        self.confidence = np.array(confidence)

        self.stimuli = None
        self.stimulus_ids = None
        self.stimuli_unnorm_max = None
        self.stimuli_min = None
        self.correct = None
        self.stimuli_2d = None
        self.choices_2d = None
        self.confidence_2d = None
        self.nsamples = len(self.stimuli_unnorm)

    def preproc(self):

        self.stimulus_ids = (np.sign(self.stimuli_unnorm) == 1).astype(int)
        self.correct = self.stimulus_ids == self.choices

        # convert to 0/1 scheme if choices are provides as -1's and 1's
        if np.array_equal(np.unique(self.choices[~np.isnan(self.choices)]), [-1, 1]):
            self.choices[self.choices == -1] = 0

        self.stimuli_unnorm_max = np.max(np.abs(self.stimuli_unnorm))
        # Normalize stimuli
        if self.cfg.normalize_stimuli_by_max:
            self.stimuli = self.stimuli_unnorm / self.stimuli_unnorm_max
        else:
            if np.max(np.abs(self.stimuli_unnorm)) > 1:
                raise ValueError('Stimuli are not normalized to the range [-1; 1].')
            self.stimuli = self.stimuli_unnorm

        self.stimuli_min = np.abs(self.stimuli).min()

        if self.cfg.confidence_bounds_error > 0:
            self.confidence[self.confidence <= self.cfg.confidence_bounds_error] = 0
            self.confidence[self.confidence >= 1 - self.cfg.confidence_bounds_error] = 1

        self.stimuli_2d = self.stimuli.reshape(-1, 1)
        self.choices_2d = self.choices.reshape(-1, 1)
        self.confidence_2d = self.confidence.reshape(-1, 1)

    def summary(self, full=False):
        desc = dict(
            nsamples=self.nsamples
        )
        if full:
            dict_extended = dict(
                stimulus_ids=(self.stimuli >= 0).astype(int),
                stimuli_unnorm=self.stimuli_unnorm,
                stimuli_norm=self.stimuli,
                choices=self.choices,
                confidence=self.confidence
            )
            data_extended = make_dataclass('DataExtended', dict_extended.keys())
            data_extended.__module__ = '__main__'
            desc.update(dict(data=data_extended(**dict_extended)))
        data_summary = make_dataclass('DataSummary', desc.keys())

        def _repr(self_):
            txt = f'{self_.__class__.__name__}\n'
            txt += '\n'.join([f"\t{k}: {'<not displayed>' if k == 'data' else v}"
                              for k, v in self_.__dict__.items()])
            return txt

        data_summary.__repr__ = _repr
        data_summary.__module__ = '__main__'
        return data_summary(**desc)


class Model(ReprMixin):
    def __init__(self, cfg):
        """
        Container class for behavioral data.

        Parameters
        ----------
        cfg : configuration.Configuration
            Settings
        """
        self.cfg = cfg

        self.super_thresh = None
        self.stimuli_final = None
        self.stimuli_warped_tresh = None
        self.stimuli_warped_super = None
        self.choiceprob = None
        self.posterior = None
        self.dv_sens_considered = None
        self.dv_sens_considered_abs = None
        self.dv_sens_considered_invalid = None
        self.dv_meta_considered = None
        self.dv_sens_mode = None
        self.dv_sens_pmf = None
        self.dv_sens_pmf_renorm = None
        self.dv_meta_mode = None
        self.confidence = None
        self.nsamples = None

        self.likelihood_meta = None
        self.likelihood_meta_mode = None
        self.likelihood_meta_weighted_cum = None
        self.likelihood_meta_weighted_cum_renorm = None
        self.max_negll = None
        self.noise_meta = None

        self.params = None

        self.params_sens = None
        self.params_sens_full = None
        self.params_sens_unnorm = None

        self.params_meta = None
        self.likelihood_dist_meta = None

        self.fit = ModelFit()

    def store_sens(self, negll, params_sens, choiceprob, posterior, stimuli_final, stimulus_norm_coefficent):
        self.params_sens = params_sens
        self.stimuli_final = stimuli_final.reshape(-1, 1)
        self.choiceprob = choiceprob
        self.posterior = posterior
        self.fit.fit_sens.negll = negll
        self.nsamples = len(self.stimuli_final)
        if not self.cfg.enable_noise_sens:
            self.params_sens['noise_sens'] = self.cfg.noise_sens_default
        self.params_sens_unnorm = {k: list(np.array(v) * stimulus_norm_coefficent) if hasattr(v, '__len__') else
                                   v * stimulus_norm_coefficent for k, v in params_sens.items()}

    def store_meta(self, negll, params_meta, noise_meta, likelihood, dv_meta_considered, likelihood_weighted_cum,
                   likelihood_pdf):
        self.params_meta = params_meta
        self.noise_meta = noise_meta
        self.likelihood_meta = likelihood
        self.dv_meta_considered = dv_meta_considered
        if self.cfg.detection_model:
            col_ind = np.round(np.abs(self.dv_sens_mode)).astype(int)
            self.dv_meta_mode = dv_meta_considered[np.arange(len(dv_meta_considered)), col_ind]
            self.likelihood_meta_mode = likelihood[np.arange(len(likelihood)), col_ind]
        else:
            self.dv_meta_mode = dv_meta_considered[:, int((dv_meta_considered.shape[1] - 1) / 2)]
            self.likelihood_meta_mode = likelihood[:, int((likelihood.shape[1] - 1) / 2)]
        self.likelihood_meta_weighted_cum = likelihood_weighted_cum
        self.dv_sens_pmf_renorm = self.dv_sens_pmf / np.nansum(self.dv_sens_pmf, axis=1).reshape(-1, 1)
        self.likelihood_meta_weighted_cum_renorm = np.nansum(likelihood * self.dv_sens_pmf_renorm, axis=1)
        self.fit.fit_meta.negll = negll
        self.fit.fit_meta.negll_pdf = -np.sum(np.log(np.maximum(np.nansum(self.dv_sens_pmf *
                                                                          likelihood_pdf, axis=1), 1e-10)))

    def report_fit_sens(self, verbose=True):
        if verbose:
            for k, v in self.params_sens.items():
                true_string = '' if self.cfg.true_params is None else \
                    (f" (true: [{', '.join([f'{p:.3g}' for p in self.cfg.true_params[k]])}])" if  # noqa
                     hasattr(self.cfg.true_params[k], '__len__') else f' (true: {self.cfg.true_params[k]:.3g})')  # noqa
                value_string = f"[{', '.join([f'{p:.3g}' for p in v])}]" if hasattr(v, '__len__') else f'{v:.3g}'
                print(f'{TAB}[final] {k}: {value_string}{true_string}')
            # if hasattr(self.fit.fit_sens, 'execution_time'):
            #     print(f'Final stats: {self.fit.fit_sens.execution_time:.2g} secs, {self.fit.fit_sens.nfev} fevs')
            print(f'Final neg. LL: {self.fit.fit_sens.negll:.2f}')
            if self.cfg.true_params is not None and hasattr(self.fit.fit_sens, 'negll_true'):
                print(f'Neg. LL using true params: {self.fit.fit_sens.negll_true:.2f}')
            print(f"Total fitting time: {self.fit.fit_sens.execution_time:.2g} secs")

    def report_fit_meta(self, verbose=True):
        if self.cfg.true_params is not None:
            if 'criteria' in self.cfg.meta_link_function:
                for i, k in enumerate(['', '_neg', '_pos']):
                    if f'criteria{k}_meta' in self.cfg.true_params:
                        if self.cfg.enable_levels_meta:
                            self.cfg.true_params.update(
                                {f"{'confidence_level' if np.mod(i, 2) else 'criterion'}_meta_{int(i / 2) + 1}{k}": v
                                 for i, v in
                                 enumerate(self.cfg.true_params[f'criteria{k}_meta'])})
                        else:
                            self.cfg.true_params.update({f"criterion_meta_{i + 1}{k}": v for i, v in
                                                         enumerate(self.cfg.true_params[f'criteria{k}_meta'])})

        if verbose:
            for k, v in self.params_meta.items():
                true_string = '' if self.cfg.true_params is None else \
                    (f" (true: [{', '.join([f'{p:.3g}' for p in self.cfg.true_params[k]])}])" if
                     hasattr(self.cfg.true_params[k], '__len__') else f' (true: {self.cfg.true_params[k]:.3g})')
                value_string = f"[{', '.join([f'{p:.3g}' for p in v])}]" if hasattr(v, '__len__') else f'{v:.3g}'
                print(f'{TAB}[final] {k}: {value_string}{true_string}')
            # if hasattr(self.fit.fit_meta, 'execution_time'):
                # print(f'Final stats: {self.fit.fit_meta.execution_time:.2g} secs, '
                #       f'{self.fit.fit_meta.nfev} fevs')
            print(f'Final neg. LL: {self.fit.fit_meta.negll:.2f}')
            if self.cfg.true_params is not None:
                if verbose:
                    print(f'Neg. LL using true params: {self.fit.fit_meta.negll_true:.2f}')
            print(f"Total fitting time: {self.fit.fit_meta.execution_time:.2g} secs")

    def summary(self, extended=False, fun_meta=None, confidence_gen=None, confidence_emp=None):

        if not self.cfg.skip_meta:
            if self.cfg.detection_model:
                confidence_mode = self.confidence[np.arange(len(self.confidence)),
                                                  np.round(np.abs(self.dv_sens_mode)).astype(int)]
            else:
                confidence_mode = self.confidence[:, int((self.confidence.shape[1] - 1) / 2)]

            if confidence_gen is not None:
                confidence_tiled = np.tile(confidence_emp, (confidence_gen.shape[0], 1))
                self.fit.fit_meta.confidence_gen_pearson = \
                    np.tanh(np.nanmean(np.arctanh(pearson2d(confidence_gen, confidence_tiled))))
                self.fit.fit_meta.confidence_gen_spearman = \
                    np.tanh(np.nanmean(np.arctanh(
                        spearman2d(confidence_gen, confidence_tiled, axis=1))))
                self.fit.fit_meta.confidence_gen_mae = np.nanmean(np.abs(confidence_gen - confidence_emp))
                self.fit.fit_meta.confidence_gen_medae = np.nanmedian(np.abs(confidence_gen - confidence_emp))
            self.fit.fit_meta.negll_persample = self.fit.fit_meta.negll / self.nsamples
            self.fit.fit_meta.negll_prenoise = -np.nansum(np.log(np.maximum(self.likelihood_meta_mode, 1e-10)))
            self.fit.negll = self.fit.fit_sens.negll + self.fit.fit_meta.negll

        desc = dict(
            nsamples=self.nsamples,
            nparams_sens=self.cfg.paramset_sens.nparams,
            params_sens=self.params_sens,
            evidence_sens=dict(
                negll=self.fit.fit_sens.negll,
                aic=2*self.cfg.paramset_sens.nparams + 2*self.fit.fit_sens.negll,
                bic=2*np.log(self.nsamples) + 2*self.fit.fit_sens.negll
            ),
            params=self.params_sens,
            params_sens_unnorm=self.params_sens_unnorm,
            fit=self.fit
        )

        if self.cfg.true_params is not None:
            desc['evidence_sens'].update(
                negll_true=self.fit.fit_sens.negll_true,
                aic_true=2*self.cfg.paramset_sens.nparams + 2*self.fit.fit_sens.negll_true,
                bic_true=2*np.log(self.nsamples) + 2*self.fit.fit_sens.negll_true
            )

        if not self.cfg.skip_meta:
            desc.update(dict(
                nparams_meta=self.cfg.paramset_meta.nparams,
                params_meta=self.params_meta,
                params={**self.params_sens, **self.params_meta},
                nparams=self.cfg.paramset_sens.nparams + self.cfg.paramset_meta.nparams,
                evidence_meta=dict(
                    negll=self.fit.fit_meta.negll,
                    aic=2*self.cfg.paramset_meta.nparams + 2*self.fit.fit_meta.negll,
                    bic=2*np.log(self.nsamples) + 2*self.fit.fit_meta.negll
                )
            ))
            if self.cfg.true_params is not None:
                desc['evidence_meta'].update(
                    negll_true=self.fit.fit_meta.negll_true,
                    aic_true=2*self.cfg.paramset_meta.nparams + 2*self.fit.fit_meta.negll_true,
                    bic_true=2*np.log(self.nsamples) + 2*self.fit.fit_meta.negll_true
                )


            if extended:
                likelihood_01 = fun_meta(self.fit.fit_meta.x, mock_binsize=0.1)[1]
                likelihood_025 = fun_meta(self.fit.fit_meta.x, mock_binsize=0.25)[1]
                dict_extended = dict(
                    noise_meta=self.noise_meta,
                    likelihood=self.likelihood_meta,
                    likelihood_prenoise=self.likelihood_meta_mode,
                    likelihood_weighted_cum=self.likelihood_meta_weighted_cum,
                    likelihood_weighted_cum_renorm_01=np.nansum(likelihood_01 * self.dv_sens_pmf_renorm, axis=1),
                    likelihood_weighted_cum_renorm_025=np.nansum(likelihood_025 * self.dv_sens_pmf_renorm, axis=1),
                    confidence=self.confidence,
                    dv_meta=self.dv_meta_considered,
                    dv_sens=self.dv_sens_considered,
                    dv_sens_pmf=self.dv_sens_pmf,
                    dv_sens_pmf_renorm=self.dv_sens_pmf_renorm,
                    dv_meta_mode=self.dv_meta_mode,
                    dv_sens_mode=self.dv_sens_mode,
                    confidence_mode=confidence_mode,  # noqa
                    choiceprob=self.choiceprob,
                    posterior=self.posterior,
                )
                model_extended = make_dataclass('ModelExtended', dict_extended.keys())
                model_extended.__module__ = '__main__'
                desc.update(dict(extended=model_extended(**dict_extended)))

        model_summary = make_dataclass('ModelSummary', desc.keys())

        def _repr(self_):
            txt = f'{self_.__class__.__name__}'
            for k, v in self_.__dict__.items():
                if k in ('data', 'fit'):
                    txt += f"\n\t{k}: <not displayed>"
                elif k == 'extended':
                    txt += f"\n\t{k}: additional modeling results (attributes: " \
                           f"{', '.join([a for a in self_.extended.__dict__.keys()])})"
                else:
                    txt += f"\n\t{k}: {v}"
            return txt

        model_summary.__repr__ = _repr
        model_summary.__module__ = '__main__'
        return model_summary(**desc)


class ModelFit(ReprMixin):
    fit_sens: OptimizeResult = None
    fit_meta: OptimizeResult = None
