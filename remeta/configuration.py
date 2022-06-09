import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Union, List

import numpy as np
from scipy.optimize.slsqp import _epsilon  # noqa

from .modelspec import Parameter, ParameterSet
from .util import ReprMixin
from importlib.util import find_spec


@dataclass
class Configuration(ReprMixin):
    """
    Configuration for the ReMeta toolbox

    Parameters
    ----------
    *** Basic definition of the model ***
    skip_meta : bool
        If False, only fit the sensory level.
    meta_noise_type : str
        Whether the model considers noise at readout or report.
        Possible values: 'noisy_report', 'noisy_readout'
    meta_noise_dist : str
        Metacognitive noise distribution.
        Possible valus: 'truncated_norm', 'truncated_gumbel', 'truncated_lognorm',
                        'norm', 'gumbel', 'lognorm', 'beta', 'gamma',
                        'censored_norm', 'censored_gumbel', 'censored_lognorm', 'censored_gamma'
                        Experimental (for internal use only):
                        'beta_std', 'betaprime', 'lognorm_varstd',
                        'censored_lognorm', 'censored_lognorm_varstd', 'censored_betaprime',
                        'truncated_norm_fit', 'truncated_lognorm_varstd',
                        Works only with lookup table (not deployed via pip):
                        'truncated_norm_lookup', 'truncated_gumbel_lookup'
    meta_link_function : str
        Metacognitive link function. In case of criterion-based link functions {x} refers to the number of criteria.
        Possible values: 'probability_correct', 'identity', 'tanh', 'normcdf', 'erf', 'alg', 'guder', 'linear',
                         '{x}_criteria', '{x}_criteria_linear', '{x}_criteria_linear_tanh'
                         Note: replace {x} with the number of criteria
                         Experimental (for internal use only):
                         'detection_model_linear', 'detection_model_mean', 'detection_model_mode',
                         'detection_model_full', 'detection_model_ideal'

    detection_model : bool
        Experimental. Define the model as a detection model.
    detection_model_nchannels : int
        Defines the number of sensory channels in case of a detection model.

    *** Enable or disable specific parameters ***
    * Each setting can take the values 0, 1 or 2:
    *    0: Disable parameter.
    *    1: Enable parameter.
    *    2: Enable parameter and fit separate values for the negative and positive stimulus category (in the case of
    *       sensory parameters, indicated by the suffix '_sens') or negative and positive decision values (in the
    *       case of metacognitive parameters, indicated by the suffix '_meta').
    enable_warping_sens : int (default: 0)
        Fit a non-linear transducer prior to sensory processing (the nonlinear function is defined via
        `function_warping_sens`).
    enable_noise_sens : int (default: 1)
        Fit separate sensory noise parameters for both stimulus categories.
    enable_noise_transform_sens : int (default: 0)
        Fit an additional sensory noise parameter for signal-dependent sensory noise (the type of dependency is is
        defined via `function_noise_transform_sens`).
    enable_thresh_sens : int (default: 0)
        Fit a sensory threshold.
    enable_bias_sens : int (default: 1)
        Fit a sensory bias towards one of the stimulus categories.
    enable_noise_meta : int (default: 1)
        Fit a metacognitive noise parameter
    enable_noise_transform_meta : int (default: 0)
        Fit an additional metacognitive noise parameter for signal-dependent metacognitive noise (the type of dependency
        is defined via `function_noise_transform_meta`). Note: at present, enable_noise_transform_meta=2 leads to
        biased results and is therefore discouraged.
    enable_evidence_bias_add_meta : int (default: 0)
        Fit an additive metacognitive bias loading on sensory evidence.
    evidence_bias_mult_meta : int (default: 0)
        Fit a multiplicative metacognitive bias loading on evidence. In the case of a criterion-based link function,
        multiple parameters are fitted for each criterion/confidence level.
    evidence_bias_mult_postnoise_meta : int (default: 0)
        Fit a multiplicative metacognitive bias loading on evidence, but operating after the application of readout
        noise. In the case of a criterion-based link function, multiple parameters are fitted for each
        criterion/confidence level.
    enable_confidence_bias_mult_meta : int (default: 0)
        Fit a multiplicative metacognitive bias loading on confidence.
    enable_confidence_bias_add_meta : int (default: 0)
        Fit an additive metacognitive bias loading on confidence.
    enable_criteria_meta : int (default: 0)
        Fit criteria for a criterion-based link function. Note that the number of criteria is set via the link function
        argument (see `meta_link_function`)
    enable_levels_meta : int (default: 0)
        Fit confidence levels for a criterion-based link function. If disabled, equidistant confidence levels are
        assumed (e.g. 1/3, 2/3, 1 for three criteria).


    *** Define fitting characteristics of the parameters ***
    * The fitting of each parameter is characzerized as follows:
    *     1) An initial guess.
    *     2) Lower and upper bound.
    *     3) Grid range, i.e. the range of values that are tested during the initial gridsearch search.
    * Sensible default values are provided for all parameters. To tweak those, one can either define an entire
    * ParameterSet, which is a container for a set of parameters, or each parameter individually. Note that the
    * parameters must be either defined as a Parameter instance or as List[Parameter] in case when separate values are
    * fitted for the positive and negative stimulus category/decision value).
    paramset_sens : ParameterSet
        Parameter set for the sensory level.
    paramset_meta : ParameterSet
        Parameter set for the metacognitive level.

    warping_sens : Union[Parameter, List[Parameter]]
        Parameter for the nonlinear transducer.
    noise_sens : Union[Parameter, List[Parameter]]
        Parameter for sensory noise.
    noise_transform_sens : Union[Parameter, List[Parameter]]
        Parameter for multiplicative sensory noise.
    thresh_sens : Union[Parameter, List[Parameter]]
        Parameter for the sensory threshold.
    bias_sens : Union[Parameter, List[Parameter]]
        Parameter for the sensory bias.
    noise_meta : Union[Parameter, List[Parameter]]
        Parameter for metacognitive noise.
    noise_transform_meta : Union[Parameter, List[Parameter]]
        Parameter for multiplicative megacognitive noise.
    evidence_bias_mult_meta : Union[Parameter, List[Parameter]]
        Parameter for multiplicative metacognitive bias loading on evidence.
    evidence_bias_add_meta : Union[Parameter, List[Parameter]]
        Parameter for additive metacognitive bias loading on evidence.
    evidence_bias_mult_postnoise_meta : Union[Parameter, List[Parameter]]
        Parameter for multiplicative metacognitive bias loading on evidence, but after the application of readout noise.
    confidence_bias_mult_meta : Union[Parameter, List[Parameter]]
        Parameter for multiplicative metacognitive bias loading on confidence.
    confidence_bias_add_meta : Union[Parameter, List[Parameter]]
        Parameter for additive metacognitive bias loading on confidence.
    criterion{x}_meta : Union[Parameter, List[Parameter]]
        Parameter for the xth confidence criterion (0-based).
    level{x}_meta : Union[Parameter, List[Parameter]]
        Parameter for the xth confidence level (0-based).

    *** Constraints ***
    * Constraints for the parameters are provided in the form of a function, as defined below.
    constraints_sens_callable : Callable
        Constraints for sensory parameters. See method `_constraints_sens_default` in this module for the expected
        format of this Callable. See method `_constraints_sens_default` in this module for the expected
        format of this Callable. The function must return a list of dictionaries, where each
        dictionary specifies a scipy optimize constraint.
    constraints_meta_callable : Callable
        Constraints for metacognitive parameters. The function must take the configuration object as the first argument
        and the stimulus array as the second argument. The function must return a list of dictionaries, where each
        dictionary specifies a scipy optimize constraint.

    *** Methodoligcal aspects of parameter fitting ***
    * Note: this applies to the fitting of metacognitive parameters only.
    gridsearch : bool (default: True)
        If True, perform initial (usually coarse) gridsearch search, based on the gridsearch defined for a Parameter.
    fine_gridsearch : bool (default: False)
        If True, perform an iteratively finer gridsearch search for each parameter.
    grid_multiproc : bool (default: False)
        If True, use all available cores for the gridsearch search. If False, use a single core.
    global_minimization : bool (default: False)
        If True, use a global minimization routine.
    gradient_method : str or Tuple/List (default: 'slsqp')
        Set scipy.optimize.minimize gradient method
        If provided as Tuple/List, test different gradient methods and take the best
    gradient_free : bool (default: None)
        If True, use a gradien-free optimization routine.
    slsqp_epsilon : float or Tuple/List (default: None)
        Set parameter epsilon parameter for the SLSQP optimization method.
        If provided as Tuple/List, test different eps parameters and take the best
    init_nelder_mead : bool (default: False)
        If True, jump-start parameter minimization with the gradient-free Nelder-Mead algorithm

    *** Transformation functions ***
    function_warping_sens: str (default: 'power')
        Can be one of 'power', 'exponential' or 'identity'.
    function_noise_transform_sens: str (default: 'multiplicative')
        Can be one of 'multiplicative', 'power', 'exponential' or 'logarithm'.
    function_noise_transform_meta: str (default: 'multiplicative')
        Can be one of 'multiplicative', 'power', 'exponential' or 'logarithm'.

    *** Preprocessing ***
    normalize_stimuli_by_max : bool (default: True)
        If True, normalize provided stimuli by their maximum value.
    confidence_bounds_error : float
        Set confidence < confidence_bounds_error to 0 and > confidence_bounds_erroror to 1. This might be useful
        for continuous scales, if 'finger errors' are assumed responsible for the fact that confidence sometimes is not
        *exactly* 0 or 1 but very close (i.e. the assumption is that observers intended to indicate a confidence of 0/1
        but failed to do so exactly)

    *** Parameters for the metacognitive likelihood computation ***
    min_likelihood_sens : float
        Minimum probability used during the sensory likelihood computation
    min_likelihood_meta : float
        Minimum probability used during the metacognitive likelihood computation
    binsize_meta : float
        Integration bin size for the computation of the likelihood around empirical confidence values (noisy-report)
        or metacognitive evidence (noisy-readout).
    max_dv_deviation : int
        Number of standard deviations around the mean considered for sensory uncertainty.
    nbins_dv : int
        Number of discrete decision values bins that are considered to represent sensory uncertainty.
    experimental_likelihood : bool
        Instead of using a minimum probability during the likelihood computation, use a maximum cumulative
        likelihood based on a 'guessing' model

    *** Other ***
    true_params : Dict
        Pass true (known) parameter values. This can be useful for testing to compare the likelihood of true and
        fitted parameters. The likelihood of true parameters is returned (and printed).
    force_settings : bool
        Some setting combinations are known to be incompatible and/or to produce biased fits. If True, fit the model
        nevertheless.
    settings_ignore_warnings : bool (default: False)
        If True, ignore warnings about user-specified settings.
    print_configuration : bool (default: True)
        If True, print the configuration at instatiation of the ReMeta class.
    """

    skip_meta: bool = False
    meta_noise_type: str = 'noisy_report'
    meta_noise_dist: str = 'truncated_norm'
    meta_link_function: str = 'probability_correct'
    detection_model: bool = False
    detection_model_nchannels: int = 10

    enable_warping_sens: int = 0
    enable_noise_sens: int = 1
    enable_thresh_sens: int = 0
    enable_bias_sens: int = 1
    enable_noise_meta: int = 1
    enable_evidence_bias_mult_meta: int = 1
    enable_evidence_bias_add_meta: int = 0
    enable_confidence_bias_mult_meta: int = 0
    enable_confidence_bias_add_meta: int = 0
    enable_criteria_meta: int = 0
    enable_levels_meta: int = 0
    # Experimental:
    enable_noise_transform_sens: int = 0
    enable_noise_transform_meta: int = 0
    enable_evidence_bias_mult_postnoise_meta: int = 0

    paramset_sens: ParameterSet = None
    paramset_meta: ParameterSet = None

    warping_sens: Union[Parameter, List[Parameter]] = None
    noise_sens: Union[Parameter, List[Parameter]] = None
    noise_transform_sens: Union[Parameter, List[Parameter]] = None
    thresh_sens: Union[Parameter, List[Parameter]] = None
    bias_sens: Union[Parameter, List[Parameter]] = None
    noise_meta: Union[Parameter, List[Parameter]] = None
    noise_transform_meta: Union[Parameter, List[Parameter]] = None
    evidence_bias_mult_meta: Union[Parameter, List[Parameter]] = None
    evidence_bias_add_meta: Union[Parameter, List[Parameter]] = None
    evidence_bias_mult_postnoise_meta: Union[Parameter, List[Parameter]] = None
    confidence_bias_mult_meta: Union[Parameter, List[Parameter]] = None
    confidence_bias_add_meta: Union[Parameter, List[Parameter]] = None

    constraints_sens_callable: Callable = None
    constraints_meta_callable: Callable = None

    function_warping_sens: str = 'power'
    function_noise_transform_sens: str = 'multiplicative'
    function_noise_transform_meta: str = 'multiplicative'

    gridsearch: bool = True
    fine_gridsearch: bool = False
    grid_multiproc: bool = False
    global_minimization: bool = False
    gradient_method: str = 'slsqp'
    gradient_free: bool = None
    slsqp_epsilon: float = None
    init_nelder_mead: bool = False

    normalize_stimuli_by_max: bool = True
    confidence_bounds_error: float = 0

    min_likelihood_meta: float = 1e-10
    min_likelihood_sens: float = 1e-10
    binsize_meta: float = 1e-1
    max_dv_deviation: int = 5
    nbins_dv: int = 101
    experimental_likelihood: bool = False
    experimental_wrap_binsize_meta: bool = False
    experimental_include_incongruent_dv: bool = False

    true_params: Dict = None
    force_settings: bool = False
    settings_ignore_warnings: bool = False
    print_configuration: bool = False

    noise_sens_default: float = 0.001
    noise_meta_default: float = 0.1
    noise_meta_min: float = 0.001

    _warping_sens_default: Parameter = Parameter(guess=0.1, bounds=(-10, 10), grid_range=np.arange(-10, 11, 5))
    _noise_transform_sens_default: Parameter = Parameter(guess=0, bounds=(0, 10), grid_range=np.arange(0, 1.1, 0.25))
    _noise_sens_default: Parameter = Parameter(guess=0.1, bounds=(1e-3, 100), grid_range=np.arange(0.1, 0.9, 0.25))
    _thresh_sens_default: Parameter = Parameter(guess=0, bounds=(0, 1), grid_range=np.arange(0, 0.41, 0.2))
    _bias_sens_default: Parameter = Parameter(guess=0, bounds=(-1, 1), grid_range=np.arange(-0.1, 0.11, 0.1))
    _noise_meta_default: Parameter = Parameter(guess=0.2, bounds=(1e-5, 50), grid_range=np.arange(0.05, 1, 0.1))
    _noise_transform_meta_default: Parameter = Parameter(guess=0, bounds=(0, 10),
                                                         grid_range=np.arange(0, 1.1, 0.25))
    _evidence_bias_mult_meta_default: Parameter = Parameter(guess=1, bounds=(0.1, 50),
                                                            grid_range=np.arange(0.2, 1.71, 0.3))
    _evidence_bias_add_meta_default: Parameter = Parameter(guess=0, bounds=(-1, 1),
                                                           grid_range=np.arange(-0.2, 0.21, 0.1))
    _evidence_bias_mult_postnoise_meta_default: Parameter = Parameter(guess=1, bounds=(0.1, 50),
                                                                      grid_range=np.arange(0.2, 1.71, 0.3))
    _confidence_bias_mult_meta_default: Parameter = Parameter(guess=1, bounds=(0.1, 10),
                                                              grid_range=np.arange(0.5, 2.01, 0.3))
    _confidence_bias_add_meta_default: Parameter = Parameter(guess=0, bounds=(-1, 1),
                                                             grid_range=np.arange(-0.3, 0.31, 0.15))
    _criterion_meta_default: Parameter = Parameter(guess=0, bounds=(1e-6, 50),
                                                   grid_range=np.exp(np.linspace(0, np.log(2), 8)) - 0.9)
    _level_meta_default: Parameter = Parameter(guess=0, bounds=(1e-6, 1),
                                               grid_range=np.exp(np.linspace(0, np.log(2), 8)) - 0.9)

    def setup(self):

        if self.gradient_free is None:
            if '_criteria' in self.meta_link_function or '_transform' in self.meta_noise_dist:
                self.gradient_free = True
            else:
                self.gradient_free = False

        if self.slsqp_epsilon is None:
            if self.meta_noise_type == 'noisy_readout':
                self.slsqp_epsilon = 1e-4
            else:
                self.slsqp_epsilon = _epsilon

        if self._confidence_bias_mult_meta_default is None and (self.meta_noise_type == 'noisy_readout'):
            self._confidence_bias_mult_meta_default = Parameter(guess=1, bounds=(0.1, 10),
                                                                grid_range=np.arange(0.4, 1.01, 0.2))

        if find_spec('multiprocessing_on_dill') is None:
            warnings.warn(f'Multiprocessing on dill is not installed. Setting grid_multiproc is changed to False.')
            self.grid_multiproc = False

        self._check_compatibility()

        self._prepare_params_sens()
        self._prepare_params_meta()
        if self.print_configuration:
            self.print()

    def _check_compatibility(self):

        if self.enable_noise_transform_meta:
            text = 'Fitting signal-dependent metacognitive noise parameters leads to biased estimates (for currently ' \
                   'unknown reasons) and is thus discouraged.'
            if self.force_settings:
                warnings.warn(text)
            else:
                raise ValueError(text)
        # if self.enable_noise_transform_meta == 2:
        #     warnings.warn('Fitting separate signal-dependent metacognitive noise parameters for the two stimulus '
        #                   'categories leads to biased estimates (for currently unknown reasons) and is thus '
        #                   'discouraged.')

        if self.detection_model and 'detection_model' not in self.meta_link_function:
            raise ValueError('Detection Models require a compatible metacogntive link function. Choose one of: '
                             'detection_model_linear, detection_model_mean, detection_model_mode, '
                             'detection_model_full, detection_model_ideal')

        if not self.enable_criteria_meta and '_criteria' in self.meta_link_function:
            raise ValueError('A criterion-based link function was set, but confidence criteria were not enabled.')

        if not self.settings_ignore_warnings:

            if not self.enable_noise_meta:
                warnings.warn(f'Setting enable_noise_meta=False was provided -> noise_meta is set to its default value '
                              f'({self.noise_meta_default}). You may change this value via the configuration.')

            if self.enable_criteria_meta and '_criteria' not in self.meta_link_function:
                self.enable_criteria_meta = 0
                warnings.warn('Confidence criteria were enabled but no criterion-based link function was set -> '
                              'auto-setting enable_criteria_meta = 0.')
            if self.enable_levels_meta and '_criteria' not in self.meta_link_function:
                self.enable_levels_meta = 0
                warnings.warn('Confidence criteria were enabled but no criterion-based link function was set -> '
                              'auto-setting enable_levels_meta = 0.')
            if self.enable_levels_meta and not self.enable_criteria_meta:
                self.enable_criteria_meta = self.enable_levels_meta
                warnings.warn(f'Confidence leves were enabled, but confidence criteria not -> auto-setting'
                              f'enable_criteria_meta = {self.enable_levels_meta}')

            if self.meta_noise_type == 'noisy_readout':
                if self.enable_confidence_bias_add_meta:
                    warnings.warn('The setting enable_confidence_bias_add_meta has been enabled for a model of type '
                                  'noisy-readout. This setting results in a link function that is not monotonically '
                                  'increasing and likely leads to biased estimates.')
                if self.enable_confidence_bias_mult_meta:
                    warnings.warn('The setting enable_confidence_bias_mult_meta has been enabled for a model of type '
                                  'noisy-readout. Use this only if you have strong reasons to belief that values of '
                                  'this parameter are <= 1, since true values > 1 cannot be recovered for '
                                  'noisy-readout models.')

            if self.enable_criteria_meta and self.enable_evidence_bias_mult_meta:
                self.enable_evidence_bias_mult_meta = 0
                warnings.warn('Confidence criteria were enabled, which is in conflict with '
                              'enable_evidence_bias_mult_meta > 0 -> auto-setting enable_evidence_bias_mult_meta = 0.')

            if self.enable_criteria_meta and self.enable_evidence_bias_mult_postnoise_meta:
                self.enable_evidence_bias_mult_postnoise_meta = 0
                warnings.warn('Confidence criteria were enabled, which is in conflict with '
                              'enable_evidence_bias_mult_postnoise_meta > 0 -> auto-setting '
                              'enable_evidence_bias_mult_postnoise_meta = 0.')

    def _prepare_params_sens(self):
        if self.paramset_sens is None:

            param_names_sens = []
            params_sens = ('warping', 'noise', 'noise_transform', 'thresh', 'bias')
            for param in params_sens:
                if getattr(self, f'enable_{param}_sens'):
                    param_names_sens += [f'{param}_sens']
                    if getattr(self, f'{param}_sens') is None:
                        param_default = getattr(self, f'_{param}_sens_default')
                        if getattr(self, f'enable_{param}_sens') == 2:
                            setattr(self, f'{param}_sens', [param_default, param_default])
                        else:
                            setattr(self, f'{param}_sens', param_default)

            parameters = {k: getattr(self, k) for k in param_names_sens}
            self.paramset_sens = ParameterSet(parameters, param_names_sens)

        if self.constraints_sens_callable is None:
            self.constraints_sens_callable = _constraints_sens_default

    def _prepare_params_meta(self):

        if self.paramset_meta is None:

            if self.enable_noise_meta and self.noise_meta is None:
                if self.meta_noise_dist == 'beta':
                    self._noise_meta_default.bounds = (1e-5, 0.5)
                    self._noise_meta_default.grid_range = np.arange(0.05, 0.5, 0.05)
                elif self.meta_noise_type == 'noisy_readout':
                    self._noise_meta_default.bounds = (1e-5, 250)

            param_names_meta = []
            params_meta = ('noise', 'noise_transform', 'evidence_bias_mult', 'evidence_bias_add',
                           'evidence_bias_mult_postnoise', 'confidence_bias_mult', 'confidence_bias_add')
            for param in params_meta:
                if getattr(self, f'enable_{param}_meta'):
                    param_names_meta += [f'{param}_meta']
                    if getattr(self, f'{param}_meta') is None:
                        param_default = getattr(self, f'_{param}_meta_default')
                        if getattr(self, f'enable_{param}_meta') == 2:
                            setattr(self, f'{param}_meta', [param_default.copy(), param_default.copy()])
                        else:
                            setattr(self, f'{param}_meta', param_default.copy())

            if self.enable_criteria_meta and '_criteria' in self.meta_link_function:  # noqa
                ncriteria_meta = int(self.meta_link_function.split('_')[0])
                guess_criteria = np.linspace(1 / (ncriteria_meta + 1), 1 - 1 / (ncriteria_meta + 1), ncriteria_meta)

                for i in range(ncriteria_meta):
                    if not hasattr(self, f'criterion{i}_meta') or getattr(self, f'criterion{i}_meta') is None:  # noqa
                        param_names_meta += [f'criterion{i}_meta']
                        param_default = self._criterion_meta_default.copy()
                        param_default.guess = guess_criteria[i]
                        if self.enable_criteria_meta == 2:
                            setattr(self, f'criterion{i}_meta', [param_default.copy(), param_default.copy()])
                        else:
                            setattr(self, f'criterion{i}_meta', param_default)
                    if self.enable_levels_meta and \
                            (not hasattr(self, f'level{i}_meta') or getattr(self, f'level{i}_meta') is None):
                        param_names_meta += [f'level{i}_meta']
                        param_default = self._level_meta_default.copy()
                        param_default.guess = guess_criteria[i]
                        if self.enable_levels_meta == 2:
                            setattr(self, f'level{i}_meta', [param_default.copy(), param_default.copy()])
                        else:
                            setattr(self, f'level{i}_meta', param_default)

            parameters = {k: getattr(self, k) for k in param_names_meta}
            self.paramset_meta = ParameterSet(parameters, param_names_meta)

        if self.constraints_meta_callable is None:
            self.constraints_meta_callable = _constraints_meta_default

    def print(self):
        # print('***********************')
        print(f'{self.__class__.__name__}')
        for k, v in self.__dict__.items():
            if not self.skip_meta or ('meta_' not in k and '_meta' not in k):
                print('\n'.join([f'\t{k}: {v}']))
        # print('***********************')

    def __repr__(self):
        txt = f'{self.__class__.__name__}\n'
        txt += '\n'.join([f'\t{k}: {v}' for k, v in self.__dict__.items()])
        return txt


def _constraints_sens_default(remeta):
    constraints = []
    if remeta.cfg.enable_noise_transform_sens:
        constraints += [{'type': 'ineq', 'fun': lambda x: np.min(remeta._negll_sens(x, return_noise=True)) - 0.001}]  # noqa
    return constraints


def _constraints_meta_default(remeta):
    constraints = []

    # Set constraints for the metacognitive noise slope
    # Note that metacognitive noise is based on confidence in case of the noisy-report model and
    # on metacognitive decision values in case of the noisy-readout model
    if remeta.cfg.enable_noise_transform_meta:
        constraints += [{'type': 'ineq', 'fun': lambda x: np.min(remeta.fun_meta_helper(x, return_noise=True,
                                                                                        constraint_mode=True)) - 0.001}]

    if '_criteria' in remeta.cfg.meta_link_function:
        ncrit_meta = int(remeta.cfg.meta_link_function.split('_')[0])
        # in case of a linear_tanh link function, the last parameter criterion corresponds to the slope of the tanh
        ncrit_meta_ = ncrit_meta - 1 if 'linear_tanh' in remeta.cfg.meta_link_function else ncrit_meta
        if remeta.cfg.enable_criteria_meta == 2:
            def fun_criteria(x):
                crit = remeta.fun_meta_helper(x, return_criteria=True, constraint_mode=True)
                return np.sum([(crit[0][i] > (0 if i == 0 else crit[0][i - 1])) for i in range(ncrit_meta_)]) + \
                    np.sum([(crit[1][i] > (0 if i == 0 else crit[1][i - 1])) for i in range(ncrit_meta_)]) - \
                    2 * ncrit_meta_
        else:
            def fun_criteria(x):
                crit = remeta.fun_meta_helper(x, return_criteria=True, constraint_mode=True)
                return np.sum([(crit[i] > (0 if i == 0 else crit[i - 1])) for i in range(ncrit_meta_)]) - ncrit_meta_
        constraints += [{'type': 'eq', 'fun': fun_criteria}]
        if remeta.cfg.enable_levels_meta:
            if remeta.cfg.enable_levels_meta == 2:
                def fun_levels(x):
                    lev = remeta.fun_meta_helper(x, return_levels=True, constraint_mode=True)
                    return np.sum([(lev[0][i] > (0 if i == 0 else lev[0][i - 1])) for i in range(ncrit_meta)]) + \
                        np.sum([(lev[1][i] > (0 if i == 0 else lev[1][i - 1])) for i in range(ncrit_meta)]) - \
                        2 * ncrit_meta
            else:
                def fun_levels(x):
                    lev = remeta.fun_meta_helper(x, return_levels=True, constraint_mode=True)
                    return np.sum([(lev[i] > (0 if i == 0 else lev[i - 1])) for i in range(ncrit_meta)]) - ncrit_meta
            constraints += [{'type': 'eq', 'fun': fun_levels}]
    return constraints
