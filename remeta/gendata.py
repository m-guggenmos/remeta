import numpy as np
from scipy.stats import logistic as logistic_dist

from .configuration import Configuration
from .dist import get_dist
from .transform import warp, noise_meta_transform, noise_sens_transform, \
    link_function
from .util import _check_param, TAB, type2roc


class Simulation:
    def __init__(self, nsubjects, nsamples, params, cfg, stimulus_ids, stimuli, choices, dv_sens, dv_sens_mode,
                 dv_meta, confidence, noise_meta, likelihood_dist=None):
        self.nsubjects = nsubjects
        self.nsamples = nsamples
        self.params = params
        self.params_sens = {k: v for k, v in self.params.items() if k.endswith('_sens')}
        self.params_meta = {k: v for k, v in self.params.items() if k.endswith('_meta')}
        self.cfg = cfg
        self.stimulus_ids = stimulus_ids
        self.stimuli = stimuli
        self.choices = choices
        self.correct = stimulus_ids == choices
        self.dv_sens = dv_sens
        self.dv_sens_mode = dv_sens_mode
        self.dv_meta = dv_meta
        self.confidence = confidence
        self.noise_meta = noise_meta
        self.likelihood_dist = likelihood_dist

    def squeeze(self):
        for var in ('stimulus_ids', 'stimuli', 'choices', 'correct', 'dv_sens', 'dv_meta', 'confidence'):
            setattr(self, var, getattr(self, var).squeeze())
        return self


def generate_stimuli(nsubjects, nsamples, stepsize=0.02):
    levels = np.hstack((-np.arange(stepsize, 1.01, stepsize)[::-1], np.arange(stepsize, 1.01, stepsize)))
    stimuli = np.array([np.random.permutation(np.tile(levels, int(nsamples / len(levels)))) for _ in range(nsubjects)])
    return stimuli


def simu_type1_responses(stimuli, params, cfg):
    if cfg.enable_warping_sens:
        stimuli_final = warp(stimuli, params['warping_sens'], cfg.function_warping_sens)
    else:
        stimuli_final = stimuli
    if cfg.enable_noise_sens:
        noise_sens = noise_sens_transform(
            stimuli=stimuli, function_noise_transform_sens=cfg.function_noise_transform_sens, **params)
    else:
        noise_sens = cfg.noise_sens_default

    thresh_sens = _check_param(params['thresh_sens'] if cfg.enable_thresh_sens else 0)
    bias_sens = _check_param(params['bias_sens'] if cfg.enable_bias_sens else 0)

    dv_sens_before_noise = np.full(stimuli_final.shape, np.nan)
    # dv_sens_before_noise[stimuli_final < 0] = (np.abs(stimuli_final[stimuli_final < 0]) > thresh_sens[0]) * \
    #     (stimuli_final[stimuli_final < 0] - np.sign(stimuli_final[stimuli_final < 0]) * thresh_sens[0]) - bias_sens[0]
    # dv_sens_before_noise[stimuli_final >= 0] = (np.abs(stimuli_final[stimuli_final >= 0]) > thresh_sens[1]) * \
    #     (stimuli_final[stimuli_final >= 0] - np.sign(stimuli_final[stimuli_final >= 0]) * thresh_sens[1]) - bias_sens[1]
    dv_sens_before_noise[stimuli_final < 0] = (np.abs(stimuli_final[stimuli_final < 0]) > thresh_sens[0]) * \
        stimuli_final[stimuli_final < 0] - bias_sens[0]
    dv_sens_before_noise[stimuli_final >= 0] = (np.abs(stimuli_final[stimuli_final >= 0]) > thresh_sens[1]) * \
        stimuli_final[stimuli_final >= 0] - bias_sens[1]

    if cfg.detection_model:
        nchannels = cfg.detection_model_nchannels  # number of sensory channels
        p_active = np.tanh(np.abs(dv_sens_before_noise) / noise_sens)  # probability that a channel is active
        active = np.repeat(p_active[:, :, None], nchannels, axis=2) > np.random.rand(*dv_sens_before_noise.shape,
                                                                                     nchannels)
        nactive = np.sum(active, axis=2)  # = k
        correct = np.zeros(dv_sens_before_noise.shape)
        correct[(nactive > 0) & (np.sign(dv_sens_before_noise) == np.sign(stimuli_final))] = 1
        correct[(nactive > 0) & (np.sign(dv_sens_before_noise) != np.sign(stimuli_final))] = 0
        correct[nactive == 0] = np.random.randint(0, 2, np.sum(nactive == 0))
        stimulus_ids = (np.sign(stimuli) > 0).astype(int)
        choices = (correct == stimulus_ids).astype(int)
        dv_sens_before_noise = p_active * nchannels
        dv_sens = nactive
        # if the decision value should be in original space, use this:
        # dv_sens = np.sign(dv_sens_before_noise) * noise_transform_sens * \
        #     np.arctanh(np.minimum(nchannels-1e-5, nactive) / nchannels)
    else:
        dv_sens = dv_sens_before_noise + logistic_dist(scale=noise_sens * np.sqrt(3) / np.pi).rvs(size=stimuli.shape)
        choices = (dv_sens >= 0).astype(int)

    return choices, stimuli_final, dv_sens, dv_sens_before_noise


def simu_data(nsubjects, nsamples, params, cfg=None, stimuli_ext=None, verbose=True, stimuli_stepsize=0.02,
              squeeze=False, force_settings=True, **kwargs):
    if cfg is None:
        # Set configuration attributes that match keyword arguments
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in Configuration.__dict__}
        cfg = Configuration(force_settings=force_settings, **cfg_kwargs)

    if cfg.meta_noise_model is None:
        cfg.meta_noise_model = dict(noisy_report='beta', noisy_readout='gamma')[cfg.meta_noise_type]

    if cfg.meta_noise_model == 'truncated_norm_transform':
        if cfg.meta_noise_type == 'noisy_report':
            lookup_table = np.load('lookup_truncated_norm_noisy_report.npz')
        elif cfg.meta_noise_type == 'noisy_readout':
            lookup_table = np.load('lookup_truncated_norm_noisy_readout.npz')
    else:
        lookup_table = None

    # Make sure no unwanted parameters have been passed
    for p in ('warping', 'thresh', 'bias', 'noise_transform'):
        if not getattr(cfg, f'enable_{p}_sens'):
            params.pop(f'{p}_sens', None)
    for p in ('readout_term', 'noise_transform', 'confidence_term', 'scaling', 'slope'):
        if not getattr(cfg, f'enable_{p}_meta'):
            params.pop(f'{p}_meta', None)
    if not cfg.enable_noise_sens:
        params['noise_sens'] = cfg.noise_sens_default
    if not cfg.enable_noise_meta:
        params['noise_meta'] = cfg.noise_meta_default

    if cfg.enable_criteria_meta:
        ncrit_meta = int(cfg.meta_link_function.split('_')[0])
        if cfg.enable_criteria_meta == 2:
            params['criteria_meta'] = [[params[f'criterion{i}_meta'][j] for i in range(ncrit_meta)] for j in range(2)]
        else:
            params['criteria_meta'] = [params[f'criterion{i}_meta'] for i in range(ncrit_meta)]
        for i in range(ncrit_meta):
            params.pop(f'criterion{i}_meta', None)
    else:
        for p_ in [p for p in params if p.startswith('criterion')]:
            params.pop(p_, None)
    if cfg.enable_levels_meta:
        ncrit_meta = int(cfg.meta_link_function.split('_')[0])
        if cfg.enable_levels_meta == 2:
            params['levels_meta'] = [[params[f'level{i}_meta'][j] for i in range(ncrit_meta)] for j in range(2)]
        else:
            params['levels_meta'] = [params[f'level{i}_meta'] for i in range(ncrit_meta)]
        for i in range(ncrit_meta):
            params.pop(f'level{i}_meta', None)
    else:
        for p_ in [p for p in params if p.startswith('level')]:
            params.pop(p_, None)

    if stimuli_ext is None:
        stimuli = generate_stimuli(nsubjects, nsamples, stepsize=stimuli_stepsize)
    else:
        stimuli = stimuli_ext / np.max(np.abs(stimuli_ext))
        if stimuli_ext.shape != (nsubjects, nsamples):
            stimuli = np.tile(stimuli, (nsubjects, 1))
    stimulus_ids = (np.sign(stimuli) > 0).astype(int)
    choices, stimuli_final, dv_sens, dv_sens_mode = simu_type1_responses(stimuli, params, cfg)

    if cfg.meta_noise_type is None:
        dv_meta, confidence, noise_meta = None, None, None
    else:
        if cfg.enable_readout_term_meta == 1:
            dv_meta_before_noise = np.maximum(0, np.abs(dv_sens) + params['readout_term_meta'])
        elif cfg.enable_readout_term_meta == 2:
            dv_meta_before_noise = np.full(dv_sens.shape, np.nan)
            neg, pos = dv_sens < 0, dv_sens >= 0
            dv_meta_before_noise[neg] = np.maximum(0, np.abs(dv_sens[neg]) + params['readout_term_meta'][0])
            dv_meta_before_noise[pos] = np.maximum(0, np.abs(dv_sens[pos]) + params['readout_term_meta'][1])
        else:
            dv_meta_before_noise = np.maximum(0, np.abs(dv_sens))
        if cfg.meta_noise_type == 'noisy_readout':

            if cfg.enable_noise_meta:
                noise_meta = noise_meta_transform(
                    dv_meta_before_noise, dv_sens=dv_sens,
                    function_noise_transform_meta=cfg.function_noise_transform_meta, **params
                )
            else:
                noise_meta = cfg.noise_meta_default
            dist = get_dist(cfg.meta_noise_model, mode=dv_meta_before_noise, scale=noise_meta,
                            meta_noise_type=cfg.meta_noise_type, lookup_table=lookup_table)  # noqa

            dv_meta = np.maximum(0, dist.rvs((nsubjects, nsamples)))
        else:
            dv_meta = dv_meta_before_noise

        confidence = link_function(
            dv_meta=dv_meta, link_fun=cfg.meta_link_function, dv_sens=dv_sens, stimuli=stimuli_final,
            function_noise_transform_sens=cfg.function_noise_transform_sens,
            **params
        )

        if cfg.meta_noise_type == 'noisy_report':
            if cfg.enable_noise_meta:
                noise_meta = noise_meta_transform(
                    confidence, dv_sens=dv_sens, function_noise_transform_meta=cfg.function_noise_transform_meta,
                    **params
                )
                noise_meta = np.maximum(cfg.noise_meta_min, noise_meta)
            else:
                noise_meta = cfg.noise_meta_default

            if cfg.meta_noise_model == 'beta':
                if np.any(noise_meta > 0.5):
                    raise ValueError(f'max(noise_meta) = {np.max(noise_meta):.2f}, but maximum allowed value for '
                                     f'noise_meta is 0.5 for metacognitive type {cfg.meta_noise_type} and noise model '
                                     f'{cfg.meta_noise_model}')

            dist = get_dist(cfg.meta_noise_model, mode=confidence, scale=noise_meta,
                            meta_noise_type=cfg.meta_noise_type, lookup_table=lookup_table)
            confidence = np.maximum(0, np.minimum(1, dist.rvs((nsubjects, nsamples))))

    if squeeze:
        stimulus_ids = stimulus_ids.squeeze()
        stimuli = stimuli.squeeze()
        choices = choices.squeeze()
        dv_sens = dv_sens.squeeze()
        dv_meta = dv_meta.squeeze()
        confidence = confidence.squeeze()

    simulation = Simulation(
        nsubjects=nsubjects, nsamples=nsamples, params=params, cfg=cfg,
        stimulus_ids=stimulus_ids, stimuli=stimuli, choices=choices,
        dv_sens=dv_sens, dv_sens_mode=dv_sens_mode, dv_meta=dv_meta,
        confidence=confidence,
        noise_meta=noise_meta  # noqa
    )
    if verbose:
        print('----------------------------------')
        print('Basic stats of the simulated data:')
        correct = (stimulus_ids == choices).astype(int)  # noqa
        print(f'{TAB}Performance: {100 * np.mean(correct):.1f}% correct')
        print(f'{TAB}Confidence: {confidence.mean():.2f}')
        choice_bias = 100*choices.mean()
        print(f"{TAB}Choice bias: {('-', '+')[int(choice_bias > 50)]}{np.abs(choice_bias - 50):.1f}%")
        print(f'{TAB}AUROC2: {type2roc(correct, confidence):.2f}')
        print('----------------------------------')

    return simulation


if __name__ == '__main__':
    params_simulation = dict(
        noise_sens=0.2,
        thresh_sens=0.2,
        bias_sens=0.2,
        noise_meta=0.2,
        readout_term_meta=0
    )
    options = dict(meta_noise_type='noisy_report', enable_thresh_sens=1, enable_bias_sens=1, enable_readout_term_meta=1)
    m = simu_data(1, 1000, params_simulation, **options)
    # model = reconstruct_observer(m.stimuli[0], m.choices[0], m.confidence[0], **{**options, **params_simulation})
    # plot_sensory_meta(model)
