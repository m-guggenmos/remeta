import pickle
import numpy as np
import remeta
from remeta.gendata import simu_data  # noqa
import os
import pathlib
from confidence.mca.type2_SDT_MLE import type2_SDT_MLE
from confidence.mca.type2roc import type2roc
from scipy.stats import norm
import gzip

mode = 'simple'
# mode = 'sens_complex'
# mode = 'meta_simple'
# mode = 'meta_complex'
# mode = 'meta'
# mode = 'noisy_readout'
# mode = 'criteria'
# mode = 'criteria_levels'

def conf(x, bounds):
    confidence = np.full(x.shape, np.nan)
    bounds = np.hstack((bounds, np.inf))
    for i, b in enumerate(bounds[:-1]):
        confidence[(bounds[i] <= x) & (x < bounds[i + 1])] = i + 1
    return confidence
bounds = np.arange(0, 0.81, 0.2)


if mode == 'simple':
    nsamples = 1000
    seed = 1
    stimuli_stepsize = 0.25
    params = dict(
        noise_sens=0.7,
        bias_sens=0.2,
        noise_meta=0.1,
        evidence_bias_mult_meta=1.2
    )
    cfg = remeta.Configuration()
elif mode == 'sens_complex':
    nsamples = 5000
    seed = 1
    stimuli_stepsize = 0.02
    params = dict(
        noise_sens=[0.5, 0.7],
        thresh_sens=0.1,
        bias_sens=[0.6, 0.1],
    )
    cfg = remeta.Configuration()
    cfg.enable_noise_sens = 2
    cfg.enable_thresh_sens = 1
    cfg.enable_bias_sens = 2
    cfg.skip_meta = True
elif mode == 'meta_simple':
    nsamples = 1000
    seed = 1
    stimuli_stepsize = 0.25
    params = dict(
        noise_sens=0.6,
        bias_sens=0,
        noise_meta=0.1,
        evidence_bias_mult_meta=0.8,
    )
    cfg = remeta.Configuration()
elif mode == 'meta':
    nsamples = 1000
    seed = 1
    stimuli_stepsize = 0.25
    params = dict(
        noise_sens=1.4,
        bias_sens=-0.1,
        noise_meta=0.1,
        evidence_bias_mult_meta=1.2,
        evidence_bias_add_meta=0.1,
    )
    cfg = remeta.Configuration()
    cfg.enable_evidence_bias_add_meta = 1
elif mode == 'meta_complex':
    nsamples = 5000
    seed = 1
    stimuli_stepsize = 0.25
    params = dict(
        noise_sens=0.6,
        bias_sens=0,
        noise_meta=[0.1, 0.3],
        evidence_bias_mult_meta=0.8,
        evidence_bias_add_meta=0.1,
        confidence_bias_mult_meta=1.2,
    )
    cfg = remeta.Configuration()
    cfg.enable_noise_meta = 2
    cfg.enable_evidence_bias_add_meta = 1
    cfg.enable_confidence_bias_mult_meta = 1
elif mode == 'noisy_readout':
    nsamples = 1000
    seed = 1
    stimuli_stepsize = 0.25
    params = dict(
        noise_sens=0.4,
        bias_sens=0,
        noise_meta=0.1,
        evidence_bias_mult_meta=0.6
    )
    cfg = remeta.Configuration()
    cfg.meta_noise_type = 'noisy_readout'
elif mode == 'criteria':
    nsamples = 1000
    seed = 1
    stimuli_stepsize = 0.25
    params = dict(
        noise_sens=0.6,
        bias_sens=0,
        noise_meta=0.1,
        criteria_meta=[0.1, 0.25, 0.5, 0.8]
    )
    cfg = remeta.Configuration()
    cfg.enable_criteria_meta = 1
    cfg.enable_evidence_bias_mult_meta = 0
    cfg.meta_link_function = '4_criteria'
elif mode == 'criteria_levels':
    nsamples = 1000
    seed = 1
    stimuli_stepsize = 0.25
    params = dict(
        noise_sens=0.6,
        bias_sens=0,
        noise_meta=0.1,
        criteria_meta=[0.3, 0.7],
        levels_meta=[0.2, 0.6]
    )
    cfg = remeta.Configuration()
    cfg.enable_criteria_meta = 1
    cfg.enable_levels_meta = 1
    cfg.enable_evidence_bias_mult_meta = 0
    cfg.meta_link_function = '2_criteria'


np.random.seed(seed)
data = simu_data(nsubjects=1, nsamples=nsamples, params=params, cfg=cfg, stimuli_ext=None, verbose=True,
                 stimuli_stepsize=stimuli_stepsize, squeeze=True)

stats = dict()
stimulus_ids = (data.stimuli >= 0).astype(int)
correct = (data.choices == stimulus_ids).astype(int)
stats['d1'] = norm.ppf(min(1-1e-3, max(1e-3, data.choices[stimulus_ids == 1].mean()))) - \
              norm.ppf(min(1-1e-3, max(1e-3, data.choices[stimulus_ids == 0].mean().mean())))
stats['performance'] = np.mean(correct)
stats['choice_bias'] = data.choices.mean() - 0.5
if not cfg.skip_meta:
    fit = type2_SDT_MLE(stimulus_ids, data.choices, conf(data.confidence, bounds), len(bounds))
    stats['confidence'] = np.mean(data.confidence)
    stats['auroc2'] = type2roc(correct, data.confidence)
    stats['mratio'] = fit.M_ratio

path = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'demo/data', f'example_data_{mode}.pkl.gz')
save = (data.stimuli, data.choices, data.confidence, params, data.cfg, data.dv_sens_prenoise, stats)
with gzip.open(path, "wb") as f:
    pickle.dump(save, f)


# import bz2
# import lzma
# import timeit
# t0 = timeit.default_timer()
# with gzip.open(path, "wb") as f:
#     pickle.dump(save, f)
# print(f'[write, gz] {timeit.default_timer() - t0:.4f} secs')
# t0 = timeit.default_timer()
# with bz2.BZ2File(path.replace('.pkl.gz', '.pkl.bz2'), 'wb') as f:
#     pickle.dump(save, f)
# print(f'[write, bz2] {timeit.default_timer() - t0:.4f} secs')
# t0 = timeit.default_timer()
# with lzma.open(path.replace('.pkl.gz', '.pkl.xz'), "wb") as f:
#     pickle.dump(save, f)
# print(f'[write, lzma] {timeit.default_timer() - t0:.4f} secs')
#
# t0 = timeit.default_timer()
# with gzip.open(path, "rb") as f:
#     pickle.load(f)
# print(f'[read, gz] {timeit.default_timer() - t0:.4f} secs')
# t0 = timeit.default_timer()
# with bz2.BZ2File(path.replace('.pkl.gz', '.pkl.bz2'), 'rb') as f:
#     pickle.load(f)
# print(f'[read, bz2] {timeit.default_timer() - t0:.4f} secs')
# t0 = timeit.default_timer()
# with lzma.open(path.replace('.pkl.xz', '.pkl.xz'), "rb") as f:
#     pickle.load(f)
# print(f'[read, lzma] {timeit.default_timer() - t0:.4f} secs')
