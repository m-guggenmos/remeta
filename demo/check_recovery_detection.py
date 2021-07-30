import numpy as np
import pandas as pd
from scipy.stats import sem
from tabulate import tabulate

from confidence.mca.type2roc import type2roc
from confidence.remeta.gendata import simu_data
from confidence.remeta import ReMeta
from mg.stats.nansem import nansem
# import warnings
# warnings.filterwarnings('error')

seed = 14
np.random.seed(seed)

# Ns
####
nsamples = 10000  # number of trials for each subject
nsubjects = 4



# True parameters
#################
true_params = dict(
    noise_intercept_sens=5,
    noise_slope_sens=0.2,
    noise_intercept_neg_sens=4,
    noise_intercept_pos_sens=6,
    thresh_sens=0.2,
    bias_sens=0.2,
    warping_sens=0.2,
    noise_intercept_meta=0.2,
    noise_slope_meta=0.2,
    slope_or_criteria_meta=0.8,
    # slope_meta=[0.8],
    # slope_or_criteria_neg_meta=0.7,
    # slope_or_criteria_pos_meta=0.8,
    # slope_or_criteria_neg_meta=[0.7],
    # slope_or_criteria_pos_meta=[0.8],
    # slope_or_criteria_neg_meta=[0.273, 0.476, 0.715],
    # slope_or_criteria_pos_meta=[0.373, 0.576, 0.815],
    # slope_or_criteria_neg_meta=[0.273, 0.476],
    # slope_or_criteria_pos_meta=[0.273, 0.476],
    # slope_or_criteria_pos_meta=[0.373, 0.576],
    # slope_meta=[0.273, 0.476, 0.715],
    # slope_meta=[0.273, 0.476],
    # slope_meta=[0.2, 0.35, 0.7],
    readout_term_meta=-0.1
)

options = dict(
    param_warping_sens=False,
    param_duplex_sens=False,
    param_thresh_sens=False,
    param_bias_sens=False,
    param_noise_slope_sens=False,
    param_readout_term_meta=False,
    param_duplex_meta=False,
    param_noise_slope_meta=False,

    function_warping='exponential',
    function_noise_slope_sens='linear',
    function_noise_slope_meta='linear',

    # meta_link_function=f'{len(slope_meta)}_criteria',
    # meta_link_function=f'{len(slope_meta)}_criteria_linear',
    # meta_link_function=f'2_criteria_linear_tanh',
    # meta_link_function=f'1_criteria_linear_tanh',
    meta_link_function=f'1_criteria',
    # meta_link_function='tanh',
    # meta_link_function='detection_model_linear_scaled',
    # meta_link_function='linear',
    # meta_link_function='probability_correct',
    # meta_link_function='probability_correct_ideal',
    meta_noise_type='noisy_readout',
    # meta_noise_type='noisy_report',
    meta_noise_model='truncnorm',
    # meta_noise_model='truncnorm_transform',
    # meta_noise_model='censored_norm',

    grid=True,
    grid_multiproc=False,
    global_minimization=False,

    detection_model=True

)

skip_meta = False
ignore_warnings = False
verbose = True

m = simu_data(nsubjects, nsamples, true_params, **options)

auc = np.full(nsubjects, np.nan)
auc_ideal = np.full(nsubjects, np.nan)
bounds = np.arange(0.2, 1, 0.2)
mratio = np.full(nsubjects, np.nan)
for s in range(nsubjects):
    # auc[s] = type2roc((stimulus_ids[s] == choices[s]).astype(int), np.abs(dv_meta[s]) / np.abs(dv_meta[s]).max())
    auc[s] = type2roc((m.stimulus_ids[s] == m.choices[s]).astype(int), m.confidence[s])
    # mratio[s] = type2_SDT_MLE(m.stimulus_ids[s], m.choices[s], conf(m.confidence[s], bounds), len(bounds)+1).M_ratio
print(f'Performance: {100*(m.stimulus_ids == m.choices).mean()}%')
print(f'AUROC2: {np.mean(auc):.3f} ± {sem(auc):.3f}')
# print(f'M-Ratio: {np.mean(mratio):.3f} ± {sem(mratio):.3f}')
# Absolute confidence ratings (not used here)
# confidence_absolute = np.abs(dv_meta)

# Reconstruct parameters
########################
# param_names_sens = ('warping_sens', 'noise_multi_sens', 'noise_multi_sens', 'thresh_sens', 'bias_sens')
# param_names_meta = ('noise_meta', 'noise_multi_meta', 'slope_meta', 'readout_term_meta')
fit_vars_sens = ('success_sens', 'execution_time_sens')
fit_vars_meta = ('success_meta', 'execution_time_meta')
other_vars = ('ll_sens', 'll_sens_true', 'll_meta', 'll_meta_true', 'mae_confratings', 'best_link_function')
# columns = param_names_sens + param_names_meta + fit_vars_sens + fit_vars_meta + other_vars

rem = ReMeta(true_params=true_params, **options)

def loop(s):
    df = pd.DataFrame(index=range(1))
    np.random.seed(s+seed)
# for s in range(nsubjects):
    print(f'\nReconstructing subject {s + 1} / {nsubjects}')
    rem.fit(m.stimuli[s], m.choices[s], m.confidence[s], skip_meta=skip_meta, verbose=verbose,
            ignore_warnings=ignore_warnings)
    reconstruction = rem.summary(extended=True).model
    for p in reconstruction.params_sens.keys():
        df.loc[0, p] = reconstruction.params_sens[p]
    if not skip_meta:
        for p in reconstruction.params_meta.keys():
            df.loc[0, p] = reconstruction.params_meta[p]
    for v in fit_vars_sens:
        if hasattr(reconstruction.fit_sens, ('_').join(v.split('_')[:-1])):
            df.loc[0, v] = getattr(reconstruction.fit_sens, ('_').join(v.split('_')[:-1]))
    if not skip_meta:
        for v in fit_vars_meta:
            if hasattr(reconstruction.fit_meta, ('_').join(v.split('_')[:-1])):
                df.loc[0, v] = getattr(reconstruction.fit_meta, ('_').join(v.split('_')[:-1]))
    for v in other_vars:
        if hasattr(reconstruction, v):
            df.loc[0, v] = getattr(reconstruction, v)

    if skip_meta:
        df.attrs.update({'params': list(reconstruction.params_sens.keys())})
    else:
        df.attrs.update({'params': list(reconstruction.params_sens.keys()) + list(reconstruction.params_meta.keys())})

    return df

# with Pool(cpu_count() - 1 or 1) as pool:
#     result = list(pool.map(loop, range(nsubjects)))
result = [None] * nsubjects
for s in range(0, nsubjects):
    result[s] = loop(s)
df = pd.concat(result, keys=range(nsubjects)).reset_index().drop(columns='level_1').rename(columns=dict(level_0='subject'))

columns = [c for c in df.columns if c in result[0].attrs['params'] or c in ['success_sens', 'success_meta', 'best_link_function'] or 'll_' in c]
table = [None] * len(columns)
for i, col in enumerate(columns):
    table[i] = [None] * 3
    table[i][0] = col
    if not ((col in true_params and f'param_{col}' in options and not options[f'param_{col}']) or (skip_meta and col in true_params and 'meta' in col)):
        # table[i][1] = (f"{true_params[col][int(list(filter(str.isdigit, col))[0])-1]:.5g}" if 'criterion_meta' in col else f'{true_params[col]:.5g}') if (col in true_params or 'criterion_meta' in col) else ''
        table[i][1] = (f"{true_params[col]:.5g}" if 'criterion_meta' in col else f'{true_params[col]:.5g}') if (col in true_params or 'criterion_meta' in col) else ''
        x = df[col].values
        table[i][2] = np.unique(x, return_counts=True) if isinstance(x[0], (str, )) else f'{np.nanmean(x.astype(float)):.3f} ± {nansem(x):.3f} (median={np.median(x):.3f})'

print(tabulate(table, headers=('Var', 'True', 'Estimate')))
