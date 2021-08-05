#%% md

# _ReMeta_ Toolbox

#%%

import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.extend(['../../'])
from remeta.gendata import simu_data
from remeta.plot import plot_psychometric
from remeta.configuration import Configuration
from remeta import ReMeta
print('Imports succesful')
np.random.seed(1)


#%%

options_basic = dict(
    meta_link_function='probability_correct',
    meta_noise_type='noisy_report',
    meta_noise_model='beta'
)

#%%

options_enable = dict(
    enable_warping_sens=0,
    enable_noise_sens=2,  # default 1
    enable_noise_transform_sens=0,
    enable_thresh_sens=0,
    enable_bias_sens=1,
    enable_noise_meta=1,
    enable_noise_transform_meta=0,
    enable_readout_term_meta=1,  # default 0
    enable_slope_meta=1,
    enable_scaling_meta=0,
    enable_criteria_meta=0,
    enable_levels_meta=0
)

#%%

params = dict(
    noise_sens=[0.3, 0.4],
    bias_sens=-0.1,
    noise_meta=0.2,
    readout_term_meta=-0.1,
    slope_meta=1.4,
)

#%%

data = simu_data(nsubjects=1, nsamples=2000, params=params, print_configuration=False, squeeze=True,
                 **options_basic, **options_enable)

#%%

cfg = Configuration(**{**options_basic, **options_enable, 'print_configuration': False})
plot_psychometric(cfg, data.choices, data.stimuli, data.params_sens)

#%%

rem = ReMeta(true_params=params, **{**options_basic, **options_enable, 'print_configuration': False})

#%%

rem.fit(data.stimuli, data.choices, data.confidence, verbose=False)

#%%

result = rem.summary(extended=True)

#%%

print(f'Negative log-likelihood of true sensory parameters: {result.model.fit.fit_sens.negll_true:.1f}')
print(f'Negative log-likelihood of fitted sensory  parameters: {result.model.fit.fit_sens.negll:.1f}')

#%%

for k, v in result.model.params_meta.items():
    print(f'{k} = {v:.3f}')

#%%

print(f'Negative log-likelihood of true metacognitive parameters: {result.model.fit.fit_meta.negll_true:.1f}')
print(f'Negative log-likelihood of fitted metacognitive parameters: {result.model.fit.fit_meta.negll:.1f}')

