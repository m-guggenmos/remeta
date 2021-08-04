import pickle
import numpy as np
from remeta.gendata import simu_data  # noqa

np.random.seed(1)

params = dict(
    noise_sens=0.4,
    bias_sens=0.2,
    noise_meta=0.4,
    slope_meta=1.2
)

cfg = None

data = simu_data(nsubjects=1, nsamples=2000, params=params, cfg=cfg, stimuli_ext=None, verbose=True,
                 stimuli_stepsize=0.25, squeeze=True)

pickle.dump((data.stimuli, data.choices, data.confidence, data.cfg, params), open('example_data_simple.pkl', 'wb'))