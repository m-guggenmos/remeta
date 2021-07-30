import os

import numpy as np
from scipy.stats import truncnorm

xrange_report = np.arange(0, 1.002, 0.002)
xrange_readout = np.hstack((np.arange(0, 1, 0.002), np.exp(np.linspace(0, np.log(10), 500))))
srange_report = np.exp(np.linspace(0.001, np.log(3), 500)) - 1
srange_readout = np.exp(np.linspace(0.001, np.log(6), 500)) - 1

nsamples = 10000000
niter = 1

reload_report = False
reload_readout = False

if reload_report:
    data_report = np.full((niter, len(xrange_report), len(srange_report)), np.nan)
    for i, mode in enumerate(xrange_report):
        print(f'{i + 1} / {len(xrange_report)}')
        for j, scale in enumerate(srange_report):
            dist = truncnorm(-mode / scale, (1 - mode) / scale, loc=mode, scale=scale)
            data_report[:, i, j] = dist.rvs((niter, nsamples)).std(axis=1)
    np.savez_compressed('lookup_truncated_norm_scale_report.npz', mode=xrange_report, scale=srange_report,
                        truncscale=data_report[0])
else:
    data_report = np.load(os.path.join('lookup_truncated_norm_scale_noisy_report.npz'))

if reload_readout:
    data_readout = np.full((niter, len(xrange_readout), len(srange_readout)), np.nan)
    for i, mode in enumerate(xrange_readout):
        print(f'{i + 1} / {len(xrange_readout)}')
        for j, scale in enumerate(srange_readout):
            dist = truncnorm(-mode / scale, np.inf, loc=mode, scale=scale)
            data_readout[:, i, j] = dist.rvs((niter, nsamples)).std(axis=1)
    np.savez_compressed('lookup_truncated_norm_scale_readout.npz', mode=xrange_readout, scale=srange_readout,
                        truncscale=data_readout[0])
else:
    data_readout = np.load(os.path.join('lookup_truncated_norm_scale_noisy_readout.npz'))
