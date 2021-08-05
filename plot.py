import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde, sem

from .gendata import simu_data
from .dist import get_dist, get_likelihood
from .transform import link_function
from .util import _check_param


color_linkfunction = np.array([141, 141, 175]) / 255
color_logistic = color_linkfunction
color_generative_meta = np.array([231, 168, 116]) / 255
color_data = [0.6, 0.6, 0.6]

color_model = np.array([57, 127, 95]) / 255
color_model_wrong = np.array([152, 75, 75]) / 255

symbols = dict(
    warping_sens=r'$\omega$',
    noise_sens=r'$\sigma$',
    noise_transform_sens=r'$\sigma_1$',
    thresh_sens=r'$\vartheta$',
    bias_sens=r'$\delta$',
    noise_meta=r'$\Omega$',
    noise_transform_meta=r'$\Omega_1$',
    readout_term_meta=r'$\Theta$',
    scaling_meta=r'$\Lambda$',
    slope_meta=r'$\Psi$',
    criterion0_meta=r'$\Psi_1$',
    criterion1_meta=r'$\Psi_2$',
    criterion2_meta=r'$\Psi_3$',
    criterion3_meta=r'$\Psi_4$',
    criterion4_meta=r'$\Psi_5$',
    level0_meta=r'$C_1$',
    level1_meta=r'$C_2$',
    level2_meta=r'$C_3$',
    level3_meta=r'$C_4$',
    level4_meta=r'$C_5$'
)


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):  # noqa
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        # title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title


def logistic(x, sigma, thresh, bias):
    beta = np.pi / (np.sqrt(3) * sigma)
    return \
        (np.abs(x) >= thresh) * (
                1 / (1 + np.exp(-beta * (x - bias - np.sign(x) * thresh)))) + \
        (np.abs(x) < thresh) * (1 / (1 + np.exp(beta * bias)))


def posterior_detection(x, sigma, thresh, bias, nchannels):
    y = (np.abs(x) > thresh) * x - bias
    p_active = np.tanh(np.abs(y) / sigma)  # probability that a channel is active
    p_correct = (np.sign(x) == np.sign(y)) * (1 - 0.5 * (1 - p_active) ** nchannels) + \
                (np.sign(x) != np.sign(y)) * 0.5 * (1 - p_active) ** nchannels
    stimulus_ids = (x >= 0).astype(int)
    posterior = (stimulus_ids == 1) * p_correct + (stimulus_ids == 0) * (1 - p_correct)
    return posterior


def linear(x, thresh, bias):
    y = (np.abs(x) > thresh) * (x - np.sign(x) * thresh) - bias
    return y


def tanh(x, beta, thresh, offset):
    return \
        (np.abs(x) > thresh) * (
                (1 - offset) * np.tanh(beta * (x - np.sign(x) * thresh)) + np.sign(x) * offset) + \
        (np.abs(x) <= thresh) * np.sign(x) * offset


def plot_meta_condensed(ax, s, m, nsamples_gen=1000):
    cfg = m.cfg

    if hasattr(m, 'model'):
        params_sens = m.model.params_sens
        params_meta = m.model.params_meta
        if '_criteria' in cfg.meta_link_function:
            params_meta['slope_meta'] = [v for k, v in m.model.params_meta.items() if
                                         'criterion_meta' in k or 'confidence_level' in k]
        data = m.data.data
        stimuli_norm = data.stimuli_norm
        confidence = data.confidence
    else:
        params_sens = m.params_sens
        params_meta = m.params_meta
        stimuli_norm = m.stimuli
        confidence = m.confidence

    simu = simu_data(nsamples_gen, len(stimuli_norm), {**params_sens, **params_meta}, cfg=cfg, stimuli_ext=stimuli_norm,
                     verbose=False)

    if 'readout_term_meta' not in params_meta:
        params_meta['readout_term_meta'] = 0
    if 'thresh_sens' not in params_sens:
        params_sens['thresh_sens'] = 0
    if 'bias_sens' not in params_sens:
        params_sens['bias_sens'] = 0

    levels = np.unique(stimuli_norm)
    nbins = 20

    counts, counts_gen, bins = [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)]
    for k in range(2):
        levels_ = (levels[levels < 0], levels[levels > 0])[k]
        for i, v in enumerate(levels_):
            hist = np.histogram(confidence[stimuli_norm == v], density=True, bins=nbins)
            counts[k] += [hist[0]]
            bins[k] += [hist[1]]
            counts_gen[k] += [np.histogram(simu.confidence[np.tile(stimuli_norm, (nsamples_gen, 1)) == v], density=True,
                                           bins=bins[k][i])[0] / (len(bins[k][i]) - 1)]
    counts = np.array(counts) / np.max(counts)
    counts_gen = np.array(counts_gen) / np.max(counts_gen)
    bins = np.array(bins)

    for k in range(2):
        levels_ = (levels[levels < 0], levels[levels > 0])[k]
        for i, v in enumerate(levels_):
            plt.barh(y=bins[k, i][:-1] + np.diff(bins[k, i]) / 2, width=((1, -1)[k]) * 0.3 * counts[k, i],
                     height=1 / nbins, left=0.005 + v, color=color_data, linewidth=0, alpha=1, zorder=10,
                     label='Data: histogram' if ((k == 0) & (i == 0)) else None)
            plt.plot(v + (1, -1)[k] * 0.3 * counts_gen[k, i], bins[k, i][:-1] + (bins[k, i][1] - bins[k, i][0]) / 2,
                     color=0.85 * color_generative_meta, zorder=11, lw=1.5,
                     label='Model: density' if ((k == 0) & (i == 0)) else None)

    plt.xlim((-1, 1))
    ylim = (-0.01, 1.01)
    plt.plot([0, 0], ylim, 'k-', lw=0.5)
    plt.ylim(ylim)

    if s == 17:
        plt.xlabel('Stimulus ($x$)', fontsize=11)
        ax.xaxis.set_label_coords(1.1, -0.25)
    if s == 8:
        plt.ylabel('Confidence', fontsize=11)
    if s < 16:
        plt.xticks([])
    if np.mod(s, 4) != 0:
        plt.yticks([])
    plt.title(rf"$\Theta$=${params_meta['readout_term_meta']:.2f}$ $\Psi$=${params_meta['slope_meta']:.2f}$" 
              "$\Omega$=${params_meta['noise_meta']:.2f}$",  # noqa
        fontsize=9.5, y=0.97)
    plt.text(0, 0.8, f'{s + 1}', bbox=dict(fc=[0.8, 0.8, 0.8], ec=[0.5, 0.5, 0.5], lw=0.5, pad=2, alpha=0.8),
             fontsize=10, ha='center')


def plot_psychometric(cfg, choices, stimuli, params_sens):

    noise_sens = _check_param(params_sens['noise_sens'])
    thresh_sens = params_sens['thresh_sens'] if cfg.enable_thresh_sens else 0
    bias_sens = params_sens['bias_sens'] if cfg.enable_bias_sens else 0

    stimulus_ids = (stimuli > 0).astype(int)
    levels = np.unique(stimuli)
    xrange_neg = np.arange(-1, 0.001, 0.001)
    xrange_pos = np.arange(0.001, 1.001, 0.001)
    choiceprob_neg = np.array([np.mean(choices[(stimuli == v) & (stimulus_ids == 0)] ==
                                       stimulus_ids[(stimuli == v) & (stimulus_ids == 0)])
                               for v in levels[levels < 0]])
    choiceprob_pos = np.array([np.mean(choices[(stimuli == v) & (stimulus_ids == 1)] ==
                                       stimulus_ids[(stimuli == v) & (stimulus_ids == 1)])
                               for v in levels[levels > 0]])
    if cfg.detection_model:
        posterior_neg = posterior_detection(
            xrange_neg, noise_sens[0], thresh_sens, bias_sens, cfg.detection_model_nchannels
        )
        posterior_pos = posterior_detection(
            xrange_pos, noise_sens[1], thresh_sens, bias_sens, cfg.detection_model_nchannels
        )
    else:
        posterior_neg = logistic(xrange_neg, noise_sens[0], thresh_sens, bias_sens)
        posterior_pos = logistic(xrange_pos, noise_sens[1], thresh_sens, bias_sens)

    ax = plt.gca()

    plt.plot(levels[levels < 0], 1 - choiceprob_neg, 'o', markersize=6.5, mew=1, mec='k', label='Data: $S^-$ Mean',
             color=color_data, clip_on=False, zorder=11)
    plt.plot(levels[levels > 0], choiceprob_pos, 's', markersize=5.5, mew=1, mec='k', label='Data: $S^+$ Mean',
             color=color_data, clip_on=False, zorder=11)

    plt.plot(xrange_neg, posterior_neg, '-', lw=2, color=color_logistic, clip_on=False, zorder=10, label=f'Model fit')
    plt.plot(xrange_pos, posterior_pos, '-', lw=2, color=color_logistic, clip_on=False, zorder=10)

    plt.plot([-1, 1], [0.5, 0.5], 'k-', lw=0.5)
    plt.plot([0, 0], [-0.02, 1.02], 'k-', lw=0.5)

    ax.yaxis.grid('on', color=[0.9, 0.9, 0.9])
    plt.xlim((-1, 1))
    plt.ylim((0, 1))
    plt.xlabel('Stimulus ($x$)')
    plt.ylabel('Choice probability $S^+$')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    leg = plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, handlelength=0.5)
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(1)  # noqa
    anot_sens = []
    for i, (k, v) in enumerate(params_sens.items()):
        if hasattr(v, '__len__'):
            val = ', '.join([f"{p:{'.0f' if p == 0 else '.3f'}}" for p in v])
            anot_sens += [f"${symbols[k][1:-1]}=" + f"[{val}]$"]
        else:
            anot_sens += [f"${symbols[k][1:-1]}={v:{'.0f' if v == 0 else '.3f'}}$"]
    plt.text(1.045, -0.1, r'Parameters:' + '\n' + '\n'.join(anot_sens), transform=plt.gca().transAxes,
             bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=5), fontsize=9)
    set_fontsize(label=13, tick=11)

    return ax


def plot_confidence(stimuli, confidence):
    ax = plt.gca()

    for v in sorted(np.unique(stimuli)):
        plt.errorbar(v, np.mean(confidence[stimuli == v]), yerr=sem(confidence[stimuli == v]), marker='o', markersize=5,
                     mew=1, mec='k', color='None', ecolor='k', mfc=color_data, clip_on=False, elinewidth=1.5,
                     capsize=5)
    plt.plot([0, 0], [0, 1], 'k-', lw=0.5)
    plt.ylim(0, 1)
    plt.xlabel('Stimulus ($x$)')
    plt.ylabel('Confidence')
    set_fontsize(label=13, tick=11)

    return ax


def plot_link_function(cfg, stimuli, confidence, dv_sens_mode, params_sens, params_meta,
                       plot_likelihood=False, var_likelihood=None, noise_meta_transformed=None, dv_range=(45, 50, 55),
                       nsamples_gen=1000, nsamples_dist=100000, bw=0.03):

    ax = plt.gca()

    generative = simu_data(nsamples_gen, len(stimuli), {**params_sens, **params_meta}, cfg=cfg, stimuli_ext=stimuli,
                           verbose=False, squeeze=True)

    vals_dv = np.unique(dv_sens_mode)
    vals_dv_gen = np.unique(generative.dv_sens_mode)
    for k in range(2):

        vals_dv_ = vals_dv[vals_dv < 0] if k == 0 else vals_dv[vals_dv > 0]
        vals_dv_gen_ = vals_dv_gen[vals_dv_gen < 0] if k == 0 else vals_dv_gen[vals_dv_gen > 0]

        conf_data_means = [np.mean(confidence[dv_sens_mode == v]) for v in vals_dv_]
        conf_data_std_neg = [np.std(confidence[(dv_sens_mode == v) & (confidence < conf_data_means[i])]) for i, v in
                             enumerate(vals_dv_)]
        conf_data_std_pos = [np.std(confidence[(dv_sens_mode == v) & (confidence >= conf_data_means[i])]) for i, v in
                             enumerate(vals_dv_)]

        conf_gen_means = [np.mean(generative.confidence[generative.dv_sens_mode == v]) for v in vals_dv_gen_]
        conf_gen_std_neg = [np.std(generative.confidence[
                                       (generative.dv_sens_mode == v) & (generative.confidence < conf_gen_means[i])])
                            for i, v in enumerate(vals_dv_gen_)]
        conf_gen_std_pos = [np.std(generative.confidence[
                                       (generative.dv_sens_mode == v) & (generative.confidence > conf_gen_means[i])])
                            for i, v in enumerate(vals_dv_gen_)]

        _, cap, barlinecols = plt.errorbar(
            vals_dv_-0.015, conf_data_means, yerr=[conf_data_std_neg, conf_data_std_pos],
            label='Data: Mean (SD)' if k == 0 else None, marker='o', markersize=7, mew=1, mec='k', color='None',
            ecolor='k', mfc=color_data, clip_on=False, zorder=35, elinewidth=1.5, capsize=5
        )
        [cap[i].set_markeredgewidth(1.5) for i in range(len(cap))]
        [cap[i].set_clip_on(False) for i in range(len(cap))]
        barlinecols[0].set_clip_on(False)

        _, cap, barlinecols = plt.errorbar(
            vals_dv_gen_+0.015, conf_gen_means, yerr=[conf_gen_std_neg, conf_gen_std_pos],
            label='Generative model' if k == 0 else None, marker='o', markersize=7, mew=1, mec=color_generative_meta,
            color='None', ecolor=color_generative_meta, mfc=color_generative_meta, clip_on=False, zorder=35,
            elinewidth=1.5, capsize=5
        )
        [cap[i].set_markeredgewidth(1.5) for i in range(len(cap))]
        [cap[i].set_clip_on(False) for i in range(len(cap))]
        barlinecols[0].set_clip_on(False)

        if plot_likelihood:
            if cfg.detection_model:
                var_likelihood_means = [np.nanmean(var_likelihood[dv_sens_mode == v]) for v in vals_dv_]
            else:
                var_likelihood_means = [np.nanmean(var_likelihood[dv_sens_mode == v, dv_range[1]]) for v in vals_dv_]

            for i, v in enumerate(vals_dv_):

                noise_meta = noise_meta_transformed[dv_sens_mode == v, int(noise_meta_transformed.shape[1] / 2)].\
                    mean() if noise_meta_transformed.ndim == 2 else noise_meta_transformed
                x = np.linspace(0, 1, 1000)
                if cfg.meta_noise_type == 'noisy_report':
                    likelihood = get_likelihood(x, cfg.meta_noise_model, np.maximum(1e-3, var_likelihood_means[i]),
                                                noise_meta, logarithm=False)
                else:
                    dist = get_dist(cfg.meta_noise_model, np.maximum(1e-3, var_likelihood_means[i]), noise_meta)
                    dv_meta_generative = dist.rvs(nsamples_dist)
                    conf_generative = link_function(
                        dv_meta_generative, cfg.meta_link_function, stimuli=stimuli, dv_sens=dv_meta_generative,
                        function_noise_transform_sens=cfg.function_noise_transform_sens, **params_sens, **params_meta
                    )
                    likelihood = gaussian_kde(conf_generative, bw_method=bw).evaluate(x)
                likelihood -= likelihood.min()
                like_max = likelihood.max()
                likelihood_norm = likelihood / like_max if like_max > 0 else np.zeros(likelihood.shape)
                plt.plot(v + (0.26 * likelihood_norm + 0.005) * ((1, -1)[k]), x, color=color_model, zorder=25, lw=2.5,
                         label=(None, r'Likelihood for $\overline{y}$')[int((k == 0) & (i == 0))])

                ax.annotate(rf"$\mathbf{{\overline{{y}}_{{{('', '+')[k]}{(i - len(vals_dv_), i + 1)[k]}}}}}$",
                            xy=(v, 1.008), xycoords='data', xytext=(v, 1.09), color=color_model, weight='bold',
                            fontsize=9, ha='center', bbox=dict(pad=0, facecolor='w', lw=0),
                            arrowprops=dict(facecolor=color_model, headwidth=7, lw=0, headlength=3, width=2))

        xrange = np.arange(-5, 0.001, 0.001) if k == 0 else np.arange(0, 5.001, 0.001)
        conf_model = link_function(
            np.abs(xrange) + params_meta['readout_term_meta'], cfg.meta_link_function, stimuli=xrange, dv_sens=xrange,
            function_noise_transform_sens=cfg.function_noise_transform_sens, **params_sens, **params_meta
        )
        plt.plot(xrange, conf_model, color=color_linkfunction, lw=3.5, zorder=5, alpha=0.9,
                 label='Link function' if k == 0 else None)

    ylim = (-0.01, 1.01)
    plt.plot([0, 0], ylim, 'k-', lw=0.5)
    plt.ylim(ylim)
    plt.xlim((-1.1 * np.abs(vals_dv).max(), 1.1 * np.abs(vals_dv).max()))
    plt.xlabel(r'Sensory decision value ($y$)')
    plt.ylabel('Confidence')
    handles, labels = plt.gca().get_legend_handles_labels()
    if plot_likelihood:
        order = [2, 1, 0]
        plt.legend([handles[i] for i in order], [labels[i] for i in order], bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=9)
    else:
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid('on', color=[0.9, 0.9, 0.9])

    anot_meta = [f"${symbols[p][1:-1]}={params_meta[p]:{'.0f' if params_meta[p] == 0 else '.3f'}}$" for p in params_meta
                 if not ((p == 'slope_meta') and '_criteria' in cfg.meta_link_function)]
    plt.text(1.045, -0.2, r'Estimated parameters:' + '\n' + '\n'.join(anot_meta), transform=plt.gca().transAxes,
             bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=5), fontsize=9)

    set_fontsize(label=13, tick=11)

    return ax


def plot_confidence_dist(cfg, stimuli, confidence, params_sens, params_meta, nsamples_gen=1000,
                         plot_likelihood=True, var_likelihood=None, noise_meta_transformed=None, dv_sens=None,
                         likelihood_weighting=None, dv_range=(45, 50, 55), nsamples_dist=100000, bw=0.03,
                         figure_paper=False):
    generative = simu_data(nsamples_gen, len(stimuli), {**params_sens, **params_meta}, cfg=cfg, stimuli_ext=stimuli,
                           verbose=False)

    nbins = 20
    levels = np.unique(stimuli)
    counts, counts_gen, bins = [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)]
    for k in range(2):
        levels_ = (levels[levels < 0], levels[levels > 0])[k]
        for i, v in enumerate(levels_):
            hist = np.histogram(confidence[stimuli == v], density=True, bins=nbins)
            counts[k] += [hist[0]]
            bins[k] += [hist[1]]
            counts_gen[k] += [np.histogram(generative.confidence[np.tile(stimuli, (nsamples_gen, 1)) == v],
                                           density=True, bins=bins[k][i])[0] / (len(bins[k][i]) - 1)]
    counts = np.array(counts) / np.max(counts)
    counts_gen = np.array(counts_gen) / np.max(counts_gen)
    bins = np.array(bins)

    ax = plt.gca()

    dist_labels = [r'for $\overline{y}$ $−$ 0.5 SD', r'for $\overline{y}$', r'for $\overline{y}$ $+$ 0.5 SD']
    for k in range(2):

        levels_ = (levels[levels < 0], levels[levels > 0])[k]

        if plot_likelihood:
            confp_means = [[np.nanmean(var_likelihood[stimuli == v, z]) for z in dv_range] for v in levels_]

            weighting_p = np.array(
                [[np.nanmean(likelihood_weighting[stimuli == v, z]) for z in dv_range] for v in levels_])
            weighting_p /= np.max(weighting_p)

        for i, v in enumerate(levels_):

            plt.barh(y=bins[k, i][:-1] + np.diff(bins[k, i]) / 2, width=((1, -1)[k]) * 0.26 * counts[k, i],
                     height=1 / nbins, left=0.005 + v, color=color_data, linewidth=0, alpha=1, zorder=10,
                     label='Data: histogram' if ((k == 0) & (i == 0)) else None)

            plt.plot(v + (1, -1)[k] * 0.26 * counts_gen[k, i], bins[k, i][:-1] + (bins[k, i][1] - bins[k, i][0]) / 2,
                     color=color_generative_meta, zorder=11, lw=2,
                     label='Generative model' if ((k == 0) & (i == 0)) else None)

            if plot_likelihood:
                for j, dv in enumerate(dv_range):
                    noise_meta = noise_meta_transformed[stimuli == v, dv].mean() if noise_meta_transformed.ndim == 2 \
                        else noise_meta_transformed
                    x = np.linspace(0, 1, 1000)
                    if cfg.meta_noise_type == 'noisy_report':
                        likelihood = get_likelihood(x, cfg.meta_noise_model,
                                              np.maximum(1e-3, confp_means[i][j]),  # noqa
                                              noise_meta, logarithm=False)
                    else:
                        dist = get_dist(cfg.meta_noise_model, np.maximum(1e-3, confp_means[i][j]), noise_meta)
                        dv_meta_generative = dist.rvs(nsamples_dist)
                        if 'censored_' in cfg.meta_noise_model:
                            dv_meta_generative[dv_meta_generative < 0] = 0
                        dvm_to_conf = link_function(
                            dv_meta_generative, cfg.meta_link_function, stimuli=dv_meta_generative,
                            function_noise_transform_sens=cfg.function_noise_transform_sens,
                            **params_sens, **params_meta
                        )
                        likelihood = gaussian_kde(dvm_to_conf, bw_method=bw).evaluate(x)
                    likelihood -= likelihood.min()
                    likelihood_max = likelihood.max()
                    likelihood_norm = likelihood / likelihood_max if likelihood_max > 0 else np.zeros(likelihood.shape)
                    likelihood_norm[likelihood_norm < 0.05] = np.nan
                    correct = np.sign(dv_sens[stimuli == v, dv][0]) == (-1, 1)[k]
                    if cfg.detection_model:
                        color_shade = [1]
                    else:
                        color_shade = [[0.175], [0], [0.175]][j]
                    plt.plot(v + (weighting_p[i][j] * 0.26 * likelihood_norm + 0.005) * ((1, -1)[k]),  # noqa
                             x, color=(color_model_wrong, color_model)[int(correct)] + color_shade,
                             zorder=25, lw=2.5, dashes=[(2, 1), (None, None), (None, None)][j],
                             label=(None, dist_labels[j])[int((k == 1) & (i == 1))])

    plt.xlim((-1, 1))
    ylim = (-0.01, 1.01)
    plt.plot([0, 0], ylim, 'k-', lw=0.5)
    plt.ylim(ylim)

    plt.xlabel('Stimulus ($x$)')
    plt.ylabel('Confidence')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    if plot_likelihood:
        handles, labels = plt.gca().get_legend_handles_labels()
        handles += ['', 'Likelihood']
        labels += ['', '']
        if figure_paper:
            labels += [r'for $\overline{y}$ $+$ ' + '0.5 SD\n    (invalid)']
            handles += [Line2D([0], [0], color=color_model_wrong + 0.175,
                               **{k: getattr(handles[3], f'_{k}') for k in ('linestyle', 'linewidth')})]
            order = [4, 0, 5, 6, 2, 1, 3, 7]
        else:
            order = [4, 0, 5, 6, 2, 1, 3]
        plt.legend([handles[i] for i in order], [labels[i] for i in order],
                   bbox_to_anchor=(1.02, 1.1 if figure_paper else 1), loc="upper left", fontsize=9,
                   handler_map={str: LegendTitle()})
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid('on', color=[0.9, 0.9, 0.9], zorder=-10)

    set_fontsize(label=13, tick=11)

    return ax


def plot_sensory_meta(m, plot_subject_id=False, nsamples_gen=1000, figure_paper=False):
    import matplotlib
    matplotlib.mathtext.SHRINK_FACTOR = 0.6

    cfg = m.cfg

    if hasattr(m, 'model'):
        simulation = False
        params_sens = m.model.params_sens
        params_meta = m.model.params_meta
        if '_criteria' in cfg.meta_link_function:
            params_meta['slope_meta'] = [v for k, v in m.model.params_meta.items() if
                                         'criterion_meta' in k or 'confidence_level' in k]
        data = m.data.data
        stimuli_norm = data.stimuli_norm
        choices = data.choices
        confidence = data.confidence
        if m.cfg.detection_model:
            var_likelihood = dict(noisy_report=m.model.extended.confidence_mode.reshape(-1, 1),
                                  noisy_readout=m.model.extended.dv_meta_mode.reshape(-1, 1))[cfg.meta_noise_type]
        else:
            var_likelihood = dict(noisy_report=m.model.extended.confidence,
                                  noisy_readout=m.model.extended.dv_meta)[cfg.meta_noise_type]
        likelihood_weighting = m.model.extended.dv_sens_pmf
        noise_meta_transformed = m.model.extended.noise_meta
        dv_sens = m.model.extended.dv_sens
        dv_sens_mode = m.model.extended.dv_sens_mode
        if m.cfg.detection_model:
            dv_sens = dv_sens_mode.reshape(-1, 1)
    else:
        simulation = True
        params_sens = m.params_sens
        params_meta = m.params_meta
        choices = m.choices
        stimuli_norm = m.stimuli
        confidence = m.confidence
        dv_sens_mode = m.dv_sens_mode.squeeze()
        dv_sens = None
        likelihood_weighting = None
        var_likelihood = None
        noise_meta_transformed = None

    if 'readout_term_meta' not in params_meta:
        params_meta['readout_term_meta'] = 0
    if 'thresh_sens' not in params_sens:
        params_sens['thresh_sens'] = 0
    if 'bias_sens' not in params_sens:
        params_sens['bias_sens'] = 0

    fig = plt.figure(figsize=(8, 7))
    if plot_subject_id and hasattr(m, 'subject_id') and (m.subject_id is not None):
        fig.suptitle(f'Subject {m.subject_id}', fontsize=16)

    plt.subplot(3, 1, 1)
    ax1 = plot_psychometric(cfg, choices, stimuli_norm, params_sens)
    ax1.yaxis.set_label_coords(-0.1, 0.43)
    plt.text(-0.15, 1.01, 'A', transform=ax1.transAxes, color=(0, 0, 0), fontsize=19)

    plt.subplot(3, 1, 2)
    ax2 = plot_link_function(cfg, stimuli_norm, confidence, dv_sens_mode, params_sens, params_meta,
                             plot_likelihood=not simulation, var_likelihood=var_likelihood,
                             noise_meta_transformed=noise_meta_transformed,
                             dv_range=(0,) if simulation else (45, 50, 55))
    plt.text(-0.15, 1.01, 'B', transform=ax2.transAxes, color=(0, 0, 0), fontsize=19)

    plt.subplot(3, 1, 3)
    ax3 = plot_confidence_dist(cfg, stimuli_norm, confidence, params_sens, params_meta, nsamples_gen,
                               plot_likelihood=not simulation, var_likelihood=var_likelihood,
                               noise_meta_transformed=noise_meta_transformed, dv_sens=dv_sens,
                               likelihood_weighting=likelihood_weighting, dv_range=(0,) if simulation else (45, 50, 55),
                               figure_paper=figure_paper)
    plt.text(-0.15, 1.01, 'C', transform=ax3.transAxes, color=(0, 0, 0), fontsize=19)

    # hack to not cut the right edges in saved images
    # if figure_paper:
    #     plt.text(1.29, 1.01, 'C', transform=plt.gca().transAxes, color='r', fontsize=9)

    set_fontsize(label=11, tick=10)
    plt.subplots_adjust(hspace=0.5, top=0.96, right=0.7, left=0.1)
    ax2.set_position([*(np.array(ax2.get_position())[0] + (0, -0.02)),
                      ax2.get_position().width, ax2.get_position().height])
    ax3.set_position([*(np.array(ax3.get_position())[0] + (0, -0.02)),
                      ax3.get_position().width, ax3.get_position().height])


def set_fontsize(label=None, xlabel=None, ylabel=None, tick=None, xtick=None, ytick=None, title=None):

    fig = plt.gcf()

    for ax in fig.axes:
        if xlabel is not None:
            ax.xaxis.label.set_size(xlabel)
        elif label is not None:
            ax.xaxis.label.set_size(label)
        if ylabel is not None:
            ax.yaxis.label.set_size(ylabel)
        elif label is not None:
            ax.yaxis.label.set_size(label)

        if xtick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(xtick)
        elif tick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(tick)
        if ytick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(ytick)
        elif tick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(tick)

        if title is not None:
            ax.title.set_fontsize(title)
