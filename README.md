Go directly to:
- [**Basic Usage**](https://github.com/m-guggenmos/remeta/blob/master/demo/basic_usage.ipynb)

# ReMeta Toolbox

The purpose of the ReMeta toolbox ("Reverse engineering of Metacognition") is the estimation of certain latent type 1 and type 2 ('metacognitve') parameters underlying the behavior in decision-making tasks. At present, it is primarily focused on perceptual decision-making tasks with exactly two stimulus categories.

**Important**: a basic prerequisite for the approach of the toolbox is that there are multiple stimulus intensities per stimulus category!

The following parameters are currently supported:

**Type 1 parameters:**
- `noise_sens`: type 1 noise
- `bias_sens`: type 1 bias towards one of the two stimulus categories
- `thresh_sens`: a (sensory) threshold, building on the assumption that a certain minimal stimulus intensity is required to elicit behavior
- `noise_transform_sens`: parameter to specify stimulus-dependent type 1 noise (e.g. multiplicative noise)
- `warping`: a nonlinear transducer parameter, allowing for nonlinear transformations of stimulus intensities

**Type 2 parameters:**
- `noise_meta`: metacognitive 'type 2' noise
- `readout_term_meta`: additive metacognitive bias at the readout of type 1 decision values
- `confidence slope`: multiplicative metacognitive bias applying to the estimation of type 1 noise
- `confidence scaling`: multiplicative metacognitive bias applying to confidence reports
- `noise_transform_meta`: parameter to specify decision-value-dependent type 2 noise (e.g. multiplicative noise)
- `criterion{i}_meta`: i-th confidence criterion (in case of a criterion-based link function)
- `level{i}_meta`: i-th confidence level (in case of a criterion-based link function, confidence levels correspond to the confidence at the respective criteria)

In addition, each parameter can be fitted in "duplex mode", such that separate values are fitted depending on the stimulus category (for type 1 parameters) or depending on the sign of the type 1 decision values (for type 2 parameters).

### Minimal example
Let's start with a minimal example using a pre-generated dataset. In general, we need three pieces of data to fit a model:
- `stimuli`: list/array of signed stimulus intensity values, where the sign codes the stimulus category and the absolute value codes the intensity. The stimuli should be normalized to [-1; 1], although there is a setting for auto-normalizing the stimuli by max(abs(stimuli)).
- `choices`: list/array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive stimulus category.
- `confidence`: list/array of confidence ratings. The (maximal) range of confidence ratings must be [0; 1]. Discrete confidence ratings must be normalized accordingly (e.g., if confidence ratings are 1-4, subtract 1 and divide by 3).

```python
# Minimal example
import remeta
stimuli, choices, confidence = remeta.load_dataset('simple')
rem = remeta.ReMeta()
rem.fit(stimuli, choices, confidence)
```
Output:
```
Loading dataset 'simple' which was generated with the following parameters:
    [true] noise_sens: 0.4
    [true] bias_sens: 0.2
    [true] noise_meta: 0.4
    [true] slope_meta: 1.2
No. subjects: 1, No. samples: 2000

+++ Sensory component +++
Initial neg. LL: 1597.00
    [initial] noise_sens: 0.1
    [initial] bias_sens: 0
Performing local optimization
    [final] noise_sens: 0.405
    [final] bias_sens: 0.179
Final neg. LL: 555.42
Stats: 0.30 secs, 231 fevs

+++ Metacognitive component +++
Initial neg. LL: 13239.78
    [initial] noise_meta: 0.2
    [initial] slope_meta: 1
Grid search activated (grid size = 60)
    [grid] noise_meta: 0.35
    [grid] slope_meta: 0.8
Grid neg. LL: 12924.7
Grid stats: 8.17 secs
Performing local optimization
    [final] noise_meta: 0.400
    [final] slope_meta: 1.168
Final neg. LL: 12911.61
Stats: 3.11 secs, 27 fevs
```

Since the dataset is based on simulation, we know the true parameters of the underlying generative model, which are quite close to the fitted parameters.

By default, the model fits parameters for type 1 noise (`noise_sens`) and a type 1 bias (`bias_sens`), as well as metacognitive 'type 2' noise (`noise_meta`) and a confidence slope (`slope_meta`). Moreover, by default the model assumes that metacognitive noise occurs at the stage of the confidence report (setting `meta_noise_type='noisy_report'`), that observers aim at reporting probability correct with their confidence ratings (setting `meta_link_function='probability_correct'`) and that metacognitive noise can be described by a Beta distribution (setting `meta_noise_model='beta'`).

A more detailed guide to use the toolbox is provided in the following Jupyter notebook: [**Basic Usage**](https://github.com/m-guggenmos/remeta/blob/master/demo/basic_usage.ipynb)