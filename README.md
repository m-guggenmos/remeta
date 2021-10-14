Go directly to:
- [**Basic Usage**](https://github.com/m-guggenmos/remeta/blob/master/demo/basic_usage.ipynb)
- [**Installation**](https://github.com/m-guggenmos/remeta/blob/master/INSTALL.md)

# ReMeta Toolbox

The ReMeta toolbox allows researchers to estimate latent type 1 and type 2 parameters based on data of cognitive or perceptual decision-making tasks. At present, it is constrained to tasks with exactly two stimulus categories and requires multiple stimulus intensities per stimulus category.


### Minimal example
Three types of data are required to fit a model:
- `stimuli`: list/array of signed stimulus intensity values, where the sign codes the stimulus category and the absolute value codes the intensity. The stimuli should be normalized to [-1; 1], although there is a setting for auto-normalizing the stimuli by max(abs(stimuli)).
- `choices`: list/array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive stimulus category.
- `confidence`: list/array of confidence ratings. Confidence ratings must be normalized to [0; 1]. Discrete confidence ratings must be normalized accordingly (e.g., if confidence ratings are 1-4, subtract 1 and divide by 3).

A minimal example would be the following:
```python
# Minimal example
import remeta
stimuli, choices, confidence = remeta.load_dataset('simple')  # load example dataset
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

+++ Sensory level +++
Initial neg. LL: 1597.00
    [initial] noise_sens: 0.1
    [initial] bias_sens: 0
Performing local optimization
    [final] noise_sens: 0.405
    [final] bias_sens: 0.179
Final neg. LL: 555.42
Stats: 0.30 secs, 231 fevs

+++ Metacognitive level +++
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

Since the dataset is based on simulation, we know the true parameters of the underlying generative model, which are indeed quite close to the fitted parameters.

By default, the model fits parameters for type 1 noise (`noise_sens`) and a type 1 bias (`bias_sens`), as well as metacognitive 'type 2' noise (`noise_meta`) and a confidence slope (`slope_meta`). Moreover, by default the model assumes that metacognitive noise occurs at the stage of the confidence report (setting `meta_noise_type='noisy_report'`), that observers aim at reporting probability correct with their confidence ratings (setting `meta_link_function='probability_correct'`) and that metacognitive noise can be described by a Beta distribution (setting `meta_noise_model='beta'`).

### Supported parameters

_Type 1 parameters_:
- `noise_sens`: type 1 noise
- `bias_sens`: type 1 bias towards one of the two stimulus categories
- `thresh_sens`: a (sensory) threshold, building on the assumption that a certain minimal stimulus intensity is required to elicit behavior
- `noise_transform_sens`: parameter to specify stimulus-dependent type 1 noise (e.g. multiplicative noise)
- `warping`: a nonlinear transducer parameter, allowing for nonlinear transformations of stimulus intensities

_Type 2 (metacognitive) parameters:_
- `noise_meta`: metacognitive noise
- `readout_term_meta`: additive metacognitive bias at the readout of type 1 decision values
- `slope_meta`: multiplicative metacognitive bias applying to the estimation of type 1 noise
- `scaling_meta`: multiplicative metacognitive bias applying to confidence reports
- `noise_transform_meta`: parameter to specify decision-value-dependent type 2 noise (e.g. multiplicative noise)
- `criterion{i}_meta`: i-th confidence criterion (in case of a criterion-based link function)
- `level{i}_meta`: i-th confidence level (in case of a criterion-based link function, confidence levels correspond to the confidence at the respective criteria)

In addition, each parameter can be fitted in "duplex mode", such that separate values are fitted depending on the stimulus category (for type 1 parameters) or depending on the sign of the type 1 decision values (for type 2 parameters).

A more detailed guide to use the toolbox is provided in the following Jupyter notebook: [**Basic Usage**](https://github.com/m-guggenmos/remeta/blob/master/demo/basic_usage.ipynb)