Go directly to:
- [**Installation**](https://github.com/m-guggenmos/remeta/blob/master/INSTALL.md)
- [**Basic Usage**](https://github.com/m-guggenmos/remeta/blob/master/demo/basic_usage.ipynb)
- [**Common use cases**](https://github.com/m-guggenmos/remeta/blob/master/demo/common_use_cases.ipynb)
- [**Exotic use cases**](https://github.com/m-guggenmos/remeta/blob/master/demo/exotic_use_cases.ipynb)

# ReMeta Toolbox

The ReMeta toolbox allows researchers to estimate latent type 1 and type 2 parameters based on data of cognitive or perceptual decision-making tasks with two response categories. 


### Minimal example
Three types of data are required to fit a model:
- `stimuli`: list/array of signed stimulus intensity values, where the sign codes the stimulus category and the absolute value codes the intensity. The stimuli should be normalized to [-1; 1], although there is a setting (`normalize_stimuli_by_max`) to auto-normalize stimuli.
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
Loading dataset 'simple' which was generated as follows:
..Generative model:
    Metatacognitive noise type: noisy_report
    Metatacognitive noise distribution: truncated_norm
    Link function: probability_correct
..Generative parameters:
    noise_sens: 0.7
    bias_sens: 0.2
    noise_meta: 0.1
    evidence_bias_mult_meta: 1.2
..Characteristics:
    No. subjects: 1
    No. samples: 1000
    Type 1 performance: 78.5%
    Avg. confidence: 0.668
    M-Ratio: 0.921
    
+++ Sensory level +++
Initial guess (neg. LL: 1902.65)
    [guess] noise_sens: 0.1
    [guess] bias_sens: 0
Performing local optimization
    [final] noise_sens: 0.745 (true: 0.7)
    [final] bias_sens: 0.24 (true: 0.2)
Final neg. LL: 461.45
Neg. LL using true params: 462.64
Total fitting time: 0.15 secs

+++ Metacognitive level +++
Initial guess (neg. LL: 1938.81)
    [guess] noise_meta: 0.2
    [guess] evidence_bias_mult_meta: 1
Grid search activated (grid size = 60)
    [grid] noise_meta: 0.15
    [grid] evidence_bias_mult_meta: 1.4
Grid neg. LL: 1879.3
Grid runtime: 2.43 secs
Performing local optimization
    [final] noise_meta: 0.102 (true: 0.1)
    [final] evidence_bias_mult_meta: 1.21 (true: 1.2)
Final neg. LL: 1872.24
Neg. LL using true params: 1872.27
Total fitting time: 3.4 secs
```

Since the dataset is based on simulation, we know the true parameters (in brackets above) of the underlying generative model, which are indeed quite close to the fitted parameters.

We can access the fitted parameters by invoking the `summary()` method on the `ReMeta` instance:

```python
# Access fitted parameters
result = rem.summary()
for k, v in result.model.params.items():
    print(f'{k}: {v:.3f}')
```

Ouput:
```
noise_sens: 0.745
bias_sens: 0.240
noise_meta: 0.102
evidence_bias_mult_meta: 1.213
```

By default, the model fits parameters for type 1 noise (`noise_sens`) and a type 1 bias (`bias_sens`), as well as metacognitive 'type 2' noise (`noise_meta`) and a metacognitive bias (`evidence_bias_mult_meta`). Moreover, by default the model assumes that metacognitive noise occurs at the stage of the confidence report (setting `meta_noise_type='noisy_report'`), that observers aim at reporting probability correct with their confidence ratings (setting `meta_link_function='probability_correct'`) and that metacognitive noise can be described by a truncated normal distribution (setting `meta_noise_dist='truncated_norm'`).

All settings can be changed via the `Configuration` object which is optionally passed to the `ReMeta` instance. For example:

```python
cfg = remeta.Configuration()
cfg.meta_noise_type = 'noisy_readout'
rem = remeta.ReMeta(cfg)
...
```

### Supported parameters

_Type 1 parameters_:
- `noise_sens`: type 1 noise
- `bias_sens`: type 1 bias towards one of the two stimulus categories
- `thresh_sens`: a (sensory) threshold, building on the assumption that a certain minimal stimulus intensity is required to elicit behavior; use only if there are stimulus intensities close to threshold
- `noise_transform_sens`: parameter to specify stimulus-dependent type 1 noise (e.g. multiplicative noise)
- `warping`: a nonlinear transducer parameter, allowing for nonlinear transformations of stimulus intensities-

_Type 2 (metacognitive) parameters:_
- `noise_meta`: metacognitive noise
- `evidence_bias_mult_meta`: multiplicative metacognitive bias applying at the level of evidence
- `evidence_bias_add_meta`: additive metacognitive bias applying at the level of evidence
- `confidence_bias_mult_meta`: multiplicative metacognitive bias applying at the level of confidence
- `confidence_bias_add_meta`: additive metacognitive bias applying at the level of confidence
- `confidence_bias_exp_meta`: exponential metacognitive bias applying at the level of confidence
- `noise_transform_meta`: (experimental) parameter to specify decision-value-dependent type 2 noise (e.g. multiplicative noise)
- `criterion{i}_meta`: i-th confidence criterion (in case of a criterion-based link function)
- `level{i}_meta`: i-th confidence level (in case of a criterion-based link function, confidence levels correspond to the confidence at the respective criteria)

In addition, each parameter can be fitted in "duplex mode", such that separate values are fitted depending on the stimulus category (for type 1 parameters) or depending on the sign of the type 1 decision values (for type 2 parameters).

A more detailed guide to use the toolbox is provided in the following Jupyter notebook: [**Basic Usage**](https://github.com/m-guggenmos/remeta/blob/master/demo/basic_usage.ipynb)