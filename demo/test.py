import remeta
import warnings
warnings.filterwarnings('error')
stimuli, choices, confidence = remeta.load_dataset('simple')  # load example dataset
rem = remeta.ReMeta()
rem.fit(stimuli, choices, confidence)