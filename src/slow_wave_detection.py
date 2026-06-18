import numpy as np
import yasa

from config import PATHS, SFREQ

def detect_slow_waves(eeg_data):
    sw = yasa.sw_detect(eeg_data, sfreq=SFREQ)
    return sw

def extract_slow_wave_features(sw_events):
    if sw_events is None or len(sw_events) == 0:
        return None
    return sw_events[['Start', 'Duration', 'Peak', 'NegPeak', 'PosPeak', 'Slope']].values
