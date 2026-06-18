import os
import mne
import yasa
import numpy as np
from glob import glob

from config import PATHS, SFREQ, EPOCH_SEC, SAMPLES_PER_EPOCH, ARTIFACT_THRESHOLD

def preprocess_file(file_path):
    subject_id = os.path.basename(file_path).replace('.edf', '')
    
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw.pick('eeg')
    raw.filter(0.5, 40.0, fir_design='firwin', verbose=False)
    
    if raw.info['sfreq'] != SFREQ:
        raw.resample(SFREQ, verbose=False)
    
    raw.set_eeg_reference('average', verbose=False)
    
    sls = yasa.SleepStaging(raw, eeg_name=raw.ch_names[0])
    hypno = sls.predict()
    hypno_num = hypno.as_int()
    
    data = raw.get_data()
    nrem_epochs = []
    
    for i, stage in enumerate(hypno_num):
        if stage in [1, 2, 3]:
            start = i * SAMPLES_PER_EPOCH
            stop = start + SAMPLES_PER_EPOCH
            if stop <= data.shape[1]:
                epoch = data[:, start:stop]
                if np.max(np.ptp(epoch, axis=1)) < ARTIFACT_THRESHOLD:
                    nrem_epochs.append(epoch)
    
    if nrem_epochs:
        epoch_stack = np.stack(nrem_epochs)
        out_path = f"{PATHS['preprocessed']}/{subject_id}_nrem_epochs.npy"
        np.save(out_path, epoch_stack)
        return subject_id, len(nrem_epochs)
    
    return subject_id, 0

def preprocess_dataset(dataset_name='dreams'):
    edf_files = sorted(glob(f'{PATHS["dreams_raw"]}/**/*.edf', recursive=True))
    results = []
    
    for file in edf_files:
        subject_id, n_epochs = preprocess_file(file)
        results.append({'subject': subject_id, 'epochs': n_epochs})
    
    return results
