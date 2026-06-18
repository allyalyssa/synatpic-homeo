import os
import requests
import pandas as pd
import mne
import patoolib
from glob import glob

from config import PATHS

def download_dreams():
    os.makedirs(PATHS['dreams_raw'], exist_ok=True)
    api_url = "https://zenodo.org/api/records/2650142"
    
    r = requests.get(api_url)
    files = r.json().get('files', [])
    
    target_file = 'DatabasePatients.rar'
    f_info = next((f for f in files if f['key'] == target_file), None)
    
    if f_info:
        rar_path = os.path.join(PATHS['dreams_raw'], target_file)
        if not os.path.exists(rar_path):
            res = requests.get(f_info['links']['self'])
            with open(rar_path, 'wb') as f:
                f.write(res.content)
        
        patoolib.extract_archive(rar_path, outdir=PATHS['dreams_raw'], verbosity=-1)

def summarize_dreams():
    edf_files = sorted(glob(f'{PATHS["dreams_raw"]}/**/*.edf', recursive=True))
    summary_data = []
    
    for file in edf_files:
        raw = mne.io.read_raw_edf(file, preload=False, verbose=False)
        raw_eeg = raw.copy().pick_types(eeg=True, eog=False, emg=False)
        
        subject_id = os.path.basename(file).replace('.edf', '')
        sfreq = raw_eeg.info['sfreq']
        duration_min = (raw_eeg.n_times / sfreq) / 60
        
        summary_data.append({
            'subject_id': subject_id,
            'n_channels': len(raw_eeg.ch_names),
            'sfreq': sfreq,
            'duration_minutes': round(duration_min, 2)
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(f'{PATHS["dreams_raw"]}/dataset_summary.csv', index=False)
        return df_summary
