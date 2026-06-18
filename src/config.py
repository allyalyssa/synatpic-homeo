import os
import torch

if os.path.exists('/content/synatpic-homeo'):
    BASE = '/content/synatpic-homeo/data'
else:
    BASE = './data'

PATHS = {
    'dreams_raw': f'{BASE}/dreams',
    'preprocessed': f'{BASE}/preprocessed',
    'slow_waves': f'{BASE}/slow_waves',
    'curves': f'{BASE}/curves',
    'models': f'{BASE}/models',
    'outputs': f'{BASE}/outputs',
}

for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SFREQ = 256
EPOCH_SEC = 30
SAMPLES_PER_EPOCH = int(EPOCH_SEC * SFREQ)
ARTIFACT_THRESHOLD = 150e-6
