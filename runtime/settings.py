from pathlib import Path
import os
import torch

APP_ROOT = Path(__file__).resolve().parents[1]


def model_path() -> Path:
    return Path(os.environ.get('DENTILIN_MODEL_DIR', str(APP_ROOT / 'models'))) / 'fdi.pt'


def device() -> torch.device:
    choice = os.environ.get('DENTILIN_DEVICE', 'auto').lower()
    if choice not in {'auto', 'cpu', 'cuda'}:
        raise ValueError('DENTILIN_DEVICE debe ser auto, cpu o cuda.')
    if choice == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA no está disponible; utiliza DENTILIN_DEVICE=cpu.')
    torch.set_num_threads(max(1, int(os.environ.get('DENTILIN_NUM_THREADS', '4'))))
    return torch.device('cuda' if choice != 'cpu' and torch.cuda.is_available() else 'cpu')
