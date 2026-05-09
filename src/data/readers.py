import h5py
import wfdb
import numpy as np

import torch
from torchaudio.transforms import Resample

SAMPLE_RATE = 100
N_LEADS = 12
SEQ_LEN = 734
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

_resampler_400_to_100 = Resample(
    orig_freq=400,
    new_freq=100,
    resampling_method="sinc_interp_hann",
)


def read_hdf5(path: str, exam_index: int) -> np.ndarray:
    with h5py.File(path, "r") as f:
        raw = np.array(f["tracings"][exam_index])  # type: ignore

    raw = raw[:2938, :]
    t = torch.from_numpy(raw).T.unsqueeze(0).float()
    t = _resampler_400_to_100(t)

    return t.squeeze(0).numpy()[:, :SEQ_LEN]


def read_wfdb(record_path: str) -> np.ndarray:
    signal, _ = wfdb.rdsamp(record_path)
    signal = signal[:SEQ_LEN, :]  # type: ignore

    return signal.T.astype(np.float32)
