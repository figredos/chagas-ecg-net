import wfdb
import numpy as np

import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from scipy.signal import iirfilter, sosfiltfilt

SAMPLE_RATE = 100
N_LEADS = 12
SEQ_LEN = 734
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# --- Scenario constraints ---

SAMITROP_NATIVE_HZ = 400
PTBXL_NATIVE_HZ = 500

# shared intermediate rate before final downsample
COMMON_RATE_HZ = 500
# 100 Hz, final model input rate

TARGET_RATE_HZ = SAMPLE_RATE

# Single shared resampler instance
_shared_downsampler = Resample(
    orig_freq=COMMON_RATE_HZ,
    new_freq=TARGET_RATE_HZ,
    resampling_method="sinc_interp_hann",
)


def _shared_highpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """Elliptical high-pass filter for baseline-wander removal

    Args:
        signal (np.ndarray): numpy array of signal, with shape (n_leads, n_samples).
        fs (float): rate of signal.

    Returns:
        np.ndarray: Filtered signal.
    """

    sos = iirfilter(
        N=4,
        Wn=0.8,
        rs=40,
        rp=0.5,
        btype="highpass",
        ftype="ellip",
        fs=fs,
        output="sos",
    )
    filtered = sosfiltfilt(sos, signal, axis=-1)

    return np.ascontiguousarray(filtered)


def _to_common_rate(t: torch.Tensor, native_fs: int) -> torch.Tensor:
    """Upsamples if the source's native rate is below COMMON_RATE_HZ.

    No-ops if already at COMMON_RATE_HZ.

    Args:
        t (torch.Tensor): Signal tensor
        native_fs (int): Signal's native rate.

    Returns:
        torch.Tensor: t resampled to COMMON_RATE_HZ.
    """

    if native_fs == COMMON_RATE_HZ:
        return t
    resampler = Resample(
        orig_freq=native_fs,
        new_freq=COMMON_RATE_HZ,
        resampling_method="sinc_interp_hann",
    )
    return resampler(t)


def _normalize(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score normalise a single sample across all leads and time steps.

    z = (t - mean) / (std + epsilon)

    Args:
        t (torch.Tensor): Signal of shape (12, SEQ_LEN), to be normalized.
        eps (float, optional): Epsilon value. Defaults to 1e-6.

    Returns:
        torch.Tensor: Normalized signal with mean = 0 and std = 1
    """
    mean = t.mean()
    std = t.std()
    return (t - mean) / (std + eps)


def _pad_or_truncate(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is exactly (12, SEQ_LEN) via CENTRE-crop/pad.

    No-ops if signal's len is the same as SEQ_LEN.
    Signal is cropped if it is longer than SEQ_LEN/

    Args:
        t (torch.Tensor): Signal tensor

    Returns:
        torch.Tensor: Padded, and truncated signal.
    """

    cur_len = t.shape[-1]

    if cur_len == SEQ_LEN:
        return t

    if cur_len > SEQ_LEN:
        start = (cur_len - SEQ_LEN) // 2
        return t[:, start : start + SEQ_LEN]

    total_pad = SEQ_LEN - cur_len
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad

    return F.pad(t, (left_pad, right_pad))


def _equalized_preprocess(raw_signal: np.ndarray, native_fs: int) -> torch.Tensor:
    """Shared pipeline applied identically regardless of source.

    1. high-pass filter (baseline wander) at native rate
    2. resample to COMMON_RATE_HZ (upsample if needed)
    3. resample COMMON_RATE_HZ -> TARGET_RATE_HZ via the SAME resampler instance
    4. centre-crop/pad to SEQ_LEN
    5. per-sample z-score normalise

    Args:
        raw_signal (np.ndarray): numpy array of signal, with shape (n_leads, n_samples).
        native_fs (int): Signal's native rate.

    Returns:
        torch.Tensor: Pre-processed signal of shape (12, SEQ_LEN).
    """

    filtered = _shared_highpass_filter(raw_signal, fs=native_fs)
    t = torch.from_numpy(filtered).float().unsqueeze(0)

    # bring to COMMON_RATE_HZ
    t = _to_common_rate(t, native_fs)

    t = _shared_downsampler(t)

    t = t.squeeze(0)

    t = _pad_or_truncate(t)
    return _normalize(t)


def read_wfdb(record_path: str) -> np.ndarray:
    """Reads and preprocesses a WFDB record (.dat/.hea).

    Args:
        record_path (str): Path to record.

    Raises:
        ValueError: Records must be samples either at 400 Hz or 500 Hz.

    Returns:
        np.ndarray: Preprocessed signal from record.
    """
    signal, fields = wfdb.rdsamp(record_path)
    native_fs = int(fields["fs"])

    if native_fs == 100:
        raise ValueError(
            "Received a 100 Hz WFDB record. This pipeline preprocesses "
            "only files sampled at 400 Hz or 500 Hz."
        )

    assert signal is not None

    raw_leads_first = signal.T.astype(np.float32)  # (12, T)
    t = _equalized_preprocess(raw_leads_first, native_fs=native_fs)
    return t.numpy()  # (12, SEQ_LEN)
