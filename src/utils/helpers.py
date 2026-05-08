"""
Helper functions
"""

import pywt
import ptwt
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_12_lead_ecg(
    ecg_tensor: torch.Tensor | np.ndarray,
    sample_rate: int = 360,
    title: str = "12-Lead ECG",
) -> None:
    """
    Plot a 12-lead ECG signal in standard medical format.

    Args:
        ecg_tensor (torch.Tensor or np.ndarray): Shape (length, 12) - ECG data for 12 leads.
        sample_rate (int): Sampling rate in Hz (default 360 for MIT-BIH).
        title (str): Plot title.
    """
    if isinstance(ecg_tensor, torch.Tensor):
        if hasattr(ecg_tensor, 'cpu'):
            ecg_data = ecg_tensor.cpu().numpy()
        elif hasattr(ecg_tensor, 'numpy'):
            ecg_data = ecg_tensor.numpy()
        else:
            ecg_data = np.array(ecg_tensor)
    else:
        ecg_data = ecg_tensor

    length, num_leads = ecg_data.shape
    if num_leads != 12:
        raise ValueError(f"Expected 12 leads, got {num_leads}")

    time = np.arange(length) / sample_rate

    lead_names = [
        'I',
        'II',
        'III',
        'aVR',
        'aVL',
        'aVF',
        'V1',
        'V2',
        'V3',
        'V4',
        'V5',
        'V6',
    ]

    fig, axes = plt.subplots(12, 1, figsize=(15, 20), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i in range(12):
        axes[i].plot(time, ecg_data[:, i], color='black', linewidth=0.8)
        axes[i].set_ylabel(lead_names[i], fontweight='bold', fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].set_xlim(0, time[-1])

        axes[i].set_facecolor('#f8f8f8')

    axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


def extract_cd2_coeffs(
    ecg_batch: torch.Tensor,
    wavelet: str = "db3",
    level: int = 3,
    mode: str = "zero",
) -> torch.Tensor:
    """Extracts CD_2 coefficients from FWT Decomposition.

    Args:
        ecg_batch(torch.Tensor): Batch of Tensors to apply FWT Decomposition
        wavelet(str): Type of wavelet to apply in FWT Decomposition.
        level(int): Number of levels to apply FWT Decomposition.
        mode(str): Padding mode for FWT Decomposition.

    Returns:
        `torch.Tensor`: A tensor with the second level detail obtained from the FWT Decomposition.
    """
    batch_size, seq_len, num_leads = ecg_batch.shape

    # Process each lead separately to maintain clear sequence structure
    cd2_coeffs = []

    for lead_idx in range(num_leads):
        lead_data = ecg_batch[:, :, lead_idx]  # Shape: (batch_size, seq_len)

        vectorized_wavedec = torch.vmap(
            lambda x: ptwt.wavedec(  # type: ignore
                x,
                wavelet=wavelet,
                level=level,
                mode=mode,  # type: ignore
            )[-2]
        )
        lead_cd2 = vectorized_wavedec(lead_data)
        cd2_coeffs.append(lead_cd2)

    # Stack to maintain (batch, time, leads) structure
    result = torch.stack(cd2_coeffs, dim=2)
    return result


def window_ecg(
    ecg_data: torch.Tensor,
    labels: torch.Tensor,
    window_size: int = 100,
    stride: int = 100,
    filter: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Window ECG data

    Args:
        ecg_data: Data tensor with ECG signals.
        labels: Chagas diagnosis for the ECG signals tensor.
        window_size: Window size for sampling.
        stride: Stride of window sampling.
        filter: Value to filter samples.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple with the windowed ECGs and its labels.
    """
    windowed_data = []
    windowed_labels = []

    for i in range(len(ecg_data)):
        ecg = ecg_data[i]
        label = labels[i]

        num_windows = (ecg.shape[0] - window_size) // stride + 1
        for w in range(num_windows):
            start_idx = w * stride
            end_idx = start_idx + window_size
            window = ecg[start_idx:end_idx]

            window_sum = abs(window.sum().item())

            if filter is not None and window_sum <= filter:
                continue

            windowed_data.append(window)
            windowed_labels.append(label)

    return torch.stack(windowed_data), torch.tensor(windowed_labels)


def create_ecg_spectrogram(ecg: torch.Tensor) -> torch.Tensor:
    """Creates spectrograms from a single ECG using wavelet transform.

    Args:
        ecg: Single ECG tensor of shape (time, 12)

    Returns:
        torch.Tensor: Spectrogram of shape (12, 64, time) = 12 channels
    """
    spectrograms = []

    for lead_idx in range(12):
        lead_data = ecg[:, lead_idx].numpy()
        scales = pywt.scale2frequency("cmor1.5-1.0", np.arange(1, 65)) * 100
        coeffs, _ = pywt.cwt(lead_data, scales, "cmor1.5-1.0")
        spectrograms.append(torch.from_numpy(np.abs(coeffs)))

    # MODIFIED: Return (12, 64, time) - NO batch dim, NO flatten
    return torch.stack(spectrograms, dim=0).to(torch.float32)
