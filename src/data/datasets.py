import torch
import torch.nn.functional as F

from src.utils.helpers import extract_cd2_coeffs, window_ecg


class RawECGDataset(torch.utils.data.Dataset):
    """
    Dataset for raw ECG signals without spectrograms

    Args:
        data: ECG signals.
        labels: Class labels of shape (N,)
    """

    def __init__(self, data, labels):
        """ """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ecg = self.data[index]

        ecg = ecg.T.to(torch.float32)

        return ecg, self.labels[index]


class STRawECGDataset(torch.utils.data.Dataset):
    """
    Dataset for raw ECG signals without spectrograms

    Args:
        data: ECG signals.
        labels: Class labels of shape (N,)
    """

    def __init__(self, data, labels):
        """ """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        ecg = self.data[index]
        ecg = ecg.T

        target_height = 32
        target_width = 768

        current_height, current_width = ecg.shape

        pad_top = (target_height - current_height) // 2
        pad_bottom = target_height - current_height - pad_top
        pad_left = (target_width - current_width) // 2
        pad_right = target_width - current_width - pad_left

        # Apply padding
        ecg = F.pad(
            ecg, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0
        )

        # Add channel dimension: (1, 32, 768)
        ecg = ecg.unsqueeze(0)

        return ecg, self.labels[index]


class CD2ECGDataset(torch.utils.data.Dataset):
    """CD_2 ECG Dataset extracts the second level of data from the FWT Decomposition.

    Args:
        data (torch.Tensor): The tensor containing the ECG signal data.
        labels (torch.Tensor): The tensor containing the labels of the ECG data.
        wavelet (str): The wavelet to apply to the ECG data.
        level (int): Which level of the decomposition to use.
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        wavelet: str | bool = "db3",
        level: int = 3,
        window_augment: bool = False,
        window_size: int = 100,
        stride: int = 100,
        filter: float | None = None,
    ) -> None:
        """Dataset extracts the second level of data from the FWT Decomposition

        Args:
            data (torch.Tensor): Tensor containing the ECG signal data.
            labels (torch.Tensor): Tensor with ECG labels.
            wavelet (str | bool, optional): Wavelet to apply to data. Defaults to "db3".
            level (int, optional): Level of data decomposition. Defaults to 3.
            window_augment (bool, optional): Whether the data should be split into windows. Defaults to False.
            window_size (int, optional): Size of windows for ECGs. Defaults to 100.
            stride (int, optional): Stride of data splitting. Defaults to 100.
            filter (float | None, optional): Amplitude filter for ECGs. Defaults to None.
        """
        super().__init__()
        if window_augment:
            data, labels = window_ecg(
                ecg_data=data,
                labels=labels,
                window_size=window_size,
                stride=stride,
                filter=filter,
            )

        if isinstance(wavelet, str):
            data = extract_cd2_coeffs(ecg_batch=data, wavelet=wavelet, level=level)

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
