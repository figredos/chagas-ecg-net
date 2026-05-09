import os
import tempfile
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from src.data.readers import read_hdf5, read_wfdb, LEADS, SAMPLE_RATE


@dataclass
class ECGSignal:
    signal: np.ndarray
    sample_rate: int
    lead_names: list[str] = LEADS


class ECGParser:
    def from_hdf5(self, file_bytes: bytes) -> ECGSignal:
        with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            signal = read_hdf5(tmp.name, exam_index=0)

        return ECGSignal(signal=signal, sample_rate=SAMPLE_RATE)

    def from_wfdb(self, dat_bytes: bytes, hea_bytes: bytes) -> ECGSignal:
        with tempfile.TemporaryDirectory() as tmp:
            record_path = os.path.join(tmp, "record")
            Path(record_path + ".dat").write_bytes(dat_bytes)
            Path(record_path + ".hea").write_bytes(hea_bytes)
            signal = read_wfdb(record_path)

        return ECGSignal(signal=signal, sample_rate=SAMPLE_RATE)
