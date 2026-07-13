import os
import tempfile
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field

from src.data.readers import read_wfdb, LEADS, SAMPLE_RATE


@dataclass
class ECGSignal:
    signal: np.ndarray
    sample_rate: int
    lead_names: list[str] = field(default_factory=lambda: LEADS.copy())


class ECGParser:

    @staticmethod
    def from_wfdb(
        dat_bytes: bytes,
        hea_bytes: bytes | None,
        base_path: str = "tmp",
        base_name: str = "record",
    ) -> ECGSignal:
        with tempfile.TemporaryDirectory() as tmp:
            if base_path != "tmp":
                record_path = os.path.join(base_path, base_name)
            else:
                record_path = os.path.join(tmp, base_name)

            Path(record_path + ".dat").write_bytes(dat_bytes)

            if hea_bytes is not None:
                Path(record_path + ".hea").write_bytes(hea_bytes)

            signal = read_wfdb(record_path)

        return ECGSignal(signal=signal, sample_rate=SAMPLE_RATE)
