import numpy as np

from src.data.readers import LEADS
from src.inference.parsers import ECGParser


def test_ecg_wfdb_parser(tmp_path):
    record_path = "tests/fixtures/"

    with open(record_path + "sample.dat", "rb") as dat:
        dat_bytes = dat.read()

    with open(record_path + "sample.hea", "rb") as hea:
        hea_bytes = hea.read()

    ecg_signal = ECGParser.from_wfdb(
        dat_bytes=dat_bytes,
        hea_bytes=hea_bytes,
        base_name="sample",
        base_path=tmp_path,
    )

    assert ecg_signal.sample_rate in [100, 400, 500]

    assert ecg_signal.lead_names == LEADS

    assert ecg_signal.signal.dtype == np.float32
