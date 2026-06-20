from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

import numpy as np

from scope_driver import ScopeDriver


@dataclass
class AcqConfig:
    channel: int = 1
    scale_vdiv: float = 0.05          # V/div
    offset_v: float = 0.0
    coupling: str = "DCLimit"          # DCLimit | ACLimit  (manual p.316, NOT DC/AC!)
    bandwidth: str = "B20"             # FULL | B20  (manual p.316-317)
    acquisition_time_s: float = 300.0  # teljes tartomány: 12 ns – 6000 s (TIM:SCAL: 1 ns/div – 500 s/div)
    record_length: int = 500000        # predefined: 10k..20M (manual p.324-325)
    acq_type: str = "SAMPle"           # SAMPle (native ADC) | HRESolution (oversampling, jobb vertikális felbontás)
    trigger_level_v: float = 0.0


@dataclass
class AcqMetadata:
    idn: str
    channel: int
    scale_vdiv: float
    offset_v: float
    coupling: str
    bandwidth: str
    sample_rate_hz: float
    record_length: int
    acq_mode: str
    gain: float = 1.0
    f_low_hz: float = 0.1
    f_high_hz: float = 10.0
    timestamp: str = ""
    psd_window: str = "hann"
    psd_segment_s: float = 50.0
    psd_overlap: float = 0.5
    psd_detrend: str = "linear"
    warmup_elapsed_s: float = 0.0
    acquisition_index: int = 0


class WaveformAcquirer:
    def __init__(self, driver: ScopeDriver):
        self._d = driver
        self._config: Optional[AcqConfig] = None
        self._idn: str = ""

    def configure(self, config: AcqConfig, idn: str = "") -> None:
        self._config = config
        self._idn = idn
        ch = config.channel
        timebase = config.acquisition_time_s / 12.0

        cmds = [
            "CHAN1:AOFF",                            # all channels off (manual p.314)
            f"CHAN{ch}:STAT ON",                     # measurement channel on
            f"CHAN{ch}:PROB 1",
            f"CHAN{ch}:SCAL {config.scale_vdiv}",
            f"CHAN{ch}:OFFS {config.offset_v}",
            f"CHAN{ch}:COUP {config.coupling}",      # DCLimit or ACLimit
            f"CHAN{ch}:BAND {config.bandwidth}",
            f"TIM:SCAL {timebase:.6g}",              # s/div = acquisition_time / 12
            "ACQ:POIN:AUT OFF",
            f"ACQ:POIN {config.record_length}",
            f"CHAN:TYPE {config.acq_type}",
            "ACQ:TYPE REFresh",                      # REFresh = no averaging (manual p.325)
            "TRIG:A:MODE AUTO",
            "TRIG:A:TYPE EDGE",
            f"TRIG:A:SOUR CH{ch}",
            f"TRIG:A:LEV{ch} {config.trigger_level_v}",
            "TRIG:A:EDGE:SLOP POS",
        ]
        for cmd in cmds:
            self._d.write(cmd)

    def read_metadata(self) -> AcqMetadata:
        srate = float(self._d.query("ACQ:SRAT?"))
        header_str = self._d.query(
            f"CHAN{self._config.channel}:DATA:HEAD?"
        )
        parts = header_str.strip().split(",")
        n_samples = int(parts[2])

        return AcqMetadata(
            idn=self._idn,
            channel=self._config.channel,
            scale_vdiv=self._config.scale_vdiv,
            offset_v=self._config.offset_v,
            coupling=self._config.coupling,
            bandwidth=self._config.bandwidth,
            sample_rate_hz=srate,
            record_length=n_samples,
            acq_mode=self._config.acq_type,
            timestamp=datetime.now().isoformat(),
        )

    def acquire_single(
        self, timeout_s: float = 720.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Extend timeout for long acquisitions
        self._d.set_timeout(int((timeout_s + 30) * 1000))

        # SING;*OPC? blocks until acquisition complete (manual p.312, p.582)
        self._d.query("SING;*OPC?")

        self._d.set_timeout(120000)
        self._d.write("STOP")
        self._d.write(
            f"CHAN{self._config.channel}:DATA:POIN MAX"   # STOP required first! (manual p.435)
        )

        # MSBF + REAL,32 must be used together, dtype='>f4' (manual p.431-433)
        self._d.write("FORM REAL,32")
        self._d.write("FORM:BORD MSBF")

        size_bytes = self._config.record_length * 4
        download_timeout_ms = max(120_000, int(size_bytes / 1_000_000 * 5_000))
        self._d.set_timeout(download_timeout_ms)
        self._d.write(f"CHAN{self._config.channel}:DATA?")
        raw = self._d.read_ieee_block()

        voltages = self._parse_binary_real32(raw)

        # Time axis from ACQ:SRAT?
        fs = float(self._d.query("ACQ:SRAT?"))
        time_arr = np.arange(len(voltages)) / fs

        # Sanity check: actual duration vs requested acquisition time
        actual_duration = len(voltages) / fs
        requested = self._config.acquisition_time_s
        if requested > 0 and abs(actual_duration - requested) / requested > 0.1:
            import warnings
            warnings.warn(
                f"Időtengely eltérés: tényleges {actual_duration:.4g} s "
                f"vs kért {requested:.4g} s "
                f"(actual_fs={fs:.6g} Sa/s, n={len(voltages)})",
                stacklevel=2,
            )

        return time_arr, voltages

    @staticmethod
    def _parse_binary_real32(raw: bytes) -> np.ndarray:
        # IEEE 488.2 block header: #<N><L><data>
        # e.g. #520000<data_bytes> => 5 digits, 20000 bytes => 5000 floats
        assert raw[0:1] == b"#", f"Invalid IEEE block header: {raw[:4]}"
        n_digits = int(chr(raw[1]))
        n_bytes = int(raw[2 : 2 + n_digits])
        data_start = 2 + n_digits
        data = raw[data_start : data_start + n_bytes]
        # MSBF = big endian = '>f4'
        return np.frombuffer(data, dtype=">f4").copy()
