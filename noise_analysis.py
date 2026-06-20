from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid


@dataclass
class NoiseResult:
    time: np.ndarray
    voltage_scope: np.ndarray
    voltage_filtered: np.ndarray
    rms_scope_v: float
    pp_scope_v: float
    rms_input_v: float
    pp_input_v: float
    integrated_noise_rms_v: float   # V RMS (NOT V/sqrt(Hz)!)
    freqs: np.ndarray
    psd_v2_per_hz: np.ndarray
    clipping_detected: bool
    mains_spurs_hz: List[float]
    metadata: object                # AcqMetadata instance
    crest_factor: float = 0.0
    drift_slope_v_s: float = 0.0
    robust_pp_v: float = 0.0
    parseval_error_pct: float = 0.0
    dominant_spurs: List[Tuple[float, float]] = field(default_factory=list)


def remove_dc_drift(v: np.ndarray, t: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(t, v, 1)
    return v - np.polyval(coeffs, t)


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v ** 2)))


def peak_to_peak(v: np.ndarray) -> float:
    return float(v.max() - v.min())


def compute_psd(
    v: np.ndarray,
    fs: float,
    window: str = "hann",
    segment_s: float = 50.0,
    overlap: float = 0.5,
    detrend: str = "linear",
) -> Tuple[np.ndarray, np.ndarray]:
    nperseg = min(int(segment_s * fs), len(v))
    # Minimum 3 szegmens a Welch-hez (statisztikai stabilitás)
    max_nperseg = max(256, len(v) // 3)
    nperseg = min(nperseg, max_nperseg)
    nperseg = max(nperseg, 256)
    noverlap = int(nperseg * overlap)
    freqs, psd = welch(
        v,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        scaling="density",
    )
    return freqs, psd


def integrated_noise(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    # Band-integrated RMS noise [V RMS] = sqrt(trapezoid(PSD, f))
    # NOTE: unit is V RMS, NOT V/sqrt(Hz)
    mask = (freqs >= f_low) & (freqs <= f_high)
    if mask.sum() < 2:
        return 0.0
    return float(np.sqrt(trapezoid(psd[mask], freqs[mask])))


def check_clipping(v: np.ndarray, scale_vdiv: float) -> bool:
    # RTB2: ±5 div vertical range (manual p.315)
    full_scale_half = scale_vdiv * 5.0
    return bool(np.abs(v).max() > 0.95 * full_scale_half)


def detect_mains_spurs(
    freqs: np.ndarray,
    psd: np.ndarray,
    mains_hz: float = 50.0,
) -> List[float]:
    # Only valid when fs >= 2 * mains_hz + margin
    # If freqs.max() < mains_hz * 1.1, 50 Hz is not detectable (aliased)
    delta_f = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    spurs = []
    for harmonic in [mains_hz, 2 * mains_hz, 3 * mains_hz]:
        if harmonic > freqs[-1] * 0.9:
            continue
        idx = int(np.argmin(np.abs(freqs - harmonic)))
        # Reject if nearest bin is more than 1.5 * Δf away from target
        # (coarse resolution: harmonic unresolvable) or is DC
        if freqs[idx] < 1.0 or abs(freqs[idx] - harmonic) > 1.5 * delta_f:
            continue
        # Prominent peak: >3x local median (±10 bins)
        lo = max(0, idx - 10)
        hi = min(len(psd), idx + 11)
        local_median = np.median(psd[lo:hi])
        if local_median > 0 and psd[idx] > 3.0 * local_median:
            spurs.append(float(freqs[idx]))
    # Deduplicate and sort
    return sorted(set(spurs))


def crest_factor(v: np.ndarray) -> float:
    """peak / RMS arány. Sine = sqrt(2) ≈ 1.414."""
    rms_val = rms(v)
    if rms_val == 0.0:
        return 0.0
    return float(np.max(np.abs(v)) / rms_val)


def robust_peak_to_peak(v: np.ndarray, percentile: float = 0.1) -> float:
    """p–p a percentile–(100-percentile) tartományban, spike-rezisztens."""
    lo = np.percentile(v, percentile)
    hi = np.percentile(v, 100.0 - percentile)
    return float(hi - lo)


def drift_slope(v: np.ndarray, t: np.ndarray) -> float:
    """Lineáris drift meredeksége [V/s]."""
    coeffs = np.polyfit(t, v, 1)
    return float(coeffs[0])


def parseval_check(
    v: np.ndarray,
    freqs: np.ndarray,
    psd: np.ndarray,
) -> float:
    """
    Parseval-ellenőrzés: (∫PSD df - RMS²) / RMS² * 100 [%].
    Pozitív = PSD felülbecsüli a zajt, negatív = alulbecsüli.
    """
    rms_sq = float(np.mean(v ** 2))
    if rms_sq == 0.0:
        return 0.0
    integrated = float(trapezoid(psd, freqs))
    return (integrated - rms_sq) / rms_sq * 100.0


def find_dominant_spurs(
    freqs: np.ndarray,
    psd: np.ndarray,
    n: int = 5,
    min_prominence_ratio: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Az n legerősebb kiemelkedő PSD csúcs visszaadása (freq_hz, psd_val) listaként.
    min_prominence_ratio: csúcs / helyi medián arány küszöbe.
    """
    results: List[Tuple[float, float]] = []
    if len(psd) < 22:
        return results
    for i in range(1, len(psd) - 1):
        lo = max(0, i - 10)
        hi = min(len(psd), i + 11)
        local_med = np.median(psd[lo:hi])
        if local_med > 0 and psd[i] > min_prominence_ratio * local_med:
            if psd[i] > psd[i - 1] and psd[i] >= psd[i + 1]:
                results.append((float(freqs[i]), float(psd[i])))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n]


def referred_to_input(value: float, gain: float) -> float:
    return value / gain


def noise_report(v: np.ndarray, t: np.ndarray, metadata) -> NoiseResult:
    v_filtered = remove_dc_drift(v, t)
    clipping = check_clipping(v, metadata.scale_vdiv)
    rms_scope = rms(v_filtered)
    pp_scope = peak_to_peak(v_filtered)

    freqs, psd = compute_psd(
        v_filtered,
        metadata.sample_rate_hz,
        window=metadata.psd_window,
        segment_s=metadata.psd_segment_s,
        overlap=metadata.psd_overlap,
        detrend=metadata.psd_detrend,
    )
    integ = integrated_noise(freqs, psd, metadata.f_low_hz, metadata.f_high_hz)
    spurs = detect_mains_spurs(freqs, psd)

    return NoiseResult(
        time=t,
        voltage_scope=v,
        voltage_filtered=v_filtered,
        rms_scope_v=rms_scope,
        pp_scope_v=pp_scope,
        rms_input_v=referred_to_input(rms_scope, metadata.gain),
        pp_input_v=referred_to_input(pp_scope, metadata.gain),
        integrated_noise_rms_v=referred_to_input(integ, metadata.gain),
        freqs=freqs,
        psd_v2_per_hz=psd,
        clipping_detected=clipping,
        mains_spurs_hz=spurs,
        metadata=metadata,
        crest_factor=crest_factor(v_filtered),
        drift_slope_v_s=drift_slope(v_filtered, t),
        robust_pp_v=robust_peak_to_peak(v_filtered),
        parseval_error_pct=parseval_check(v_filtered, freqs, psd),
        dominant_spurs=find_dominant_spurs(freqs, psd),
    )


def save_results(result: NoiseResult, output_dir: str = "results") -> tuple:
    import os
    import json
    import csv
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV: time series
    csv_path = os.path.join(output_dir, f"waveform_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "v_scope_V", "v_filtered_V"])
        for t, vs, vf in zip(
            result.time, result.voltage_scope, result.voltage_filtered
        ):
            writer.writerow([f"{t:.6f}", f"{vs:.8e}", f"{vf:.8e}"])

    # JSON: metadata + metrics
    meta = result.metadata
    report = {
        "timestamp": getattr(meta, "timestamp", ts),
        "idn": getattr(meta, "idn", ""),
        "channel": getattr(meta, "channel", 1),
        "scale_vdiv": getattr(meta, "scale_vdiv", 0),
        "offset_v": getattr(meta, "offset_v", 0),
        "coupling": getattr(meta, "coupling", ""),
        "bandwidth": getattr(meta, "bandwidth", ""),
        "sample_rate_hz": getattr(meta, "sample_rate_hz", 0),
        "record_length": getattr(meta, "record_length", 0),
        "acq_mode": getattr(meta, "acq_mode", ""),
        "gain": getattr(meta, "gain", 5000.0),
        "f_low_hz": getattr(meta, "f_low_hz", 0.1),
        "f_high_hz": getattr(meta, "f_high_hz", 10.0),
        "psd_window": getattr(meta, "psd_window", "hann"),
        "psd_segment_s": getattr(meta, "psd_segment_s", 50.0),
        "psd_overlap": getattr(meta, "psd_overlap", 0.5),
        "psd_detrend": getattr(meta, "psd_detrend", "linear"),
        "warmup_elapsed_s": getattr(meta, "warmup_elapsed_s", 0.0),
        "acquisition_index": getattr(meta, "acquisition_index", 0),
        "rms_scope_uV": result.rms_scope_v * 1e6,
        "pp_scope_uV": result.pp_scope_v * 1e6,
        "rms_input_nV": result.rms_input_v * 1e9,
        "pp_input_nV": result.pp_input_v * 1e9,
        "integrated_noise_nV_rms": result.integrated_noise_rms_v * 1e9,
        "clipping_detected": result.clipping_detected,
        "mains_spurs_hz": result.mains_spurs_hz,
    }
    json_path = os.path.join(output_dir, f"metadata_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    return csv_path, json_path
