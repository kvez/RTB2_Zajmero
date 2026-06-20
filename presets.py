from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class MeasurementPreset:
    name: str
    description: str
    coupling: str             # "DCLimit" | "ACLimit"
    bandwidth: str            # "FULL" | "B20"
    adc_mode: str             # "SAMPle" | "HRESolution"
    acquisition_time_s: float
    record_length: int
    f_low_hz: float
    f_high_hz: float
    psd_segment_auto: bool
    psd_segment_s: Optional[float]   # None = auto
    fft_window: str                  # "hann" | "flattop" | "blackman"
    trigger_level_v: float
    default_scale_vdiv: float
    warning: str                     # "" = nincs figyelmeztetés


PRESETS: dict[str, MeasurementPreset] = {
    "1f_noise": MeasurementPreset(
        name="1/f Noise",
        description="Klasszikus 1/f zajmérés 0.1–10 Hz sávban, nagy gain erősítővel",
        coupling="ACLimit",
        bandwidth="B20",
        adc_mode="HRESolution",
        acquisition_time_s=300.0,
        record_length=500_000,
        f_low_hz=0.1,
        f_high_hz=10.0,
        psd_segment_auto=True,
        psd_segment_s=None,
        fft_window="hann",
        trigger_level_v=0.0,
        default_scale_vdiv=0.05,
        warning="",
    ),
    "psu_ripple": MeasurementPreset(
        name="PSU Standard Ripple",
        description="Kapcsolóüzemű tápegység ripple mérés 10 Hz–100 kHz",
        coupling="ACLimit",
        bandwidth="B20",
        adc_mode="SAMPle",
        acquisition_time_s=1.0,
        record_length=1_000_000,
        f_low_hz=10.0,
        f_high_hz=100_000.0,
        psd_segment_auto=True,
        psd_segment_s=None,
        fft_window="hann",
        trigger_level_v=0.0,
        default_scale_vdiv=0.05,
        warning="",
    ),
    "psu_wideband": MeasurementPreset(
        name="PSU Wideband",
        description="Szélessávú PSU zajmérés teljes sávszélességben (100 Hz–5 MHz)",
        coupling="ACLimit",
        bandwidth="FULL",
        adc_mode="SAMPle",
        acquisition_time_s=0.1,
        record_length=10_000_000,
        f_low_hz=100.0,
        f_high_hz=5_000_000.0,
        psd_segment_auto=True,
        psd_segment_s=None,
        fft_window="flattop",
        trigger_level_v=0.0,
        default_scale_vdiv=0.05,
        warning="10M minta letöltése ~5 s-t vehet igénybe LAN-on.",
    ),
    "burst_capture": MeasurementPreset(
        name="Burst Capture",
        description="Rövid tranziens / burst rögzítés teljes sávszélességben (10 ms)",
        coupling="DCLimit",
        bandwidth="FULL",
        adc_mode="SAMPle",
        acquisition_time_s=0.01,
        record_length=1_000_000,
        f_low_hz=1.0,
        f_high_hz=50_000_000.0,
        psd_segment_auto=True,
        psd_segment_s=None,
        fft_window="hann",
        trigger_level_v=0.0,
        default_scale_vdiv=0.1,
        warning="",
    ),
    "long_trend": MeasurementPreset(
        name="Long Trend",
        description="Hosszú drift / trend mérés 0.01–1 Hz sávban (1000 s)",
        coupling="ACLimit",
        bandwidth="B20",
        adc_mode="HRESolution",
        acquisition_time_s=1000.0,
        record_length=500_000,
        f_low_hz=0.01,
        f_high_hz=1.0,
        psd_segment_auto=True,
        psd_segment_s=None,
        fft_window="hann",
        trigger_level_v=0.0,
        default_scale_vdiv=0.02,
        warning="Hosszú mérés! ~17 perc. Ellenőrizd az összes kapcsolatot.",
    ),
    "custom": MeasurementPreset(
        name="Custom",
        description="Egyedi beállítások — az összes paraméter kézzel adható meg",
        coupling="ACLimit",
        bandwidth="B20",
        adc_mode="SAMPle",
        acquisition_time_s=300.0,
        record_length=500_000,
        f_low_hz=0.1,
        f_high_hz=10.0,
        psd_segment_auto=True,
        psd_segment_s=None,
        fft_window="hann",
        trigger_level_v=0.0,
        default_scale_vdiv=0.05,
        warning="",
    ),
}


def validate_preset(preset: MeasurementPreset) -> List[str]:
    """
    Visszaad egy listát a potenciális konfigurációs problémákról.
    Üres lista = nincs figyelmeztetés.
    """
    warns: List[str] = []
    fs = preset.record_length / preset.acquisition_time_s
    nyquist = fs / 2.0

    if preset.f_high_hz <= preset.f_low_hz:
        warns.append(f"f_high ({preset.f_high_hz} Hz) <= f_low ({preset.f_low_hz} Hz)")

    if preset.f_high_hz > nyquist:
        warns.append(
            f"f_high ({preset.f_high_hz:.0f} Hz) > Nyquist ({nyquist:.0f} Hz) "
            f"— aliasing lehetséges"
        )

    if preset.adc_mode == "HRESolution" and fs > 500_000:
        warns.append(
            f"HRESolution nem javasolt Fs > 500 kHz esetén "
            f"(aktuális: {fs/1e6:.1f} MSa/s)"
        )

    if preset.acquisition_time_s > 600:
        mins = preset.acquisition_time_s / 60
        warns.append(
            f"Hosszú mérés: {preset.acquisition_time_s:.0f} s "
            f"(~{mins:.0f} perc) — erősítsd meg az indítás előtt"
        )

    seg_s = (
        preset.psd_segment_s
        if not preset.psd_segment_auto or preset.psd_segment_s is not None
        else preset.acquisition_time_s / 10.0
    )
    delta_f = 1.0 / seg_s
    if preset.f_low_hz < delta_f:
        warns.append(
            f"f_low ({preset.f_low_hz} Hz) < Δf ({delta_f:.4f} Hz) "
            f"— frekvencia-felbontás nem elegendő"
        )

    return warns
