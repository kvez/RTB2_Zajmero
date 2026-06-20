import sys
import time
from typing import Optional

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication

from scope_driver import ScopeDriver
from waveform_acquire import AcqConfig, AcqMetadata, WaveformAcquirer
from noise_analysis import NoiseResult, noise_report, save_results
from presets import PRESETS, validate_preset


class MeasurementWorker(QThread):
    status_changed = pyqtSignal(str)
    progress = pyqtSignal(int)
    data_ready = pyqtSignal(object)   # NoiseResult
    error = pyqtSignal(str)

    def __init__(
        self,
        ip: str,
        config: AcqConfig,
        gain: float = 5000.0,
        f_low: float = 0.1,
        f_high: float = 10.0,
        psd_window: str = "hann",
        psd_segment_s: float = 50.0,
        psd_overlap: float = 0.5,
        output_dir: str = "results",
        acquisition_index: int = 0,
        session_start_time: Optional[float] = None,
    ):
        super().__init__()
        self._ip = ip
        self._config = config
        self._gain = gain
        self._f_low = f_low
        self._f_high = f_high
        self._psd_window = psd_window
        self._psd_segment_s = psd_segment_s
        self._psd_overlap = psd_overlap
        self._output_dir = output_dir
        self._acquisition_index = acquisition_index
        self._session_start = session_start_time or time.time()
        self._driver: Optional[ScopeDriver] = None
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True
        if self._driver is not None:
            self._driver.force_close()

    def run(self) -> None:
        self._driver = ScopeDriver()
        try:
            self.status_changed.emit("Csatlakozas...")
            self._driver.connect(self._ip)
            self.progress.emit(5)

            idn = self._driver.identify()
            self.status_changed.emit(
                f"Kapcsolodva: {idn.split(',')[1] if ',' in idn else idn}"
            )
            self._driver.reset()
            errors = self._driver.check_errors()
            if errors:
                self.error.emit(f"Inicializalasi hiba: {errors[0]}")
                return
            self.progress.emit(10)

            acquirer = WaveformAcquirer(self._driver)
            acquirer.configure(self._config, idn=idn)
            self.status_changed.emit("Scope konfigurálva")
            self.progress.emit(20)

            t_s = self._config.acquisition_time_s
            self.status_changed.emit(f"Meres folyamatban... ({t_s:.0f} s)")
            time_arr, voltage = acquirer.acquire_single(timeout_s=t_s + 120)
            self.progress.emit(70)

            meta = acquirer.read_metadata()
            meta.gain = self._gain
            meta.f_low_hz = self._f_low
            meta.f_high_hz = self._f_high
            meta.psd_window = self._psd_window
            meta.psd_segment_s = self._psd_segment_s
            meta.psd_overlap = self._psd_overlap
            meta.warmup_elapsed_s = time.time() - self._session_start
            meta.acquisition_index = self._acquisition_index

            self.status_changed.emit("Adatelemzes...")
            result = noise_report(voltage, time_arr, meta)
            self.progress.emit(90)

            if result.clipping_detected:
                self.status_changed.emit(
                    "FIGYELMEZTES: Clipping detektálva! A meres érvénytelen."
                )

            csv_p, json_p = save_results(result, self._output_dir)
            self.progress.emit(100)
            self.status_changed.emit(f"Meres kesz. Mentve: {csv_p}")
            self.data_ready.emit(result)

        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))
        finally:
            if self._driver is not None:
                self._driver.disconnect()
                self._driver = None


import os
from datetime import datetime

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QDoubleSpinBox,
    QGroupBox, QStatusBar, QProgressBar, QCheckBox,
    QMessageBox, QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class MainWindow(QMainWindow):

    DC_WARNING = (
        "BIZTONSAGI FIGYELMEZTES:\n\n"
        "A 10 V referencia DC szintjet NEM szabad kozvetlenul\n"
        "nagy erositesu zajerositorere kotni, ha nincs DC levalasztas\n"
        "vagy offset kompenzacio!\n\n"
        "Ellenorizd a meresi lancot mielott elindited a merést."
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTB2 Zajmero v2.0")
        self.resize(1200, 800)
        self._worker: Optional[MeasurementWorker] = None
        self._session_start = time.time()
        self._acquisition_index = 0
        self._last_result: Optional[NoiseResult] = None
        self._acq_timer = QTimer(self)
        self._acq_timer.timeout.connect(self._on_acq_tick)
        self._acq_time_s = 300.0
        self._acq_t0 = 0.0

        self._build_ui()
        self._update_fs_display()
        self._update_segment_label()
        self._show_dc_warning()

    # ── Helper statikus metódusok ──────────────────────────────────────

    @staticmethod
    def _parse_acqtime(text: str) -> Optional[float]:
        """
        "300" → 300.0 s, "100ms" → 0.1 s, "12ns" → 12e-9 s.
        None ha érvénytelen.
        """
        text = text.strip().lower().replace(" ", "")
        suffixes = [("ns", 1e-9), ("µs", 1e-6), ("us", 1e-6),
                    ("ms", 1e-3), ("s", 1.0)]
        for suf, mult in suffixes:
            if text.endswith(suf):
                num = text[: -len(suf)]
                try:
                    return float(num) * mult
                except ValueError:
                    return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _fmt_acqtime(s: float) -> str:
        if s < 1e-6:
            return f"{s * 1e9:.4g} ns"
        if s < 1e-3:
            return f"{s * 1e6:.4g} µs"
        if s < 1.0:
            return f"{s * 1e3:.4g} ms"
        return f"{s:.6g} s"

    @staticmethod
    def _fmt_voltage(v: float) -> str:
        abs_v = abs(v)
        if abs_v == 0.0:
            return "0 V"
        if abs_v >= 1.0:
            return f"{v:.3f} V"
        if abs_v >= 1e-3:
            return f"{v*1e3:.3f} mV"
        if abs_v >= 1e-6:
            return f"{v*1e6:.3f} µV"
        if abs_v >= 1e-9:
            return f"{v*1e9:.3f} nV"
        if abs_v >= 1e-12:
            return f"{v*1e12:.3f} pV"
        return f"{v:.3e} V"

    @staticmethod
    def _fmt_hz(hz: float) -> str:
        if hz >= 1e6:
            return f"{hz/1e6:.4g} MHz"
        if hz >= 1e3:
            return f"{hz/1e3:.4g} kHz"
        return f"{hz:.4g} Hz"

    @staticmethod
    def _format_reclen(n: int) -> str:
        if n >= 1_000_000:
            return f"{n // 1_000_000} MSa"
        if n >= 1_000:
            return f"{n // 1_000} kSa"
        return str(n)

    # ── UI build ──────────────────────────────────────────────────────

    def _on_browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Mentési mappa", self._outdir_edit.text())
        if d:
            self._outdir_edit.setText(d)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(5000)
        event.accept()

    def _show_dc_warning(self):
        QMessageBox.warning(self, "Biztonsagi Figyelmezetes", self.DC_WARNING)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)

        left_layout.addWidget(self._build_connect_group())
        left_layout.addWidget(self._build_preset_group())
        left_layout.addWidget(self._build_config_group())
        left_layout.addWidget(self._build_analysis_group())
        left_layout.addWidget(self._build_meas_mode_group())
        left_layout.addWidget(self._build_result_group())
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self._fig = Figure(figsize=(8, 6), tight_layout=True)
        self._ax_time = self._fig.add_subplot(211)
        self._ax_psd = self._fig.add_subplot(212)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        right_layout.addWidget(self._toolbar)
        right_layout.addWidget(self._canvas)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._status_bar.addPermanentWidget(self._progress)
        self._set_status("Kesz.")

    def _build_connect_group(self) -> QGroupBox:
        grp = QGroupBox("Kapcsolat")
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("IP:"))
        self._ip_edit = QLineEdit("192.168.2.82")
        row1.addWidget(self._ip_edit)
        self._btn_start = QPushButton("Meres inditas")
        self._btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )
        self._btn_start.clicked.connect(self._on_start)
        row1.addWidget(self._btn_start)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        row1.addWidget(self._btn_stop)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Mappa:"))
        default_dir = os.path.join(os.path.expanduser("~"), "Desktop", "RTB_merések")
        self._outdir_edit = QLineEdit(default_dir)
        row2.addWidget(self._outdir_edit)
        btn_browse = QPushButton("...")
        btn_browse.setMaximumWidth(30)
        btn_browse.clicked.connect(self._on_browse_dir)
        row2.addWidget(btn_browse)
        lay.addLayout(row2)

        return grp

    def _build_preset_group(self) -> QGroupBox:
        grp = QGroupBox("Mérési sablon")
        lay = QVBoxLayout(grp)

        preset_row = QHBoxLayout()
        self._preset_combo = QComboBox()
        for key, p in PRESETS.items():
            self._preset_combo.addItem(p.name, userData=key)
        preset_row.addWidget(self._preset_combo, stretch=1)
        btn_apply = QPushButton("Alkalmaz")
        btn_apply.clicked.connect(self._on_apply_preset)
        preset_row.addWidget(btn_apply)
        lay.addLayout(preset_row)

        self._preset_warning_label = QLabel("")
        self._preset_warning_label.setWordWrap(True)
        self._preset_warning_label.setStyleSheet("color: #cc6600;")
        lay.addWidget(self._preset_warning_label)

        return grp

    def _build_config_group(self) -> QGroupBox:
        grp = QGroupBox("Scope beállítások")
        lay = QVBoxLayout(grp)

        def row(label, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            lay.addLayout(h)

        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.001, 5.0)
        self._scale_spin.setValue(0.05)
        self._scale_spin.setSuffix(" V/div")
        self._scale_spin.setDecimals(3)
        row("Scope scale:", self._scale_spin)

        self._coupling_combo = QComboBox()
        self._coupling_combo.addItems(["DCLimit", "ACLimit"])
        row("Coupling:", self._coupling_combo)

        self._bandwidth_combo = QComboBox()
        self._bandwidth_combo.addItems(["B20", "FULL"])
        row("Bandwidth:", self._bandwidth_combo)

        acqtime_presets = [
            "12ns", "100ns", "1µs", "10µs", "100µs",
            "1ms", "10ms", "100ms",
            "1s", "10s", "60s", "300s", "600s", "1000s", "6000s",
        ]
        self._acqtime_combo = QComboBox()
        self._acqtime_combo.setEditable(True)
        self._acqtime_combo.addItems(acqtime_presets)
        self._acqtime_combo.setCurrentText("300s")
        self._acqtime_combo.currentTextChanged.connect(self._update_fs_display)
        self._acqtime_combo.currentTextChanged.connect(self._update_segment_label)
        self._acqtime_combo.lineEdit().editingFinished.connect(self._update_fs_display)
        self._acqtime_combo.lineEdit().editingFinished.connect(self._update_segment_label)
        row("Acq. time:", self._acqtime_combo)

        reclen_values = [
            10_000, 20_000, 50_000, 100_000, 200_000, 500_000,
            1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000
        ]
        self._reclen_combo = QComboBox()
        for v in reclen_values:
            self._reclen_combo.addItem(self._format_reclen(v), userData=v)
        self._reclen_combo.setCurrentIndex(5)   # 500 kSa default
        self._reclen_combo.currentIndexChanged.connect(self._update_fs_display)
        row("Record length:", self._reclen_combo)

        self._acqtype_combo = QComboBox()
        self._acqtype_combo.addItems(["SAMPle", "HRESolution"])
        row("ADC mód:", self._acqtype_combo)

        self._gain_spin = QDoubleSpinBox()
        self._gain_spin.setRange(1.0, 1e7)
        self._gain_spin.setValue(5000.0)
        self._gain_spin.setDecimals(1)
        row("Erősítő gain:", self._gain_spin)

        # Fs info panel
        self._fs_panel = QLabel()
        self._fs_panel.setStyleSheet(
            "background: #f0f0f0; border-radius: 4px; padding: 4px; font-family: monospace;"
        )
        self._fs_panel.setWordWrap(False)
        lay.addWidget(self._fs_panel)

        return grp

    def _build_analysis_group(self) -> QGroupBox:
        grp = QGroupBox("Zajanalízis")
        lay = QVBoxLayout(grp)

        def row(label, widget):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            h.addWidget(widget)
            lay.addLayout(h)

        self._flow_spin = QDoubleSpinBox()
        self._flow_spin.setRange(0.001, 1e7)
        self._flow_spin.setValue(0.1)
        self._flow_spin.setDecimals(3)
        self._flow_spin.setSuffix(" Hz")
        row("f_low:", self._flow_spin)

        self._fhigh_spin = QDoubleSpinBox()
        self._fhigh_spin.setRange(0.001, 1e8)
        self._fhigh_spin.setValue(10.0)
        self._fhigh_spin.setDecimals(3)
        self._fhigh_spin.setSuffix(" Hz")
        row("f_high:", self._fhigh_spin)

        seg_row = QHBoxLayout()
        self._segment_auto_check = QCheckBox("Auto")
        self._segment_auto_check.setChecked(True)
        self._segment_auto_label = QLabel("(30.0 s)")
        self._segment_auto_check.stateChanged.connect(self._on_segment_auto_toggle)
        seg_row.addWidget(QLabel("PSD szegmens:"))
        seg_row.addWidget(self._segment_auto_check)
        seg_row.addWidget(self._segment_auto_label)
        lay.addLayout(seg_row)

        self._segment_spin = QDoubleSpinBox()
        self._segment_spin.setRange(0.01, 10000.0)
        self._segment_spin.setValue(50.0)
        self._segment_spin.setSuffix(" s")
        self._segment_spin.setEnabled(False)
        row("Szegmens:", self._segment_spin)

        self._psd_window_combo = QComboBox()
        self._psd_window_combo.addItems(["hann", "flattop", "blackman"])
        row("PSD ablak:", self._psd_window_combo)

        return grp

    def _build_meas_mode_group(self) -> QGroupBox:
        grp = QGroupBox("Meresi mod")
        lay = QVBoxLayout(grp)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "DUT meres (teljes lanc)",
            "Erosito zaj (DUT nelkul)",
            "Input shorted (scope csak)",
        ])
        lay.addWidget(self._mode_combo)
        return grp

    def _build_result_group(self) -> QGroupBox:
        grp = QGroupBox("Eredmenyek")
        lay = QVBoxLayout(grp)

        def result_row(label):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            lbl = QLabel("—")
            lbl.setFont(QFont("Courier", 10))
            h.addWidget(lbl)
            lay.addLayout(h)
            return lbl

        self._lbl_rms_scope = result_row("Scope RMS:")
        self._lbl_pp_scope = result_row("Scope p-p:")
        self._lbl_rms_input = result_row("Input RMS:")
        self._lbl_pp_input = result_row("Input p-p:")

        # Dinamikus integrated noise label
        integ_row = QHBoxLayout()
        self._lbl_integ_label = QLabel("0.1–10 Hz RMS:")
        self._lbl_integ = QLabel("—")
        self._lbl_integ.setFont(QFont("Courier", 10))
        integ_row.addWidget(self._lbl_integ_label)
        integ_row.addWidget(self._lbl_integ)
        lay.addLayout(integ_row)

        self._lbl_clipping = result_row("Clipping:")
        self._lbl_spurs = result_row("Mains spurs:")
        self._lbl_parseval = result_row("Parseval err:")
        return grp

    # ── Preset logika ─────────────────────────────────────────────────

    def _on_apply_preset(self):
        key = self._preset_combo.currentData()
        p = PRESETS[key]

        self._coupling_combo.setCurrentText(p.coupling)
        self._bandwidth_combo.setCurrentText(p.bandwidth)
        self._acqtype_combo.setCurrentText(p.adc_mode)
        self._acqtime_combo.setCurrentText(self._fmt_acqtime(p.acquisition_time_s))
        for i in range(self._reclen_combo.count()):
            if self._reclen_combo.itemData(i) == p.record_length:
                self._reclen_combo.setCurrentIndex(i)
                break
        self._scale_spin.setValue(p.default_scale_vdiv)
        self._flow_spin.setValue(p.f_low_hz)
        self._fhigh_spin.setValue(p.f_high_hz)
        self._psd_window_combo.setCurrentText(p.fft_window)
        if p.psd_segment_auto:
            self._segment_auto_check.setChecked(True)
        else:
            self._segment_auto_check.setChecked(False)
            if p.psd_segment_s is not None:
                self._segment_spin.setValue(p.psd_segment_s)

        warns = validate_preset(p)
        warn_text = "\n".join(warns)
        if p.warning:
            warn_text = p.warning + ("\n" + warn_text if warns else "")
        self._preset_warning_label.setText(warn_text)
        self._update_fs_display()
        self._update_segment_label()

    # ── Fs panel frissítés ────────────────────────────────────────────

    def _update_fs_display(self):
        acq_s = self._parse_acqtime(self._acqtime_combo.currentText())
        if acq_s is None or acq_s <= 0:
            self._fs_panel.setText("⚠ Érvénytelen acquisition time")
            return
        reclen = self._reclen_combo.currentData()
        if reclen is None:
            return
        fs = reclen / acq_s
        nyquist = fs / 2.0
        size_mb = reclen * 4 / 1_048_576
        dl_s = size_mb / 8.0
        timebase = acq_s / 12.0
        self._fs_panel.setText(
            f"Mintavétel:  {fs:>12,.0f} Sa/s\n"
            f"Nyquist:     {nyquist:>12,.0f} Hz\n"
            f"Letöltés:    ~{size_mb:.1f} MB  /  ~{dl_s:.1f} s\n"
            f"TIM:SCAL:    {timebase:.6g} s/div"
        )

    def _update_segment_label(self):
        acq_s = self._parse_acqtime(self._acqtime_combo.currentText())
        if acq_s and acq_s > 0:
            seg = acq_s / 10.0
            self._segment_auto_label.setText(f"({self._fmt_acqtime(seg)})")

    def _on_segment_auto_toggle(self, state):
        self._segment_spin.setEnabled(not bool(state))

    # ── Mérés indítás ─────────────────────────────────────────────────

    def _on_start(self):
        acq_s = self._parse_acqtime(self._acqtime_combo.currentText())
        if acq_s is None or acq_s <= 0:
            QMessageBox.warning(self, "Hibás érték",
                                "Érvénytelen acquisition time érték.")
            return

        if acq_s > 600:
            mins = acq_s / 60.0
            size_mb = self._reclen_combo.currentData() * 4 / 1_048_576
            reply = QMessageBox.question(
                self, "Hosszú mérés megerősítése",
                f"Az acquisition time {acq_s:.0f} s (~{mins:.1f} perc).\n"
                f"Letöltési méret: ~{size_mb:.0f} MB\n\n"
                "Biztosan indítod a mérést?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        f_low = self._flow_spin.value()
        f_high = self._fhigh_spin.value()

        self._lbl_integ_label.setText(
            f"{self._fmt_hz(f_low)}–{self._fmt_hz(f_high)} RMS:"
        )

        if self._segment_auto_check.isChecked():
            psd_segment_s = acq_s / 10.0
        else:
            psd_segment_s = self._segment_spin.value()

        config = AcqConfig(
            channel=1,
            scale_vdiv=self._scale_spin.value(),
            coupling=self._coupling_combo.currentText(),
            bandwidth=self._bandwidth_combo.currentText(),
            acquisition_time_s=acq_s,
            record_length=self._reclen_combo.currentData(),
            acq_type=self._acqtype_combo.currentText(),
        )
        self._worker = MeasurementWorker(
            ip=self._ip_edit.text().strip(),
            config=config,
            gain=self._gain_spin.value(),
            f_low=f_low,
            f_high=f_high,
            psd_window=self._psd_window_combo.currentText(),
            psd_segment_s=psd_segment_s,
            output_dir=self._outdir_edit.text().strip(),
            acquisition_index=self._acquisition_index,
            session_start_time=self._session_start,
        )
        self._worker.status_changed.connect(self._set_status)
        self._worker.progress.connect(self._on_progress)
        self._worker.data_ready.connect(self._on_data_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)

        self._acq_time_s = acq_s
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._worker.start()

    def _on_progress(self, val: int):
        self._progress.setValue(val)
        if val == 20 and not self._acq_timer.isActive():
            self._acq_t0 = time.time()
            self._acq_timer.start(1000)

    def _on_acq_tick(self):
        elapsed = time.time() - self._acq_t0
        pct = min(69, 20 + int(50 * elapsed / self._acq_time_s))
        self._progress.setValue(pct)
        remaining = max(0.0, self._acq_time_s - elapsed)
        self._set_status(
            f"Mérés folyamatban... {self._fmt_acqtime(elapsed)} / "
            f"{self._fmt_acqtime(self._acq_time_s)}"
            f"  (még ~{self._fmt_acqtime(remaining)})"
        )

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._btn_stop.setEnabled(False)
            self._set_status("Leállítás folyamatban...")
            self._worker.request_stop()

    def _on_worker_finished(self):
        self._acq_timer.stop()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setValue(0)
        if self._worker and self._worker._stop_requested:
            self._set_status("Mérés leállítva.")

    def _on_data_ready(self, result: NoiseResult):
        self._last_result = result
        self._acquisition_index += 1
        self._update_result_labels(result)
        self._update_plots(result)

    def _update_result_labels(self, r: NoiseResult):
        self._lbl_rms_scope.setText(self._fmt_voltage(r.rms_scope_v))
        self._lbl_pp_scope.setText(self._fmt_voltage(r.pp_scope_v))
        self._lbl_rms_input.setText(self._fmt_voltage(r.rms_input_v))
        self._lbl_pp_input.setText(self._fmt_voltage(r.pp_input_v))
        self._lbl_integ.setText(self._fmt_voltage(r.integrated_noise_rms_v))
        self._lbl_clipping.setText("CLIP!" if r.clipping_detected else "OK")
        if r.clipping_detected:
            self._lbl_clipping.setStyleSheet("color: red; font-weight: bold;")
        else:
            self._lbl_clipping.setStyleSheet("")
        spurs_str = ", ".join(f"{s:.0f}Hz" for s in r.mains_spurs_hz) or "—"
        self._lbl_spurs.setText(spurs_str)
        pct = r.parseval_error_pct
        parseval_str = f"{pct:+.1f}%"
        self._lbl_parseval.setText(parseval_str)
        if abs(pct) > 10.0:
            self._lbl_parseval.setStyleSheet("color: #cc6600; font-weight: bold;")
        else:
            self._lbl_parseval.setStyleSheet("")

    def _update_plots(self, r: NoiseResult):
        gain = r.metadata.gain
        f_low = r.metadata.f_low_hz
        f_high = r.metadata.f_high_hz

        self._fig.clf()
        self._ax_time = self._fig.add_subplot(211)
        self._ax_psd = self._fig.add_subplot(212)

        # --- Idotartomany ---
        v_uv = r.voltage_filtered * 1e6
        self._ax_time.plot(r.time, v_uv, linewidth=0.5, color="C0")
        self._ax_time.set_xlabel("Ido [s]")
        self._ax_time.set_ylabel("Scope [µV]", color="C0")
        self._ax_time.tick_params(axis="y", labelcolor="C0")
        self._ax_time.set_title("Idotartomany (DC-drift eltavolitva)")
        self._ax_time.grid(True, alpha=0.3)

        ax2t = self._ax_time.twinx()
        ylo, yhi = self._ax_time.get_ylim()
        ax2t.set_ylim(ylo / gain * 1e3, yhi / gain * 1e3)
        ax2t.set_ylabel("Input [nV]", color="C1")
        ax2t.tick_params(axis="y", labelcolor="C1")

        # --- PSD ---
        f = r.freqs[1:]
        psd_scope = r.psd_v2_per_hz[1:]
        psd_input = psd_scope / gain ** 2

        self._ax_psd.loglog(f, psd_scope, color="C0", label="Scope PSD")
        self._ax_psd.axvspan(f_low, f_high, alpha=0.1, color="green",
                             label=f"{self._fmt_hz(f_low)}–{self._fmt_hz(f_high)} sav")
        self._ax_psd.set_xlabel("Frekvencia [Hz]")
        self._ax_psd.set_ylabel("Scope PSD [V²/Hz]", color="C0")
        self._ax_psd.tick_params(axis="y", labelcolor="C0")
        self._ax_psd.set_title("Teljessegiszint-surűseg (Welch)")
        self._ax_psd.legend(loc="upper right")
        self._ax_psd.grid(True, which="both", alpha=0.3)

        ax2p = self._ax_psd.twinx()
        ax2p.set_yscale("log")
        ylo_p, yhi_p = self._ax_psd.get_ylim()
        ax2p.set_ylim(ylo_p / gain ** 2, yhi_p / gain ** 2)
        ax2p.set_ylabel("Input PSD [V²/Hz]", color="C1")
        ax2p.tick_params(axis="y", labelcolor="C1")

        self._fig.tight_layout()
        self._canvas.draw()

    def _on_error(self, msg: str):
        self._acq_timer.stop()
        self._set_status(f"HIBA: {msg}")
        QMessageBox.critical(self, "Meresi hiba", msg)

    def _set_status(self, msg: str):
        self._status_bar.showMessage(msg)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
