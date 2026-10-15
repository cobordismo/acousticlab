#!/usr/bin/env python3
"""
Phononic Crystal Lab - systematic white-noise / sweep measurements with / without crystal.

    pip install numpy scipy matplotlib sounddevice soundfile
    python3 phononic_lab.py

What it does
------------
* Lists the computer's audio inputs/outputs and shows the microphone in use,
  with a live level meter (dBFS, peak hold, clip warning).
* Test signal: white noise played by another device (e.g. a phone; record only,
  default) or an exponential sine sweep. The sweep is played through the chosen
  output while the microphone records (sample-synchronous), or it can also come
  from another device (e.g. the YouTube 20 Hz-20 kHz, 30 s sweep).
* White-noise takes: FFT of the whole recording -> |FFT|^2 = |H(f)|^2 up to a
  constant that cancels in with - without. Leave ~1 s of silence at the start:
  it is the background-noise reference for the SNR.
* Each recording is a "take" labelled WITH or WITHOUT crystal. A protocol button
  runs N alternating pairs (without, with, without, with, ...) and tells you
  what to do before each take.
* Sweep takes: deconvolution with the known sweep -> impulse response ->
  time window (removes harmonic distortion, background noise and late room
  reflections) -> transfer function |H(f)|, plus a noise floor / SNR estimate.
* Combines all takes: mean |H| with crystal, mean |H| without, difference in dB
  with an uncertainty band from the spread between repetitions. Optional
  "relative" view (difference minus its median), Bragg marker f = c/2a, and an
  automatic list of dips.
* Everything is saved in a session folder: WAV files, session.json, CSV, PNG.
"""
import datetime
import json
import os
import queue
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import find_peaks

try:
    import sounddevice as sd
    SD_ERROR = None
except Exception as exc:  # PortAudio missing, etc.
    sd = None
    SD_ERROR = exc

import matplotlib


def _install_tkagg_fallback():
    """Pure-Python stand-in for matplotlib's compiled _tkagg extension.

    Windows Smart App Control / Application Control policies can block the
    unsigned _tkagg.pyd ("DLL load failed ... An Application Control policy
    has blocked this file"). If so, register a module that blits the Agg
    buffer into the Tk PhotoImage via base64-encoded PNG data instead.
    """
    import sys
    try:
        import matplotlib.backends._tkagg  # noqa: F401
        return
    except ImportError:
        pass

    import base64
    import struct
    import types
    import zlib

    def png_bytes(rgba):
        h, w = rgba.shape[:2]
        raw = np.zeros((h, w * 4 + 1), dtype=np.uint8)   # filter byte 0 per row
        raw[:, 1:] = rgba.reshape(h, w * 4)

        def chunk(tag, payload):
            return (struct.pack(">I", len(payload)) + tag + payload
                    + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

        return (b"\x89PNG\r\n\x1a\n"
                + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1))
                + chunk(b"IEND", b""))

    mod = types.ModuleType("matplotlib.backends._tkagg")
    mod.TK_PHOTO_COMPOSITE_OVERLAY = 0
    mod.TK_PHOTO_COMPOSITE_SET = 1

    def blit(interp, photo_name, data, comp_rule, offsets, bbox):
        x1, x2, y1, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return
        height = data.shape[0]
        # bbox y is measured from the bottom; Agg rows start at the top.
        rgba = np.ascontiguousarray(
            data[height - y2:height - y1, x1:x2][..., list(offsets)])
        if comp_rule == mod.TK_PHOTO_COMPOSITE_SET:
            tk._default_root.tk.call(photo_name, "blank")
        tk._default_root.tk.call(
            photo_name, "put", base64.b64encode(png_bytes(rgba)).decode("ascii"),
            "-format", "png", "-to", x1, height - y2)

    mod.blit = blit
    mod.enable_dpi_awareness = lambda frame_handle, interp: False
    sys.modules["matplotlib.backends._tkagg"] = mod


_install_tkagg_fallback()
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

C_SOUND = 343.0          # m/s at ~20 C
MAX_REC_S = 300.0        # longest recording (s), also the buffer for "until I press Stop"
COL_WITH = "#eb6834"     # orange
COL_WITHOUT = "#2a78d6"  # blue
COL_DIFF = "#4a3aa7"     # violet
LABELS = {"com": "with crystal", "sem": "without crystal"}



# ============================================================================ DSP
def make_sweep(f1, f2, T, fs, amp=1.0):
    """Exponential (log) sine sweep, with 10 ms fades to avoid clicks."""
    L = T / np.log(f2 / f1)
    t = np.arange(int(round(T * fs))) / fs
    x = np.sin(2 * np.pi * f1 * L * (np.exp(t / L) - 1.0))
    nf = int(0.01 * fs)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nf))
    x[:nf] *= ramp
    x[-nf:] *= ramp[::-1]
    return (amp * x).astype(np.float32)


def noise_spectrum(y, fs, bg_s=1.0):
    """White noise from a phone: power spectrum |FFT|^2 of the whole recording. The noise is flat,
    so this is |H(f)|^2 times a constant that cancels in with - without; the fractional-octave
    smoothing in the analysis averages out the randomness of the noise.
    The first bg_s seconds (silence before the noise starts) give the background for the SNR;
    None if they are not quiet (noise already playing)."""
    n = len(y)
    f = np.fft.rfftfreq(n, 1 / fs)
    P = np.abs(np.fft.rfft(y)) ** 2 / n
    bg = y[:int(bg_s * fs)]
    if np.mean(bg ** 2) > 0.1 * np.mean(y ** 2):
        return f, P, None
    return f, P, np.abs(np.fft.rfft(bg, n)) ** 2 / len(bg)   # zero-padded: same frequency grid


def impulse_response(y, x, fs, f1, f2):
    """Regularised deconvolution h = IFFT(Y X* / (|X|^2 + eps)), band-limited to the sweep."""
    N = 1 << int(np.ceil(np.log2(len(y) + len(x))))
    X = np.fft.rfft(x, N)
    Y = np.fft.rfft(y, N)
    f = np.fft.rfftfreq(N, 1 / fs)
    lo, hi = f1 * 1.05, min(f2, 0.49 * fs) * 0.97
    w = ((f > lo) & (f < hi)).astype(float)
    w = signal.convolve(w, np.hanning(401) / np.hanning(401).sum(), mode="same")
    P = np.abs(X) ** 2
    H = Y * np.conj(X) / (P + 1e-3 * np.median(P[(f > lo) & (f < hi)])) * w
    return np.fft.irfft(H, N)


def transfer_function(h, fs, n_rec, n_sweep, win_ms, pre_ms=2.0):
    """Window the linear IR around its main peak and return f, |H|^2, noise |N|^2, peak time."""
    # The linear IR sits at the time the sweep starts in the recording. Search for it there.
    search_end = max(n_rec - n_sweep, 0) + int(1.0 * fs)
    search_end = min(search_end, len(h) // 2)
    env = np.abs(h[:search_end])
    env = np.convolve(env, np.ones(16) / 16, mode="same")
    k = int(np.argmax(env))
    pre = int(pre_ms * 1e-3 * fs)
    win = max(int(win_ms * 1e-3 * fs), 64)
    a = max(k - pre, 0)
    seg = h[a:k + win].copy()
    taper = np.ones(len(seg))
    n_in = k - a
    if n_in > 1:
        taper[:n_in] = np.hanning(2 * n_in)[:n_in]
    n_out = max(int(0.25 * win), 8)
    taper[-n_out:] *= np.hanning(2 * n_out)[n_out:]
    seg *= taper
    # noise reference: same-length window, late in the IR (after the room has decayed)
    b = k + int(0.6 * fs)
    nseg = h[b:b + len(seg)] * taper if b + len(seg) < len(h) // 2 else np.zeros(len(seg))
    nfft = max(1 << int(np.ceil(np.log2(len(seg)))), 1 << 15)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, np.abs(np.fft.rfft(seg, nfft)) ** 2, np.abs(np.fft.rfft(nseg, nfft)) ** 2, k / fs


def frac_oct(f, p, frac, fgrid):
    """Mean power in a 1/frac-octave band centred on each grid frequency."""
    c = np.concatenate([[0.0], np.cumsum(p)])
    lo = np.searchsorted(f, fgrid * 2 ** (-0.5 / frac))
    hi = np.maximum(np.searchsorted(f, fgrid * 2 ** (0.5 / frac)), lo + 1)
    hi = np.minimum(hi, len(p))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def db(p):
    return 10 * np.log10(np.maximum(p, 1e-30))


def load_audio(path, fs_target=None):
    """Read wav/flac/... with soundfile; anything else (m4a, mp3) through ffmpeg."""
    try:
        y, fs = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("This format needs ffmpeg installed (or convert it to WAV first).")
        tmp = path + ".tmp.wav"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, "-ac", "1"]
        if fs_target:
            cmd += ["-ar", str(fs_target)]
        subprocess.run(cmd + [tmp], check=True)
        y, fs = sf.read(tmp, dtype="float32", always_2d=True)
        os.remove(tmp)
    return y.mean(axis=1), fs


# ============================================================================ audio I/O
class LevelMonitor:
    """Input stream used only for the live meter while idle."""

    def __init__(self):
        self.stream = None
        self.level = 0.0
        self.lock = threading.Lock()

    def start(self, device, fs):
        self.stop()
        if sd is None or device is None:
            return
        def cb(indata, frames, t, status):
            with self.lock:
                self.level = max(self.level, float(np.max(np.abs(indata[:, 0]))))
        self.stream = sd.InputStream(device=device, channels=1, samplerate=fs, callback=cb, blocksize=1024)
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop(); self.stream.close()
            except Exception:
                pass
            self.stream = None

    def take_level(self):
        with self.lock:
            v, self.level = self.level, 0.0
        return v


class Recorder:
    """Plays `out_signal` (mono) on the output while recording the input, sample-synchronous.
    With out_signal=None it only records n_frames from the input."""

    def __init__(self):
        self.level = 0.0
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.stream = None
        self.xruns = 0

    def start(self, in_dev, out_dev, fs, n_frames, out_signal=None):
        self.buf = np.zeros(n_frames, np.float32)
        self.pos = 0
        self.xruns = 0
        self.done.clear()
        self.sig = out_signal

        def cb_in(indata, frames, t, status):
            self._store(indata, frames, status)

        def cb_duplex(indata, outdata, frames, t, status):
            chunk = self.sig[self.pos:self.pos + frames]
            outdata.fill(0)
            outdata[:len(chunk)] = chunk[:, None]
            self._store(indata, frames, status)

        fin = self.done.set
        if out_signal is None:
            self.stream = sd.InputStream(device=in_dev, channels=1, samplerate=fs, callback=cb_in,
                                         finished_callback=fin)
        else:
            out_ch = max(1, min(2, sd.query_devices(out_dev)["max_output_channels"]))
            self.stream = sd.Stream(device=(in_dev, out_dev), channels=(1, out_ch), samplerate=fs,
                                    callback=cb_duplex, finished_callback=fin, dtype="float32")
        self.stream.start()

    def _store(self, indata, frames, status):
        if status:
            self.xruns += 1
        n = min(frames, len(self.buf) - self.pos)
        if n > 0:
            self.buf[self.pos:self.pos + n] = indata[:n, 0]
        with self.lock:
            self.level = max(self.level, float(np.max(np.abs(indata[:, 0]))))
        self.pos += frames
        if self.pos >= len(self.buf):
            raise sd.CallbackStop

    def abort(self):
        if self.stream is not None:
            self.stream.abort()

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def take_level(self):
        with self.lock:
            v, self.level = self.level, 0.0
        return v

    @property
    def progress(self):
        return min(self.pos / max(len(self.buf), 1), 1.0)


# ============================================================================ GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phononic Crystal Lab")
        self.geometry("1380x880")
        self.minsize(1100, 720)

        self.session_dir = None
        self.takes = []                 # list of dicts (session.json)
        self.cache = {}                 # (take id, win_ms, fs) -> (f, P, N)
        self.monitor = LevelMonitor()
        self.recorder = Recorder()
        self.recording = False
        self.protocol_queue = []
        self.work_q = queue.Queue()
        self.peak_hold, self.peak_hold_t = -120.0, 0
        self.last_result = None
        self.analysis_lock = threading.Lock()

        self._build_vars()
        self._build_ui()
        self.refresh_devices()
        self.after(60, self._tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if SD_ERROR is not None:
            messagebox.showwarning("Audio not available",
                                   f"sounddevice could not start:\n{SD_ERROR}\n\n"
                                   "You can still open sessions and import files.")

    # ------------------------------------------------------------------ state
    def _build_vars(self):
        self.v_in = tk.StringVar(); self.v_out = tk.StringVar(); self.v_fs = tk.StringVar(value="48000")
        self.v_f1 = tk.DoubleVar(value=20.0); self.v_f2 = tk.DoubleVar(value=20000.0)
        self.v_T = tk.DoubleVar(value=30.0); self.v_amp = tk.DoubleVar(value=-12.0)
        self.v_signal = tk.StringVar(value="noise")  # test signal: "noise" (white noise) or "sweep"
        self.v_mode = tk.StringVar(value="play")
        self.v_reclen = tk.DoubleVar(value=35.0)    # total recording time (s)
        self.v_start = tk.DoubleVar(value=0.5)      # when the sweep starts inside the recording (play mode)
        self.v_manual = tk.BooleanVar(value=False)  # record until Stop is pressed
        self.v_count = tk.DoubleVar(value=0.0)      # countdown before recording starts
        self.v_timehint = tk.StringVar()
        self.v_rectime = tk.StringVar(value="")
        self.v_label = tk.StringVar(value="sem")
        self.v_notes = tk.StringVar()
        self.v_reps = tk.IntVar(value=3)
        self.v_smooth = tk.StringVar(value="1/12")
        self.v_win = tk.DoubleVar(value=30.0)
        self.v_prom = tk.DoubleVar(value=3.0)       # smallest dip prominence (dB) that is detected
        self.v_a = tk.StringVar()
        self.v_rel = tk.BooleanVar(value=True)
        self.v_status = tk.StringVar(value="Create or open a session to start.")
        self.v_mic = tk.StringVar(value="Microphone: -")
        self.v_session = tk.StringVar(value="No session")

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Head.TLabel", font=("TkDefaultFont", 10, "bold"))
        style.configure("Mic.TLabel", font=("TkDefaultFont", 10, "bold"), foreground="#14171a")
        style.configure("Big.TButton", font=("TkDefaultFont", 11, "bold"), padding=6)

        ttk.Label(self, textvariable=self.v_status, anchor="w", padding=(10, 4), relief="sunken").pack(fill="x", side="bottom")
        root = ttk.Frame(self); root.pack(fill="both", expand=True)
        # left column is scrollable so it fits small screens
        outer = ttk.Frame(root); outer.pack(side="left", fill="y")
        bg = style.lookup("TFrame", "background") or self.cget("bg")
        self.left_cv = tk.Canvas(outer, highlightthickness=0, bg=bg, width=340)
        sb = ttk.Scrollbar(outer, orient="vertical", command=self.left_cv.yview)
        left = ttk.Frame(self.left_cv, padding=10)
        self.left_cv.create_window((0, 0), window=left, anchor="nw")
        def _fit(_e=None):
            self.left_cv.configure(scrollregion=self.left_cv.bbox("all"), width=left.winfo_reqwidth())
        left.bind("<Configure>", _fit)
        self.left_cv.configure(yscrollcommand=sb.set)
        self.left_cv.pack(side="left", fill="y"); sb.pack(side="left", fill="y")
        def _wheel(e):
            delta = -1 if (getattr(e, "num", 0) == 4 or e.delta > 0) else 1
            self.left_cv.yview_scroll(delta, "units")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.left_cv.bind_all(seq, lambda e: _wheel(e) if str(e.widget).startswith(str(self.left_cv)) else None, add="+")
        right = ttk.Frame(root, padding=(0, 10, 10, 10)); right.pack(side="left", fill="both", expand=True)

        # --- session
        box = ttk.LabelFrame(left, text="Session", padding=8); box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, textvariable=self.v_session, wraplength=300).pack(anchor="w")
        row = ttk.Frame(box); row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="New…", command=self.new_session).pack(side="left")
        ttk.Button(row, text="Open…", command=self.open_session).pack(side="left", padx=4)
        ttk.Button(row, text="Import audio file…", command=self.import_file).pack(side="left")

        # --- audio devices + meter
        box = ttk.LabelFrame(left, text="Audio", padding=8); box.pack(fill="x", pady=(0, 8))
        ttk.Label(box, text="Microphone (input)").grid(row=0, column=0, sticky="w")
        self.cb_in = ttk.Combobox(box, textvariable=self.v_in, state="readonly", width=40)
        self.cb_in.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.cb_in.bind("<<ComboboxSelected>>", lambda e: self._restart_monitor())
        ttk.Label(box, text="Speaker (output)").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.cb_out = ttk.Combobox(box, textvariable=self.v_out, state="readonly", width=40)
        self.cb_out.grid(row=3, column=0, columnspan=3, sticky="ew")
        ttk.Label(box, text="Sample rate").grid(row=4, column=0, sticky="w", pady=(6, 0))
        cb_fs = ttk.Combobox(box, textvariable=self.v_fs, values=["44100", "48000", "96000"], width=8, state="readonly")
        cb_fs.grid(row=4, column=1, sticky="w", pady=(6, 0))
        cb_fs.bind("<<ComboboxSelected>>", lambda e: self._restart_monitor())
        ttk.Button(box, text="Refresh", command=self.refresh_devices).grid(row=4, column=2, sticky="e", pady=(6, 0))
        ttk.Label(box, textvariable=self.v_mic, style="Mic.TLabel", wraplength=300).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(8, 2))
        self.meter = tk.Canvas(box, height=22, width=300, bg="#1b1d20", highlightthickness=0)
        self.meter.grid(row=6, column=0, columnspan=3, sticky="ew")
        self.meter_txt = ttk.Label(box, text="-∞ dBFS")
        self.meter_txt.grid(row=7, column=0, columnspan=3, sticky="w")
        box.columnconfigure(0, weight=1)

        # --- take
        box = ttk.LabelFrame(left, text="Take", padding=8); box.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(box, text="WITHOUT crystal", variable=self.v_label, value="sem").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(box, text="WITH crystal", variable=self.v_label, value="com").grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="Notes").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(box, textvariable=self.v_notes, width=30).grid(row=2, column=0, columnspan=2, sticky="ew")
        row = ttk.Frame(box); row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.btn_rec = ttk.Button(row, text="● Record take", style="Big.TButton", command=self.record_single)
        self.btn_rec.pack(side="left")
        self.btn_stop = ttk.Button(row, text="■ Stop & save", command=self.stop_recording, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Label(row, textvariable=self.v_rectime, font=("TkFixedFont", 11)).pack(side="left", padx=4)
        self.progress = ttk.Progressbar(box, maximum=1.0)
        self.progress.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        row = ttk.Frame(box); row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(row, text="Protocol: pairs").pack(side="left")
        ttk.Spinbox(row, from_=1, to=20, textvariable=self.v_reps, width=4).pack(side="left", padx=4)
        ttk.Button(row, text="Run protocol", command=self.run_protocol).pack(side="left")

        # --- test signal
        box = ttk.LabelFrame(left, text="Test signal", padding=8); box.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(box); row.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        ttk.Radiobutton(row, text="White noise (phone)", variable=self.v_signal, value="noise").pack(side="left")
        ttk.Radiobutton(row, text="Sine sweep", variable=self.v_signal, value="sweep").pack(side="left", padx=8)
        self.rb_play = ttk.Radiobutton(box, text="Sweep: play from this computer (recommended)", variable=self.v_mode,
                                       value="play")
        self.rb_play.grid(row=1, column=0, columnspan=4, sticky="w")
        ttk.Radiobutton(box, text="Sweep: external (record only)", variable=self.v_mode,
                        value="external").grid(row=2, column=0, columnspan=4, sticky="w")
        for r, (lab, var) in enumerate([("f low (Hz)", self.v_f1), ("f high (Hz)", self.v_f2),
                                        ("duration (s)", self.v_T), ("peak (dBFS)", self.v_amp)]):
            ttk.Label(box, text=lab).grid(row=3 + r // 2, column=(r % 2) * 2, sticky="w", pady=2)
            ttk.Entry(box, textvariable=var, width=8).grid(row=3 + r // 2, column=(r % 2) * 2 + 1, sticky="w", pady=2)
        ttk.Separator(box).grid(row=5, column=0, columnspan=4, sticky="ew", pady=6)
        ttk.Label(box, text="Record for (s)").grid(row=6, column=0, sticky="w", pady=2)
        self.e_reclen = ttk.Entry(box, textvariable=self.v_reclen, width=8)
        self.e_reclen.grid(row=6, column=1, sticky="w", pady=2)
        ttk.Label(box, text="signal at (s)").grid(row=6, column=2, sticky="w", pady=2)
        self.e_start = ttk.Entry(box, textvariable=self.v_start, width=8)
        self.e_start.grid(row=6, column=3, sticky="w", pady=2)
        ttk.Checkbutton(box, text="until I press Stop", variable=self.v_manual,
                        command=self._update_time_hint).grid(row=7, column=0, columnspan=2, sticky="w")
        ttk.Label(box, text="countdown (s)").grid(row=7, column=2, sticky="w")
        ttk.Entry(box, textvariable=self.v_count, width=8).grid(row=7, column=3, sticky="w")
        self.lbl_hint = ttk.Label(box, textvariable=self.v_timehint, foreground="#666", wraplength=300)
        self.lbl_hint.grid(row=8, column=0, columnspan=4, sticky="w")
        for v in (self.v_reclen, self.v_start, self.v_T, self.v_mode, self.v_manual, self.v_signal):
            v.trace_add("write", lambda *a: self._update_time_hint())
        self.after(100, self._update_time_hint)

        # --- right side: analysis bar + tabs
        bar = ttk.Frame(right); bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="Analysis", style="Head.TLabel").pack(side="left", padx=(0, 10))
        ttk.Label(bar, text="Smoothing").pack(side="left")
        ttk.Combobox(bar, textvariable=self.v_smooth, values=["1/24", "1/12", "1/6", "1/3"], width=5,
                     state="readonly").pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="IR window (ms)").pack(side="left")
        ttk.Entry(bar, textvariable=self.v_win, width=6).pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="Min dip (dB)").pack(side="left")
        ttk.Entry(bar, textvariable=self.v_prom, width=5).pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="Lattice a (cm)").pack(side="left")
        ttk.Entry(bar, textvariable=self.v_a, width=6).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(bar, text="Relative", variable=self.v_rel).pack(side="left", padx=(0, 12))
        ttk.Button(bar, text="Update analysis", command=self.analyse).pack(side="left")
        ttk.Button(bar, text="Export", command=self.export).pack(side="left", padx=6)
        self.nb = ttk.Notebook(right); self.nb.pack(fill="both", expand=True)
        # takes table
        tab = ttk.Frame(self.nb, padding=6); self.nb.add(tab, text="Takes")
        cols = ("id", "label", "time", "mode", "len", "mic", "peak", "snr", "use", "notes")
        self.tree = ttk.Treeview(tab, columns=cols, show="headings", height=20)
        for c, w, t in [("id", 50, "#"), ("label", 120, "Configuration"), ("time", 150, "Time"),
                        ("mode", 90, "Signal"), ("len", 70, "Length"), ("mic", 260, "Microphone"), ("peak", 80, "Peak dBFS"),
                        ("snr", 90, "SNR median"), ("use", 50, "Use"), ("notes", 260, "Notes")]:
            self.tree.heading(c, text=t); self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("off", foreground="#999")
        self.tree.tag_configure("warn", foreground="#b3261e")
        row = ttk.Frame(tab); row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Use / skip selected", command=self.toggle_selected).pack(side="left")
        ttk.Button(row, text="Swap label of selected", command=self.swap_selected).pack(side="left", padx=6)
        ttk.Button(row, text="Delete selected", command=self.delete_selected).pack(side="left")
        ttk.Label(row, text="Red = clipped or low SNR. Skipped takes are excluded from the averages.",
                  foreground="#666").pack(side="left", padx=12)

        self.fig_spec, self.ax_spec = self._make_fig("Transfer functions")
        self.fig_diff, self.ax_diff = self._make_fig("With − without")
        self.fig_last = Figure(figsize=(8, 5), dpi=100, layout="constrained")
        tab = ttk.Frame(self.nb); self.nb.add(tab, text="Last recording")
        self.tab_last = tab
        self.canvas_last = FigureCanvasTkAgg(self.fig_last, master=tab)
        NavigationToolbar2Tk(self.canvas_last, tab).update()
        self.canvas_last.get_tk_widget().pack(fill="both", expand=True)

        # dips list under diff plot
        self.dips_txt = tk.Text(self.diff_tab, height=8, wrap="word", relief="flat", bg=self.cget("bg"))
        self.dips_txt.pack(fill="x")


    def _make_fig(self, title):
        tab = ttk.Frame(self.nb); self.nb.add(tab, text=title)
        fig = Figure(figsize=(8, 5), dpi=100, layout="constrained")
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=tab)
        NavigationToolbar2Tk(canvas, tab).update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        fig._canvas = canvas
        if title.startswith("With"):
            self.diff_tab = tab
        return fig, ax

    # ------------------------------------------------------------------ devices
    def refresh_devices(self):
        if sd is None:
            return
        try:
            sd._terminate(); sd._initialize()
        except Exception:
            pass
        devs = sd.query_devices()
        apis = sd.query_hostapis()
        self.dev_in, self.dev_out = [], []
        for i, d in enumerate(devs):
            name = f"[{i}] {d['name']}  ({apis[d['hostapi']]['name']})"
            if d["max_input_channels"] > 0:
                self.dev_in.append((i, name))
            if d["max_output_channels"] > 0:
                self.dev_out.append((i, name))
        self.cb_in["values"] = [n for _, n in self.dev_in]
        self.cb_out["values"] = [n for _, n in self.dev_out]
        di, do = sd.default.device
        def pick(lst, default):
            for i, n in lst:
                if i == default:
                    return n
            return lst[0][1] if lst else ""
        if self.v_in.get() not in self.cb_in["values"]:
            self.v_in.set(pick(self.dev_in, di))
        if self.v_out.get() not in self.cb_out["values"]:
            self.v_out.set(pick(self.dev_out, do))
        self._restart_monitor()

    def _dev_index(self, lst, name):
        for i, n in lst:
            if n == name:
                return i
        return None

    def _restart_monitor(self):
        if sd is None or self.recording:
            return
        i = self._dev_index(getattr(self, "dev_in", []), self.v_in.get())
        if i is None:
            self.v_mic.set("Microphone: none found")
            return
        fs = int(self.v_fs.get())
        try:
            self.monitor.start(i, fs)
            self.v_mic.set(f"Recording from: {sd.query_devices(i)['name']}  ·  {fs} Hz")
        except Exception as exc:
            self.monitor.stop()
            self.v_mic.set(f"Microphone error: {exc}")

    # ------------------------------------------------------------------ meter loop
    def _tick(self):
        lvl = self.recorder.take_level() if self.recording else self.monitor.take_level()
        d = 20 * np.log10(lvl) if lvl > 0 else -120.0
        now = self.tk.call("clock", "milliseconds")
        if d > self.peak_hold or now - self.peak_hold_t > 1500:
            self.peak_hold, self.peak_hold_t = d, now
        self._draw_meter(d)
        if self.recording:
            el = self.recorder.pos / self.pending["fs"]
            if self.pending["manual"]:
                self.progress["value"] = (el % 10) / 10
                self.v_rectime.set(f"{el:5.1f} s")
            else:
                self.progress["value"] = self.recorder.progress
                self.v_rectime.set(f"{el:5.1f} / {self.pending['reclen']:.1f} s")
            self.live_ticks += 1
            if self.live_ticks % 3 == 0:
                self._live_update()
            if self.recorder.done.is_set():
                self._finish_recording()
        try:
            while True:
                kind, payload = self.work_q.get_nowait()
                if kind == "analysis":
                    self._show_analysis(payload)
                elif kind == "status":
                    self.v_status.set(payload)
        except queue.Empty:
            pass
        self.after(60, self._tick)

    def _draw_meter(self, d):
        c = self.meter
        w = max(c.winfo_width(), 50); h = 22
        c.delete("all")
        lo = -60.0
        frac = np.clip((d - lo) / -lo, 0, 1)
        col = "#2fa84f" if d < -12 else ("#e0a800" if d < -3 else "#d93025")
        c.create_rectangle(0, 0, frac * w, h, fill=col, width=0)
        ph = np.clip((self.peak_hold - lo) / -lo, 0, 1) * w
        c.create_line(ph, 0, ph, h, fill="#ffffff", width=2)
        for t in (-48, -36, -24, -12, -6, 0):
            x = (t - lo) / -lo * w
            c.create_line(x, h - 5, x, h, fill="#888")
        txt = "-∞ dBFS" if d <= -119 else f"{d:5.1f} dBFS   (peak hold {self.peak_hold:5.1f})"
        if self.peak_hold > -1:
            txt += "   CLIPPING - lower the input gain or volume"
        self.meter_txt.configure(text=txt)

    def _update_time_hint(self):
        try:
            T, rl, st = float(self.v_T.get()), float(self.v_reclen.get()), float(self.v_start.get())
        except (tk.TclError, ValueError):
            self.v_timehint.set("Enter numbers for the times."); return
        manual = bool(self.v_manual.get())
        noise = self.v_signal.get() == "noise"
        play = self.v_mode.get() == "play" and not noise   # white noise always comes from the phone
        self.rb_play.configure(state="disabled" if noise else "normal")
        self.e_reclen.configure(state="disabled" if manual else "normal")
        self.e_start.configure(state="normal" if play else "disabled")
        if noise:
            txt = ("Press Record, wait ~1 s, then start the white noise on the phone: "
                   "the first second of silence measures the background noise. " +
                   (f"Recording runs until you press Stop (max {MAX_REC_S:.0f} s)." if manual
                    else f"Recording lasts {rl:g} s.") + " f low / f high set the analysed band.")
        elif play:
            end = st + T
            if manual:
                txt = f"Sweep plays from {st:g} s to {end:g} s. Recording runs until you press Stop (max {MAX_REC_S:.0f} s)."
            elif rl < end:
                txt = f"Recording ({rl:g} s) ends before the sweep ({end:g} s): increase 'Record for'."
            else:
                txt = f"Sweep plays from {st:g} s to {end:g} s, then {rl - end:g} s of tail. Total {rl:g} s."
        else:
            txt = ("Start the external sweep after pressing Record. " +
                   (f"Recording runs until you press Stop (max {MAX_REC_S:.0f} s)." if manual
                    else f"Recording lasts {rl:g} s; the sweep needs {T:g} s.") +
                   " Set f and duration to match the external sweep.")
        self.v_timehint.set(txt)
        self.lbl_hint.configure(foreground="#b3261e" if (play and not manual and rl < st + T) else "#666")

    # ------------------------------------------------------------------ session
    def new_session(self):
        d = filedialog.askdirectory(title="Choose (or create) an empty folder for the session")
        if not d:
            return
        name = "session_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(d, name)
        os.makedirs(os.path.join(path, "audio"), exist_ok=True)
        self.session_dir, self.takes, self.cache = path, [], {}
        self._save_session()
        self._refresh_table()
        self.v_session.set(path)
        self.v_status.set("Session created. Record takes WITHOUT and WITH the crystal, ideally with the protocol.")

    def open_session(self):
        d = filedialog.askdirectory(title="Open session folder (contains session.json)")
        if not d:
            return
        p = os.path.join(d, "session.json")
        if not os.path.exists(p):
            messagebox.showerror("Not a session", "This folder has no session.json.")
            return
        with open(p) as fh:
            data = json.load(fh)
        self.session_dir, self.takes, self.cache = d, data.get("takes", []), {}
        self.v_a.set(data.get("lattice_a_cm", ""))
        self.v_session.set(d)
        self._refresh_table()
        self.analyse()

    def _save_session(self):
        if not self.session_dir:
            return
        data = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
                    lattice_a_cm=self.v_a.get(), takes=self.takes)
        with open(os.path.join(self.session_dir, "session.json"), "w") as fh:
            json.dump(data, fh, indent=2)

    def _need_session(self):
        if self.session_dir:
            return True
        messagebox.showinfo("Session needed", "Create or open a session first.")
        return False

    def _next_id(self):
        return 1 + max([t["id"] for t in self.takes], default=0)

    def _sweep_cfg(self):
        f1, f2, T = float(self.v_f1.get()), float(self.v_f2.get()), float(self.v_T.get())
        if not (0 < f1 < f2) or T <= 1:
            raise ValueError("Check the signal settings: 0 < f low < f high, duration > 1 s.")
        return dict(signal=self.v_signal.get(), f1=f1, f2=f2, T=T, level_dbfs=float(self.v_amp.get()))

    # ------------------------------------------------------------------ recording
    def record_single(self):
        if not self._need_session():
            return
        self._start_take(self.v_label.get())

    def run_protocol(self):
        if not self._need_session():
            return
        n = max(1, int(self.v_reps.get()))
        self.protocol_queue = ["sem", "com"] * n
        self.v_status.set(f"Protocol: {2 * n} takes, alternating without / with crystal.")
        self._protocol_next()

    def _protocol_next(self):
        if not self.protocol_queue:
            self.v_status.set("Protocol finished.")
            self.analyse()
            return
        lab = self.protocol_queue[0]
        total = 2 * max(1, int(self.v_reps.get()))
        k = total - len(self.protocol_queue) + 1
        action = ("REMOVE the crystal" if lab == "sem" else "PUT the crystal in place")
        ok = messagebox.askokcancel(
            f"Take {k} of {total}: {LABELS[lab].upper()}",
            f"{action}.\n\nDo not move the speaker, the microphone or the volume.\n"
            f"Keep the room quiet during the measurement.\n\nPress OK to record.")
        if not ok:
            self.protocol_queue = []
            self.v_status.set("Protocol cancelled.")
            return
        self.protocol_queue.pop(0)
        self.v_label.set(lab)
        self._start_take(lab)

    def _start_take(self, label):
        if sd is None:
            messagebox.showerror("Audio not available", str(SD_ERROR)); return
        if self.recording:
            return
        try:
            cfg = self._sweep_cfg()
        except ValueError as exc:
            messagebox.showerror("Signal settings", str(exc)); return
        fs = int(self.v_fs.get())
        mode = "external" if cfg["signal"] == "noise" else self.v_mode.get()
        i_in = self._dev_index(self.dev_in, self.v_in.get())
        i_out = self._dev_index(self.dev_out, self.v_out.get())
        if i_in is None or (mode == "play" and i_out is None):
            messagebox.showerror("Devices", "Choose a microphone and a speaker."); return
        if cfg["f2"] > 0.49 * fs:
            cfg["f2"] = 0.49 * fs
        try:
            manual = bool(self.v_manual.get())
            reclen = float(self.v_reclen.get())
            t_start = max(float(self.v_start.get()), 0.0) if mode == "play" else 0.0
            count = max(float(self.v_count.get()), 0.0)
        except (tk.TclError, ValueError):
            messagebox.showerror("Recording time", "Recording time, sweep start and countdown must be numbers."); return
        if not manual and reclen <= 1:
            messagebox.showerror("Recording time", "Record for at least 1 s."); return
        if not manual and reclen > MAX_REC_S:
            messagebox.showerror("Recording time", f"Maximum recording time is {MAX_REC_S:.0f} s."); return
        if mode == "play":
            x = make_sweep(cfg["f1"], cfg["f2"], cfg["T"], fs, 10 ** (cfg["level_dbfs"] / 20))
            out = np.concatenate([np.zeros(int(t_start * fs), np.float32), x])
            if not manual and reclen < t_start + cfg["T"]:
                if not messagebox.askyesno("Recording shorter than the sweep",
                                           f"The sweep ends at {t_start + cfg['T']:.1f} s but the recording stops at "
                                           f"{reclen:.1f} s. The highest frequencies will be missing.\n\nRecord anyway?"):
                    return
        else:
            out = None
        n = int((MAX_REC_S if manual else reclen) * fs)
        self.pending = dict(label=label, fs=fs, mode=mode, sweep=cfg, i_in=i_in, i_out=i_out,
                            mic=sd.query_devices(i_in)["name"],
                            speaker=sd.query_devices(i_out)["name"] if (mode == "play" and i_out is not None) else "external",
                            notes=self.v_notes.get(), manual=manual, reclen=reclen, t_start=t_start,
                            n=n, out=out)
        self.btn_rec.configure(state="disabled")
        self._countdown(count)

    def _countdown(self, remaining):
        if remaining > 0.05:
            self.v_status.set(f"Recording starts in {remaining:.0f} s… keep the room quiet.")
            self.v_rectime.set(f"-{remaining:.0f} s")
            self.after(int(min(1.0, remaining) * 1000), lambda: self._countdown(remaining - min(1.0, remaining)))
            return
        self._begin_recording()

    def _begin_recording(self):
        p = self.pending
        label, mode, fs, n, out = p["label"], p["mode"], p["fs"], p["n"], p["out"]
        i_in, i_out = p["i_in"], p["i_out"]
        self.monitor.stop()
        try:
            self.recorder.start(i_in, i_out, fs, n, out)
        except Exception as exc:
            messagebox.showerror("Could not start audio", str(exc))
            self.btn_rec.configure(state="normal")
            self._restart_monitor(); return
        self.recording = True
        self.peak_hold = -120.0
        self._live_init()
        self.btn_rec.configure(state="disabled"); self.btn_stop.configure(state="normal")
        hint = ("Playing sweep and recording" if mode == "play" else
                "Recording - start the white noise on the phone now (after ~1 s of silence)"
                if p["sweep"]["signal"] == "noise" else "Recording - start the external sweep now")
        until = "until you press Stop" if p["manual"] else f"for {p['reclen']:.1f} s"
        self.v_status.set(f"{hint} {until}: {LABELS[label]}  ·  mic: {p['mic']}")

    def stop_recording(self):
        """Stop now and keep what was recorded (the protocol continues if the take is usable)."""
        if self.recording:
            self.recorder.abort()

    def _finish_recording(self):
        self.recording = False
        self.recorder.close()
        self._live_stop()
        self.btn_rec.configure(state="normal"); self.btn_stop.configure(state="disabled")
        self.progress["value"] = 0
        y = self.recorder.buf[:min(self.recorder.pos, len(self.recorder.buf))].copy()
        p = self.pending
        self.v_rectime.set("")
        self._restart_monitor()
        if len(y) < 1.0 * p["fs"]:
            self.v_status.set("Recording shorter than 1 s - take discarded.")
            self.protocol_queue = []
            return
        dur = len(y) / p["fs"]
        incomplete = p["mode"] == "play" and dur < p["t_start"] + p["sweep"]["T"]
        tid = self._next_id()
        fname = f"take{tid:03d}_{p['label']}.wav"
        sf.write(os.path.join(self.session_dir, "audio", fname), y, p["fs"], subtype="FLOAT")
        peak = float(20 * np.log10(np.max(np.abs(y)) + 1e-12))
        take = dict(id=tid, label=p["label"], file=fname, fs=p["fs"], mode=p["mode"], sweep=p["sweep"],
                    mic=p["mic"], speaker=p["speaker"], notes=p["notes"], use=True, peak_dbfs=round(peak, 1),
                    xruns=self.recorder.xruns, time=datetime.datetime.now().isoformat(timespec="seconds"),
                    duration_s=round(dur, 2), sweep_start_s=p["t_start"] if p["mode"] == "play" else None,
                    sweep_complete=not incomplete)
        self.takes.append(take)
        self._save_session()
        self._refresh_table()
        self._plot_last(y, p["fs"], take)
        msg = f"Saved take {tid} ({LABELS[p['label']]}), {dur:.1f} s. Peak {peak:.1f} dBFS."
        if incomplete:
            msg += "  NOTE: stopped before the sweep ended - high frequencies are missing in this take."
        if peak > -1.0:
            msg += "  WARNING: clipping - lower the volume or input gain and repeat."
        if self.recorder.xruns:
            msg += f"  WARNING: {self.recorder.xruns} audio dropouts."
        self.v_status.set(msg)
        self.analyse(then=self._protocol_next if self.protocol_queue else None)

    # ------------------------------------------------------------------ import
    def import_file(self):
        if not self._need_session():
            return
        paths = filedialog.askopenfilenames(title="Import recordings",
                                            filetypes=[("Audio", "*.wav *.flac *.m4a *.mp3 *.aac *.ogg"), ("All", "*.*")])
        if not paths:
            return
        try:
            cfg = self._sweep_cfg()
        except ValueError as exc:
            messagebox.showerror("Signal settings", str(exc)); return
        lab = self.v_label.get()
        what = (f"white noise, analysed {cfg['f1']:g}–{cfg['f2']:g} Hz" if cfg["signal"] == "noise"
                else f"an external sweep {cfg['f1']:g}–{cfg['f2']:g} Hz in {cfg['T']:g} s")
        if not messagebox.askyesno("Import", f"Import {len(paths)} file(s) as '{LABELS[lab]}' with {what}?"):
            return
        for path in paths:
            try:
                y, fs = load_audio(path)
            except Exception as exc:
                messagebox.showerror("Import failed", f"{os.path.basename(path)}:\n{exc}"); continue
            tid = self._next_id()
            fname = f"take{tid:03d}_{lab}.wav"
            sf.write(os.path.join(self.session_dir, "audio", fname), y, fs, subtype="FLOAT")
            self.takes.append(dict(id=tid, label=lab, file=fname, fs=fs, mode="external", sweep=cfg,
                                   mic="imported: " + os.path.basename(path), speaker="external", notes=self.v_notes.get(),
                                   use=True, peak_dbfs=round(float(20 * np.log10(np.max(np.abs(y)) + 1e-12)), 1),
                                   xruns=0, time=datetime.datetime.now().isoformat(timespec="seconds"),
                                   duration_s=round(len(y) / fs, 2)))
        self._save_session(); self._refresh_table(); self.analyse()

    # ------------------------------------------------------------------ takes table
    def _refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        for t in self.takes:
            snr = t.get("snr_median")
            tags = []
            if not t.get("use", True):
                tags.append("off")
            elif t.get("peak_dbfs", -99) > -1 or (snr is not None and snr < 15):
                tags.append("warn")
            self.tree.insert("", "end", iid=str(t["id"]), tags=tags, values=(
                t["id"], LABELS[t["label"]], t["time"].replace("T", " "),
                t["sweep"].get("signal", "sweep") + ("" if t["mode"] == "play" else " (ext)"),
                (f"{t['duration_s']:.1f} s" + ("" if t.get("sweep_complete", True) else " (cut)")) if t.get("duration_s") else "-",
                t["mic"],
                f"{t.get('peak_dbfs', float('nan')):.1f}", "-" if snr is None else f"{snr:.0f} dB",
                "yes" if t.get("use", True) else "no", t.get("notes", "")))

    def _selected(self):
        ids = {int(i) for i in self.tree.selection()}
        return [t for t in self.takes if t["id"] in ids]

    def toggle_selected(self):
        for t in self._selected():
            t["use"] = not t.get("use", True)
        self._save_session(); self._refresh_table(); self.analyse()

    def swap_selected(self):
        for t in self._selected():
            t["label"] = "com" if t["label"] == "sem" else "sem"
        self._save_session(); self._refresh_table(); self.analyse()

    def delete_selected(self):
        sel = self._selected()
        if not sel or not messagebox.askyesno("Delete", f"Delete {len(sel)} take(s)? The audio files are kept on disk."):
            return
        ids = {t["id"] for t in sel}
        self.takes = [t for t in self.takes if t["id"] not in ids]
        self._save_session(); self._refresh_table(); self.analyse()

    # ------------------------------------------------------------------ analysis
    def _analysis_params(self):
        frac = {"1/24": 24, "1/12": 12, "1/6": 6, "1/3": 3}[self.v_smooth.get()]
        try:
            a = float(self.v_a.get().replace(",", ".")) if self.v_a.get().strip() else None
        except ValueError:
            a = None
        try:
            prom = max(float(self.v_prom.get()), 0.5)
        except (tk.TclError, ValueError):
            prom = 3.0
        return dict(frac=frac, win=max(float(self.v_win.get()), 2.0), a=a, rel=bool(self.v_rel.get()), prom=prom)

    def analyse(self, then=None):
        if not self.session_dir:
            return
        params = self._analysis_params()
        takes = [dict(t) for t in self.takes]
        self._save_session()
        self.v_status.set("Analysing…")
        self.gen = getattr(self, "gen", 0) + 1
        params["gen"] = self.gen
        threading.Thread(target=self._analyse_worker, args=(takes, params, then), daemon=True).start()

    def _take_tf(self, t, win):
        key = (t["id"], t["label"], win)
        if key in self.cache:
            return self.cache[key]
        y, fs = load_audio(os.path.join(self.session_dir, "audio", t["file"]))
        s = t["sweep"]
        f2 = min(s["f2"], 0.49 * fs)
        if s.get("signal") == "noise":
            f, P, N = noise_spectrum(y, fs)
            if N is None:   # no silent part: SNR unknown, treated as OK
                N = np.full_like(P, np.nan)
            self.cache[key] = (f, P, N, fs, s["f1"], f2)
            return self.cache[key]
        x = make_sweep(s["f1"], f2, s["T"], fs)
        h = impulse_response(y, x, fs, s["f1"], f2)
        f, P, N, tpk = transfer_function(h, fs, len(y), len(x), win)
        self.cache[key] = (f, P, N, fs, s["f1"], f2)
        return self.cache[key]

    def _analyse_worker(self, takes, prm, then):
        with self.analysis_lock:
            self._analyse_worker_locked(takes, prm, then)

    def _analyse_worker_locked(self, takes, prm, then):
        try:
            used = [t for t in takes if t.get("use", True)]
            if not takes:
                self.work_q.put(("analysis", None)); return
            if prm["gen"] != self.gen:
                return  # superseded before it started
            per = []
            fmin = max(t["sweep"]["f1"] for t in takes) * 1.5
            fmax = min(min(t["sweep"]["f2"], 0.45 * t["fs"]) for t in takes)
            fgrid = np.geomspace(fmin, fmax, 900)
            for t in takes:
                f, P, N, fs, f1, f2 = self._take_tf(t, prm["win"])
                Ps = frac_oct(f, P, prm["frac"], fgrid)
                Ns = frac_oct(f, N, prm["frac"], fgrid)
                snr = db(Ps / Ns)
                band = (fgrid > 200) & (fgrid < 10000)
                t["snr_median"] = float(np.median(snr[band])) if band.any() else float(np.median(snr))
                per.append(dict(take=t, H=db(Ps), snr=snr))
            res = dict(fgrid=fgrid, per=per, prm=prm)
            groups = {}
            for lab in ("com", "sem"):
                Hs = np.array([p["H"] for p in per if p["take"]["label"] == lab and p["take"].get("use", True)])
                S = np.array([p["snr"] for p in per if p["take"]["label"] == lab and p["take"].get("use", True)])
                if len(Hs):
                    groups[lab] = dict(mean=Hs.mean(0), std=Hs.std(0, ddof=1) if len(Hs) > 1 else np.full(len(fgrid), np.nan),
                                       n=len(Hs), snr_min=S.min(0))
            res["groups"] = groups
            if "com" in groups and "sem" in groups:
                g1, g0 = groups["com"], groups["sem"]
                d = g1["mean"] - g0["mean"]
                unc = np.sqrt(g1["std"] ** 2 / g1["n"] + g0["std"] ** 2 / g0["n"])
                valid = ~(np.minimum(g1["snr_min"], g0["snr_min"]) <= 10)   # NaN (SNR unknown) counts as valid
                med = float(np.median(d[valid])) if valid.any() else 0.0
                if prm["rel"]:
                    d = d - med
                res.update(diff=d, unc=unc, valid=valid, median=med)
                dd = np.where(valid, d, np.nanmax(d))
                pk, pr = find_peaks(-dd, prominence=prm["prom"])
                dips = []
                for k, p in zip(pk, pr["prominences"]):
                    if not valid[k]:
                        continue
                    u = unc[k]
                    dips.append(dict(f=float(fgrid[k]), v=float(d[k]), prom=float(p),
                                     unc=None if not np.isfinite(u) else float(u)))
                dips.sort(key=lambda x: x["v"])
                res["dips"] = dips[:15]
            # write SNR back to the real take dicts
            snr_by_id = {p["take"]["id"]: p["take"]["snr_median"] for p in per}
            for t in self.takes:
                if t["id"] in snr_by_id:
                    s = snr_by_id[t["id"]]
                    t["snr_median"] = round(s, 1) if np.isfinite(s) else None
            res["then"] = then
            self.work_q.put(("analysis", res))
        except Exception as exc:
            self.work_q.put(("status", f"Analysis error: {exc}"))
            if then:
                self.after(0, then)

    def _fmt_axis(self, ax, fgrid):
        ax.set_xscale("log"); ax.set_xlim(fgrid[0], fgrid[-1])
        ticks = [t for t in [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000] if fgrid[0] <= t <= fgrid[-1]]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True, color="#e3e3df", lw=0.8)
        ax.set_xlabel("Frequency (Hz)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def _shade_invalid(self, ax, fgrid, valid):
        inv = (~valid).astype(int)
        edges = np.flatnonzero(np.diff(np.r_[0, inv, 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(fgrid[a], fgrid[min(b, len(fgrid) - 1)], color="#999", alpha=0.12, lw=0)

    def _show_analysis(self, res):
        if res and res["prm"].get("gen") != getattr(self, "gen", 0):
            if res.get("then"):
                self.after(300, res["then"])
            return  # a newer analysis is on its way
        self.last_result = res
        self._refresh_table()
        self._save_session()
        ax = self.ax_spec; ax.clear()
        ax2 = self.ax_diff; ax2.clear()
        self.dips_txt.delete("1.0", "end")
        if not res:
            self.fig_spec._canvas.draw_idle(); self.fig_diff._canvas.draw_idle()
            self.v_status.set("No takes yet."); return
        fg, prm = res["fgrid"], res["prm"]
        for p in res["per"]:
            t = p["take"]
            col = COL_WITH if t["label"] == "com" else COL_WITHOUT
            ax.plot(fg, p["H"], color=col, lw=0.8, alpha=0.35 if t.get("use", True) else 0.08)
        for lab, col in (("com", COL_WITH), ("sem", COL_WITHOUT)):
            g = res["groups"].get(lab)
            if g:
                ax.plot(fg, g["mean"], color=col, lw=2.2, label=f"{LABELS[lab]} - mean of {g['n']}")
        ax.set_ylabel("|H(f)|² (dB, arbitrary reference)")
        sweeps = any(p["take"]["sweep"].get("signal", "sweep") == "sweep" for p in res["per"])
        ax.set_title(f"Transfer function per take (thin) and mean (thick) · {self.v_smooth.get()} octave" +
                     (f" · IR window {prm['win']:g} ms" if sweeps else " · white noise (FFT)"),
                     loc="left", fontsize=10)
        self._fmt_axis(ax, fg)
        allH = np.concatenate([p["H"] for p in res["per"]])
        top = np.nanpercentile(allH, 99.5)
        ax.set_ylim(top - 70, top + 5)
        if res["groups"]:
            ax.legend(frameon=False)
        self.fig_spec._canvas.draw_idle()

        if "diff" in res:
            d, u, v = res["diff"], res["unc"], res["valid"]
            self._shade_invalid(ax2, fg, v)
            ax2.axhline(0, color="#888", lw=1)
            if np.isfinite(u).any():
                ax2.fill_between(fg, d - u, d + u, color=COL_DIFF, alpha=0.18, lw=0, label="± 1 standard error (between takes)")
            ax2.plot(fg, d, color=COL_DIFF, lw=2, label="with − without")
            for dp in res["dips"]:
                strong = dp["prom"] >= 6   # strong dips: filled marker; weaker ones: hollow and grey label
                ax2.plot(dp["f"], dp["v"], "o", ms=7 if strong else 5, mfc=COL_DIFF if strong else "white",
                         mec=COL_DIFF if not strong else "white", mew=1.5)
                ax2.annotate(f"{dp['f'] / 1000:.2f} kHz", (dp["f"], dp["v"]), textcoords="offset points",
                             xytext=(0, -16), ha="center", fontsize=8, color="#14171a" if strong else "#777")
            n1, n0 = res["groups"]["com"]["n"], res["groups"]["sem"]["n"]
            ttl = "Relative transmission (difference minus its median)" if prm["rel"] else "Difference with − without"
            ax2.set_title(f"{ttl} · {n1} with / {n0} without · median {res['median']:+.1f} dB · grey = SNR < 10 dB",
                          loc="left", fontsize=10)
            ax2.set_ylabel("dB")
            lim = np.nanpercentile(np.abs(d[v]), 99) if v.any() else 20
            ax2.set_ylim(-max(lim, 10) - 5, max(lim, 10) + 5)
            if prm["a"]:
                fb = C_SOUND / (2 * prm["a"] / 100)
                if fg[0] < fb < fg[-1]:
                    ax2.axvline(fb, color="#14171a", lw=1.4)
                    ax2.annotate(f"c/2a = {fb / 1000:.2f} kHz", (fb, ax2.get_ylim()[1]), textcoords="offset points",
                                 xytext=(5, -14), fontsize=9)
            self._fmt_axis(ax2, fg)
            ax2.legend(frameon=False, loc="upper left", fontsize=9)
            lines = [f"Deepest dips (prominence ≥ {prm['prom']:g} dB, SNR ok; hollow markers = prominence < 6 dB):"]
            for dp in res["dips"]:
                u = f" ± {dp['unc']:.1f}" if dp["unc"] is not None else " (repeat takes for an error bar)"
                sig = ""
                if dp["unc"]:
                    sig = "  ← significant" if dp["prom"] > 3 * dp["unc"] else "  (within the spread)"
                lines.append(f"  {dp['f']:7.0f} Hz   {dp['v']:+6.1f}{u} dB   prominence {dp['prom']:.1f} dB   "
                             f"a = c/2f ≈ {100 * C_SOUND / (2 * dp['f']):.1f} cm{sig}")
            if len(lines) == 1:
                lines.append("  none")
            self.dips_txt.insert("1.0", "\n".join(lines))
            self.v_status.set(f"Analysis updated: {n1} takes with crystal, {n0} without.")
        else:
            ax2.text(0.5, 0.5, "Record at least one take WITH and one WITHOUT the crystal.",
                     transform=ax2.transAxes, ha="center", va="center", color="#666")
            ax2.set_axis_off()
            self.v_status.set("Analysis updated. Need takes of both kinds for the difference.")
        self.fig_diff._canvas.draw_idle()
        if res.get("then"):
            self.after(300, res["then"])

    def _live_init(self):
        """Set up the 'Last recording' tab as a live view of the take being recorded."""
        p = self.pending
        fs = p["fs"]
        span = 10.0 if p["manual"] else p["reclen"]
        fig = self.fig_last; fig.clear()
        a1 = fig.add_subplot(211); a2 = fig.add_subplot(212)
        a1.set_title(f"Recording… {LABELS[p['label']]} · mic: {p['mic']}", loc="left", fontsize=10)
        a1.set_xlim(0, span); a1.set_ylim(-1, 1)
        a1.set_xlabel("time (s)"); a1.set_ylabel("amplitude")
        for v in (-1, 1):
            a1.axhline(v, color="#d93025", lw=0.6, ls=":")
        if p["mode"] == "play":
            a1.axvspan(p["t_start"], p["t_start"] + p["sweep"]["T"], color="#2fa84f", alpha=0.08)
        # moving artists are animated: drawn by blitting over a cached background, which
        # keeps refreshes cheap (a full redraw takes hundreds of ms on slow machines)
        self.live_wave, = a1.plot([], [], color="#444", lw=0.5, animated=True)
        self.live_cursor = a1.axvline(0, color="#1a73e8", lw=1, animated=True)
        a2.set_xscale("log"); a2.set_xlim(20, fs / 2); a2.set_ylim(-120, 0)
        a2.set_xlabel("frequency (Hz)"); a2.set_ylabel("dBFS"); a2.grid(True, which="both", alpha=0.3)
        self.live_spec_hold, = a2.plot([], [], color="#e0a800", lw=0.8, label="max hold", animated=True)
        self.live_spec, = a2.plot([], [], color="#1a73e8", lw=1.0, label="now", animated=True)
        a2.legend(loc="upper right", fontsize=8)
        self.live_artists = (self.live_wave, self.live_cursor, self.live_spec_hold, self.live_spec)
        self.live_nfft = 8192 if fs >= 32000 else 4096
        self.live_win = np.hanning(self.live_nfft).astype(np.float32)
        self.live_f = np.fft.rfftfreq(self.live_nfft, 1 / fs)
        self.live_hold = np.full(len(self.live_f), -200.0)
        self.live_ticks = 0
        self.live_bg = None
        if getattr(self, "live_cid", None) is None:
            # a full redraw (first show, window resize, axis change) re-caches the background
            self.live_cid = self.canvas_last.mpl_connect("draw_event", self._live_on_draw)
        self.nb.select(self.tab_last)
        self.canvas_last.draw_idle()

    def _live_on_draw(self, _event):
        if not self.recording:
            return
        c = self.canvas_last
        self.live_bg = c.copy_from_bbox(self.fig_last.bbox)
        for a in self.live_artists:
            self.fig_last.draw_artist(a)

    def _live_stop(self):
        """Leave the live view as a normal static figure (e.g. when the take is discarded)."""
        if getattr(self, "live_cid", None) is not None:
            self.canvas_last.mpl_disconnect(self.live_cid)
            self.live_cid = None
        for a in getattr(self, "live_artists", ()):
            a.set_animated(False)
        self.live_bg = None
        self.canvas_last.draw_idle()

    def _live_update(self):
        p = self.pending
        fs = p["fs"]
        buf = self.recorder.buf
        pos = min(self.recorder.pos, len(buf))
        if pos < 2:
            return
        el = pos / fs
        ax = self.live_wave.axes
        if el > ax.get_xlim()[1]:
            ax.set_xlim(0, ax.get_xlim()[1] * 2)       # manual mode: grow the time axis
            self.live_bg = None
        # min/max envelope, ~1500 bins, so the plot stays cheap for long takes
        y = buf[:pos]
        nb = min(1500, pos)
        step = pos // nb
        blk = y[:nb * step].reshape(nb, step)
        env = np.empty(2 * nb, np.float32)
        env[0::2] = blk.min(axis=1); env[1::2] = blk.max(axis=1)
        tb = np.repeat(np.arange(nb) * step / fs, 2)
        self.live_wave.set_data(tb, env)
        self.live_cursor.set_xdata([el, el])
        n = self.live_nfft
        if pos >= n:
            seg = buf[pos - n:pos] * self.live_win
            spec = 20 * np.log10(np.abs(np.fft.rfft(seg)) * 2 / self.live_win.sum() + 1e-12)
            np.maximum(self.live_hold, spec, out=self.live_hold)
            self.live_spec.set_data(self.live_f[1:], spec[1:])
            self.live_spec_hold.set_data(self.live_f[1:], self.live_hold[1:])
        c = self.canvas_last
        if self.live_bg is None:
            c.draw()                                    # full redraw; _live_on_draw caches the background
        else:
            c.restore_region(self.live_bg)
            for a in self.live_artists:
                self.fig_last.draw_artist(a)
        c.blit(self.fig_last.bbox)

    def _plot_last(self, y, fs, take):
        fig = self.fig_last; fig.clear()
        a1 = fig.add_subplot(211); a2 = fig.add_subplot(212)
        t = np.arange(len(y)) / fs
        a1.plot(t, y, color="#444", lw=0.4)
        a1.set_title(f"Take {take['id']} · {LABELS[take['label']]} · mic: {take['mic']} · peak {take['peak_dbfs']:.1f} dBFS",
                     loc="left", fontsize=10)
        a1.set_ylabel("amplitude"); a1.set_xlim(0, t[-1])
        f, tt, S = signal.spectrogram(y, fs, nperseg=2048, noverlap=1536)
        Z = 10 * np.log10(S + 1e-20)
        a2.pcolormesh(tt, f, Z, shading="auto", cmap="magma", vmin=Z.max() - 80, vmax=Z.max())
        a2.set_yscale("log"); a2.set_ylim(30, fs / 2); a2.set_xlabel("time (s)"); a2.set_ylabel("Hz")
        self.canvas_last.draw_idle()

    # ------------------------------------------------------------------ export
    def export(self):
        res = self.last_result
        if not res or not self.session_dir:
            messagebox.showinfo("Export", "Nothing analysed yet."); return
        out = os.path.join(self.session_dir, "results")
        os.makedirs(out, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fg = res["fgrid"]
        cols = {"f_Hz": fg}
        for lab in ("com", "sem"):
            g = res["groups"].get(lab)
            if g:
                cols[f"mean_{lab}_dB"] = g["mean"]; cols[f"std_{lab}_dB"] = g["std"]
        if "diff" in res:
            cols["diff_dB"] = res["diff"]; cols["stderr_dB"] = res["unc"]; cols["valid"] = res["valid"].astype(int)
        for p in res["per"]:
            cols[f"take{p['take']['id']}_{p['take']['label']}_dB"] = p["H"]
        names = list(cols)
        arr = np.column_stack([cols[n] for n in names])
        np.savetxt(os.path.join(out, f"results_{stamp}.csv"), arr, delimiter=",", header=",".join(names),
                   comments="", fmt="%.4f")
        self.fig_spec.savefig(os.path.join(out, f"transfer_{stamp}.png"), dpi=200)
        self.fig_diff.savefig(os.path.join(out, f"difference_{stamp}.png"), dpi=200)
        with open(os.path.join(out, f"dips_{stamp}.txt"), "w") as fh:
            fh.write(self.dips_txt.get("1.0", "end"))
        self.v_status.set(f"Exported to {out}")

    def _on_close(self):
        try:
            self.monitor.stop()
            if self.recording:
                self.recorder.abort()
        finally:
            self.destroy()


if __name__ == "__main__":
    App().mainloop()
