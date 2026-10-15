# Phononic Crystal Lab

Desktop app (Python + Tkinter) for measuring the acoustic transmission of a phononic (sonic) crystal: record takes WITH and WITHOUT the crystal, compare their spectra, and find the dips (band gaps).

## Install and run

```
pip install numpy scipy matplotlib sounddevice soundfile
python phononic_lab.py
```

`ffmpeg` is only needed to import m4a/mp3 files.

## Test signals

- **White noise (default):** played from another device, e.g. a phone; the app only records. Each take's spectrum is |FFT|² of the recording, smoothed in fractional-octave bands. The noise spectrum and the phone's speaker cancel in WITH − WITHOUT. Leave ~1 s of silence at the start of each take: it's used as the background-noise reference for the SNR.
- **Sine sweep:** played from the computer (sample-synchronous) or from an external device. The app deconvolves the recording into an impulse response and time-windows it, which removes room reflections and harmonic distortion.

## Analysis

- Mean and spread of the WITH and WITHOUT takes; the difference in dB with a standard-error band.
- Frequencies with SNR < 10 dB are greyed out.
- Dip detection with an adjustable minimum prominence ("Min dip (dB)"). Each dip gets a significance check against the spread between takes and the lattice constant a = c/2f that would put a Bragg gap there.
- Optional Bragg marker at f = c/2a: enter the lattice constant in cm.

Each session is saved in its own folder (WAV files, `session.json`); **Export** writes CSV, PNG and the list of dips.

## Measurement tips

- Turn off Windows audio enhancements / noise suppression for the microphone. It removes steady noise such as white noise.
- Keep the source, the microphone and the volume fixed between takes; record several alternating pairs (the **Run protocol** button).
- Use an uncompressed white-noise file that is flat up to the highest frequency you want to analyse.
