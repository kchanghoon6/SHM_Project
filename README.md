# SENTRY: Low-Cost Bridge Vibration Anomaly Detection

A low-cost screening system for small and medium bridges. A MEMS accelerometer and a temperature/humidity sensor record vibration and environment together; the part of the vibration change that temperature and humidity can explain is removed by regression, and an autoencoder trained only on normal data flags what is left as an anomaly candidate, together with which vibration feature drifted.

Built for **Korea Code Fair 2026** (team SENTRY: Changhoon Kim, Seung Choi; Cheongshim International Academy).

- Core hardware: ADXL355Z 3-axis accelerometer, DHT22 temp/humidity sensor, Raspberry Pi 5 (about KRW 190,000 for the core parts)
- Language: Python (NumPy, pandas, scikit-learn, SciPy, Flask)

## Why

Natural frequencies of a bridge shift with temperature and humidity even when nothing is damaged, and an in-service bridge cannot be damaged on purpose to collect labeled anomaly data. So the pipeline (1) learns the normal environment-to-vibration relationship, (2) works on the residuals, and (3) trains and sets its threshold on normal data only.

## Pipeline

1. **Acquisition**: 3-axis acceleration (200 Hz in v1, 500 Hz from v2) and temperature/humidity written to one CSV on the Pi.
2. **Features**: per window, DC removal, spectrum (FFT in v1, Welch from v2), then RMS, peak-to-peak, peak frequency/power, total energy and band-energy ratios. From v2, amplitudes are log-scaled and band energies are ratios so that the features reflect spectral shape, not hit strength.
3. **Environmental compensation**: regress each feature on temperature T and relative humidity RH using normal data; use the residual `r = f - g(T, RH)`.
4. **Anomaly score**: an autoencoder trained on standardized normal residuals; reconstruction error above the 99th percentile of a held-out normal set is an anomaly candidate.
5. **Explanation**: per-feature / per-band reconstruction error is reported so the alarm says which characteristic drifted.

## Versions

| Folder | Stage | What changed |
|---|---|---|
| `v1/` | Foam-board model, manual excitation, Raspberry Pi 3 A+ | First end-to-end pipeline. PCA reconstruction (equivalent to a linear autoencoder), 4 s windows, comparison of no / temperature / temperature+humidity compensation. Also early tests on public bridge datasets. |
| `v2/` | Aluminum multi-span model, solenoid auto-tapper, Raspberry Pi 5 | Fixed v1's two weaknesses (blurry resonance peak, inconsistent manual hits). Tap time and acceleration share one clock; only the free-vibration window after each tap (t0+0.05 s to 2.0 s) is analyzed. Nonlinear 8-3-8 tanh autoencoder with early stopping, strict train / threshold / test split, real-time web dashboard. |
| `v3/` | Two real pedestrian bridges | Continuous ambient-vibration logging (no tapper) with an added DS18B20 deck-surface temperature sensor for unattended field runs, plus an on-site node monitor that raises a theft/movement alarm for the unattended sensor node. Anomaly detection on field data was run offline in batch. |

### Files

**v1**
- `logger.py`: ADXL355 (I2C) + DHT22 logger to CSV.
- `bridge_anomaly_demo.py`: feature extraction, T/RH compensation, PCA reconstruction score, band contributions. `--generate-sample` creates synthetic data to try it without hardware.
- `make_report_artifacts.py`: per-condition summary tables and SVG plots from the anomaly results.
- `z24_benchmark_demo.py`: PCA reconstruction baseline on the Z24 Bridge Benchmark (KU Leuven).
- `openlab_demo.py`: early public-data trial on tracked modal frequencies (KW51 `trackedmodes.mat`, read with a small built-in MAT v5 reader) with environmental compensation and a NumPy autoencoder.

**v2**
- `logger3.py`: 500 Hz logger that also drives the solenoid tapper (warm-up taps, fixed period) and records tap events.
- `sentry.py`: all-in-one tool with `train`, `evaluate`, `dashboard` and `run` (logger + live dashboard) subcommands.
- `core.py`, `run_core.py`: the same feature / compensation / autoencoder logic as a reusable module with a `train` / `score` CLI.

**v3**
- `logger4.py`: continuous field logger (ADXL355 at 500 Hz, ±2 g, DHT22, DS18B20 surface temperature, ADXL internal temperature).
- `logger.py`: long-run version of the field logger. Splits the CSV every hour (`--rotate-seconds`), writes on a separate thread with periodic fsync, keeps a `current.json` manifest for the dashboard, never overwrites existing files, and writes a QC summary (missed slots, read errors, clipping) on exit.
- `sentry.py`: on-site node monitor dashboard (vibration RMS, peak, tilt change with calibration and vote-based alarm). In live mode it launches `logger.py` itself.

## Running

```bash
pip install numpy pandas scikit-learn scipy joblib flask
# on the Raspberry Pi, for the loggers:
pip install smbus2 adafruit-circuitpython-dht adafruit-blinka gpiozero
```

```bash
# v1: try the pipeline on synthetic data (no hardware needed)
cd v1
python bridge_anomaly_demo.py --generate-sample
python make_report_artifacts.py

# v2: train on normal runs, evaluate, open the dashboard
cd v2
python sentry.py train --bridge normal_*_bridge_log.csv
python sentry.py evaluate --normal <normal csv...> --anomaly <damage csv...>
python sentry.py dashboard --bridge <bridge_log.csv>
# live run on the Pi (logger + tapper + dashboard)
python sentry.py run --logger logger3.py --condition T0-H0-M0-B0

# v3: field logging on the Pi (30 min)
cd v3
python logger4.py --condition B31-SCR-A --duration-s 1800
# hourly-rotating logger + node monitor dashboard (live)
python sentry.py
# node monitor on an existing recording
python sentry.py --bridge data/B31-SCR-A_bridge_log.csv
```

Raw experiment data is not included in this repository.

## Results (from the final report)

- **Aluminum model, 27 conditions (environment x mass x bolt loosening)**: the compensated model detected all 840 damaged windows (recall 100%) with 4 false alarms out of 90 normal windows (4.4%), and band contributions separated bolt loosening from mass change.
- **openLAB public dataset (TU Dresden, 45 m research bridge)**: with frequency channels combined, the autumn false-alarm rate of 26% dropped to 0-3% with temperature compensation.
- **Two real footbridges**: the modal peak was tracked within 1%; a 200 kg load lowered the frequency by 0.6% and it recovered after removal. The alarm stayed silent when unloaded and fired only when loaded.

## Reports (English translations)

- [Final-round technical report](https://kchanghoon6.github.io/portfolio/docs/sentry/r3-en.pdf) / [summary](https://kchanghoon6.github.io/portfolio/docs/sentry/s3-en.pdf)
- [Preliminary-round report](https://kchanghoon6.github.io/portfolio/docs/sentry/r2-en.pdf) / [summary](https://kchanghoon6.github.io/portfolio/docs/sentry/s2-en.pdf)
- [First-round report](https://kchanghoon6.github.io/portfolio/docs/sentry/r1-en.pdf) / [summary](https://kchanghoon6.github.io/portfolio/docs/sentry/s1-en.pdf)
- [Poster](https://kchanghoon6.github.io/portfolio/docs/sentry/poster-en.pdf)
- [Project page](https://kchanghoon6.github.io/portfolio/pages/sentry.html)
