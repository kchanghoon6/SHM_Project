from __future__ import annotations

import argparse
import io
import json
import math
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify
from scipy.signal import welch
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HTML = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SENTRY</title>
<style>
body{margin:0;background:#10141b;color:#eef2f7;font-family:Arial,sans-serif}.wrap{max-width:1200px;margin:auto;padding:20px}.top{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}.title{font-size:28px;font-weight:700}.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.card{background:#1a202b;border-radius:14px;padding:16px;box-shadow:0 6px 18px #0004}.wide{grid-column:span 3}.full{grid-column:span 6}.label{font-size:13px;color:#9ca8b8}.value{font-size:28px;font-weight:700;margin-top:6px}.status{padding:10px 18px;border-radius:999px;font-weight:700}.normal{background:#176b45}.warning{background:#916b12}.anomaly{background:#9d2c2c}.waiting{background:#465164}canvas{width:100%;height:250px;background:#111722;border-radius:10px}.bands{display:flex;height:26px;border-radius:8px;overflow:hidden;margin-top:12px}.band{display:flex;align-items:center;justify-content:center;min-width:0;font-size:12px}.b1{background:#3f82d3}.b2{background:#c98b2e}.b3{background:#bd4f59}.small{font-size:12px;color:#9ca8b8;margin-top:8px}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.wide,.full{grid-column:span 2}}
</style>
</head>
<body>
<div class="wrap">
<div class="top"><div><div class="title">SENTRY 실시간 대시보드</div><div id="connection" class="small">서버 연결 확인 중</div></div><div id="status" class="status waiting">대기 중</div></div>
<div class="grid">
<div class="card"><div class="label">이상 점수 / 임계값</div><div id="score" class="value">-</div></div>
<div class="card"><div class="label">피크 주파수</div><div id="peak" class="value">-</div></div>
<div class="card"><div class="label">RMS</div><div id="rms" class="value">-</div></div>
<div class="card"><div class="label">온도</div><div id="temp" class="value">-</div></div>
<div class="card"><div class="label">습도</div><div id="hum" class="value">-</div></div>
<div class="card"><div class="label">누적 이상 후보</div><div id="detection" class="value">-</div></div>
<div class="card wide"><div class="label">최근 z축 파형</div><canvas id="wave"></canvas></div>
<div class="card wide"><div class="label">최근 타격 주파수 스펙트럼</div><canvas id="spec"></canvas></div>
<div class="card full"><div class="label">대역별 재구성 오차 기여도</div><div class="bands"><div id="b1" class="band b1"></div><div id="b2" class="band b2"></div><div id="b3" class="band b3"></div></div><div id="info" class="small">-</div></div>
</div>
</div>
<script>
function line(canvas,x,y){const c=document.getElementById(canvas),d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;const g=c.getContext('2d');g.scale(d,d);g.clearRect(0,0,w,h);g.strokeStyle='#2c3544';g.beginPath();for(let i=1;i<5;i++){let yy=h*i/5;g.moveTo(0,yy);g.lineTo(w,yy)}g.stroke();if(!y||y.length<2)return;let ymin=Math.min(...y),ymax=Math.max(...y);if(ymax===ymin){ymax+=1;ymin-=1}g.strokeStyle='#73b7ff';g.lineWidth=2;g.beginPath();for(let i=0;i<y.length;i++){let px=i/(y.length-1)*w,py=h-(y[i]-ymin)/(ymax-ymin)*h;if(i===0)g.moveTo(px,py);else g.lineTo(px,py)}g.stroke()}
function ratioText(v){if(v==null)return '-';if(v>=1000)return v.toExponential(2)+'×';return v.toFixed(2)+'×'}
async function update(){const conn=document.getElementById('connection');try{const r=await fetch('/api',{cache:'no-store'});if(!r.ok)throw new Error('api');const s=await r.json();const st=document.getElementById('status');st.textContent=s.status_text;st.className='status '+s.status_class;document.getElementById('score').textContent=ratioText(s.score_ratio);document.getElementById('peak').textContent=s.peak_hz==null?'-':s.peak_hz.toFixed(2)+' Hz';document.getElementById('rms').textContent=s.rms==null?'-':s.rms.toFixed(4)+' g';document.getElementById('temp').textContent=s.temp_c==null?'-':s.temp_c.toFixed(1)+' °C';document.getElementById('hum').textContent=s.humidity==null?'-':s.humidity.toFixed(1)+' %';document.getElementById('detection').textContent=s.evaluated_count?String(s.anomaly_count)+' / '+String(s.evaluated_count):'-';line('wave',s.wave_x,s.wave_y);line('spec',s.freq_x,s.freq_y);const b=s.bands||[0,0,0],ids=['b1','b2','b3'];for(let i=0;i<3;i++){const e=document.getElementById(ids[i]);e.style.width=(b[i]*100)+'%';e.textContent=b[i]>0.08?Math.round(b[i]*100)+'%':''}document.getElementById('info').textContent=s.info||'-';if(s.updated_at==null){conn.textContent='서버 연결됨 · 데이터 대기'}else{const age=Math.max(0,Date.now()/1000-s.updated_at);conn.textContent=age<3?'서버 연결됨 · 실시간 수집 중':'서버 연결됨 · 마지막 데이터 '+age.toFixed(1)+'초 전'}}catch(e){conn.textContent='서버 연결 끊김';const st=document.getElementById('status');st.textContent='연결 끊김';st.className='status waiting'}}
setInterval(update,700);update();
</script>
</body></html>'''

RAW_FEATURES = ["rms", "std", "p2p", "peak_hz", "peak_power", "energy_1", "energy_2", "energy_3"]
FEATURES = ["log_rms", "log_p2p", "peak_hz", "log_peak_power", "log_total_energy", "band_frac_1", "band_frac_2", "band_frac_3"]
FILTER_LPF = {"0x03": 125.0, "0x04": 62.5, "0x05": 31.25}
EPS = 1e-12


def prepare_features(raw: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(raw, pd.DataFrame):
        x = raw[RAW_FEATURES].to_numpy(float)
    else:
        x = np.asarray(raw, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
    total = x[:, 5] + x[:, 6] + x[:, 7] + EPS
    return np.column_stack([
        np.log10(np.maximum(x[:, 0], EPS)),
        np.log10(np.maximum(x[:, 2], EPS)),
        x[:, 3],
        np.log10(np.maximum(x[:, 4], EPS)),
        np.log10(total),
        x[:, 5] / total,
        x[:, 6] / total,
        x[:, 7] / total,
    ])


def model_features(model: dict, raw: pd.DataFrame | np.ndarray) -> np.ndarray:
    if model.get("feature_version") == "log_normalized_v1":
        return prepare_features(raw)
    if isinstance(raw, pd.DataFrame):
        return raw[RAW_FEATURES].to_numpy(float)
    x = np.asarray(raw, dtype=float)
    return x.reshape(1, -1) if x.ndim == 1 else x


def companion_paths(bridge: Path) -> tuple[Path, Path]:
    name = bridge.name
    if not name.endswith("_bridge_log.csv"):
        raise ValueError("bridge CSV 이름은 *_bridge_log.csv 형식이어야 합니다.")
    base = name[:-len("_bridge_log.csv")]
    return bridge.with_name(base + "_tap_events.csv"), bridge.with_name(base + "_metadata.json")


def metadata_for(bridge: Path) -> dict:
    _, meta = companion_paths(bridge)
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {"sample_rate_hz": 500, "range_g": 8, "filter_register_hex": "0x03"}


def limits(meta: dict) -> tuple[float, float, list[float]]:
    fs = float(meta.get("sample_rate_hz", 500))
    lpf = FILTER_LPF.get(str(meta.get("filter_register_hex", "0x03")).lower(), min(fs / 2, 125.0))
    top = min(fs / 2, lpf)
    if top >= 120:
        bands = [0.0, 40.0, 80.0, top]
    else:
        bands = [0.0, top / 3, top * 2 / 3, top]
    return fs, top, bands


def read_taps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pulse_start_perf_ns" in df.columns:
        df["tap_ns"] = pd.to_numeric(df["pulse_start_perf_ns"], errors="coerce")
    elif "t_perf_ns" in df.columns:
        df["tap_ns"] = pd.to_numeric(df["t_perf_ns"], errors="coerce")
    else:
        raise ValueError("tap CSV에 pulse_start_perf_ns 또는 t_perf_ns가 필요합니다.")
    return df.dropna(subset=["tap_ns"]).reset_index(drop=True)


def extract_one(seg: pd.DataFrame, fs: float, top: float, bands: list[float]) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    z = pd.to_numeric(seg["az_g"], errors="coerce").to_numpy(float)
    z = z[np.isfinite(z)]
    if len(z) < max(64, int(fs * 0.5)):
        raise ValueError("분석 구간의 유효 샘플이 부족합니다.")
    z = z - np.mean(z)
    rms = float(np.sqrt(np.mean(z * z)))
    std = float(np.std(z))
    p2p = float(np.ptp(z))
    nperseg = min(len(z), max(256, int(fs)))
    freq, power = welch(z, fs=fs, nperseg=nperseg)
    valid = (freq >= 1.0) & (freq <= top)
    if not np.any(valid):
        raise ValueError("유효 주파수 대역이 없습니다.")
    vf, vp = freq[valid], power[valid]
    idx = int(np.argmax(vp))
    peak_hz = float(vf[idx])
    peak_power = float(vp[idx])
    energies = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        m = (freq >= lo) & (freq < hi)
        energies.append(float(np.trapezoid(power[m], freq[m])) if np.any(m) else 0.0)
    vector = np.array([rms, std, p2p, peak_hz, peak_power, *energies], dtype=float)
    return vector, peak_hz, rms, vf, vp


def extract_events(bridge: Path, delay: float, window: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
    tap_path, _ = companion_paths(bridge)
    meta = metadata_for(bridge)
    fs, top, bands = limits(meta)
    data = pd.read_csv(bridge)
    taps = read_taps(tap_path)
    t = pd.to_numeric(data["t_perf_ns"], errors="coerce").to_numpy(float)
    rows, env = [], []
    for _, tap in taps.iterrows():
        start = float(tap["tap_ns"]) + delay * 1e9
        end = start + window * 1e9
        a, b = np.searchsorted(t, [start, end])
        if b - a < max(64, int(fs * window * 0.7)):
            continue
        seg = data.iloc[a:b]
        try:
            vector, _, _, _, _ = extract_one(seg, fs, top, bands)
        except ValueError:
            continue
        temp = float(pd.to_numeric(seg.get("temp_c"), errors="coerce").mean()) if "temp_c" in seg else math.nan
        hum = float(pd.to_numeric(seg.get("humidity"), errors="coerce").mean()) if "humidity" in seg else math.nan
        rows.append(vector)
        env.append([temp, hum])
    if not rows:
        raise RuntimeError("유효한 타격 구간을 찾지 못했습니다.")
    return pd.DataFrame(rows, columns=RAW_FEATURES), np.asarray(env, dtype=float), {"fs": fs, "top": top, "bands": bands}


def fill_env(env: np.ndarray, medians: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = env.copy()
    if medians is None:
        medians = np.nanmedian(x, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
    for j in range(x.shape[1]):
        x[~np.isfinite(x[:, j]), j] = medians[j]
    return x, medians


def train(args: argparse.Namespace) -> None:
    bridges = [Path(value) for value in args.bridge]
    parts = []
    file_peak_medians = []
    config = None
    requested = args.train_count + args.threshold_count + args.test_count
    for bridge in bridges:
        feat_part, env_part, part_config = extract_events(bridge, args.delay, args.window)
        if config is None:
            config = part_config
        elif part_config != config:
            raise RuntimeError("모든 정상 파일은 같은 샘플링 주파수와 분석 대역을 사용해야 합니다.")
        if len(feat_part) < requested:
            raise RuntimeError(f"{bridge.name}: 유효 정상 타격이 {requested}회보다 적습니다.")
        parts.append((bridge, feat_part.reset_index(drop=True), env_part))
        file_peak_medians.append((bridge.name, float(feat_part["peak_hz"].median())))

    train_raw = pd.concat([part.iloc[:args.train_count] for _, part, _ in parts], ignore_index=True)
    threshold_raw = pd.concat([part.iloc[args.train_count:args.train_count + args.threshold_count] for _, part, _ in parts], ignore_index=True)
    test_raw = pd.concat([part.iloc[args.train_count + args.threshold_count:requested] for _, part, _ in parts], ignore_index=True)
    train_env_raw = np.vstack([env[:args.train_count] for _, _, env in parts])
    threshold_env_raw = np.vstack([env[args.train_count:args.train_count + args.threshold_count] for _, _, env in parts])
    test_env_raw = np.vstack([env[args.train_count + args.threshold_count:requested] for _, _, env in parts])

    train_feat = prepare_features(train_raw)
    threshold_feat = prepare_features(threshold_raw)
    test_feat = prepare_features(test_raw)
    train_env, medians = fill_env(train_env_raw)
    threshold_env, _ = fill_env(threshold_env_raw, medians)
    test_env, _ = fill_env(test_env_raw, medians)
    spread = np.nanstd(train_env, axis=0)
    use_env = bool(spread[0] >= 0.2 or spread[1] >= 1.0)
    env_model = None
    if use_env:
        env_model = LinearRegression().fit(train_env, train_feat)
        train_feat = train_feat - env_model.predict(train_env)
        threshold_feat = threshold_feat - env_model.predict(threshold_env)
        test_feat = test_feat - env_model.predict(test_env)

    scaler = StandardScaler().fit(train_feat)
    x_train = scaler.transform(train_feat)
    x_threshold = scaler.transform(threshold_feat)
    ae = MLPRegressor(
        hidden_layer_sizes=(8, 3, 8),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        batch_size=min(32, len(x_train)),
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=args.seed,
    ).fit(x_train, x_train)
    threshold_error = np.mean((ae.predict(x_threshold) - x_threshold) ** 2, axis=1)
    threshold = float(np.quantile(threshold_error, 0.99))
    x_test = scaler.transform(test_feat)
    test_error = np.mean((ae.predict(x_test) - x_test) ** 2, axis=1)
    far = float(np.mean(test_error > threshold))
    model = {
        "features": FEATURES,
        "raw_features": RAW_FEATURES,
        "feature_version": "log_normalized_v1",
        "config": config,
        "delay": args.delay,
        "window": args.window,
        "env_medians": medians,
        "env_model": env_model,
        "environment_enabled": use_env,
        "environment_variant": "temp_hum",
        "scaler": scaler,
        "autoencoder": ae,
        "threshold": threshold,
        "train_count": len(train_raw),
        "threshold_count": len(threshold_raw),
        "test_count": len(test_raw),
        "normal_test_scores": test_error.tolist(),
        "baseline_files": [str(path) for path in bridges],
    }
    joblib.dump(model, args.model)
    print(f"모델 저장: {args.model}")
    print(f"정상 파일: {len(bridges)}개")
    print(f"타격 분할: 학습 {len(train_raw)}, 임계값 {len(threshold_raw)}, 정상 테스트 {len(test_raw)}")
    print(f"환경 보정: {'사용' if use_env else '미사용'}")
    print(f"정상 테스트 FAR: {far:.4f}")
    if len(file_peak_medians) > 1:
        peak_values = np.asarray([value for _, value in file_peak_medians], dtype=float)
        peak_gap = float(np.nanmax(peak_values) - np.nanmin(peak_values))
        peak_reference = float(np.nanmedian(peak_values))
        if peak_gap > max(5.0, peak_reference * 0.10):
            print("경고: 정상 파일 사이의 대표 피크 주파수 차이가 큽니다. 센서, 지지점, 태퍼, 구조 상태가 같은지 확인해야 합니다.")
            for name, value in file_peak_medians:
                print(f"  {name}: {value:.2f} Hz")


def score_features(model: dict, feat: pd.DataFrame, env: np.ndarray) -> np.ndarray:
    x = model_features(model, feat)
    env_filled, _ = fill_env(env, np.asarray(model["env_medians"]))
    if model["env_model"] is not None:
        x = x - model["env_model"].predict(env_filled)
    scaled = model["scaler"].transform(x)
    recon = model["autoencoder"].predict(scaled)
    return np.mean((recon - scaled) ** 2, axis=1)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def evaluation_row(model: dict, bridge: Path, label: str) -> tuple[dict, np.ndarray]:
    feat, env, _ = extract_events(bridge, model["delay"], model["window"])
    scores = score_features(model, feat, env)
    threshold = float(model["threshold"])
    ratios = scores / threshold
    detected = int(np.sum(scores > threshold))
    low, high = wilson_interval(detected, len(scores))
    row = {
        "condition": bridge.name.replace("_bridge_log.csv", ""),
        "label": label,
        "n": len(scores),
        "anomaly_count": detected,
        "detection_rate": detected / len(scores),
        "ci95_low": low,
        "ci95_high": high,
        "score_ratio_median": float(np.median(ratios)),
        "score_ratio_q1": float(np.quantile(ratios, 0.25)),
        "score_ratio_q3": float(np.quantile(ratios, 0.75)),
        "peak_hz_mean": float(feat["peak_hz"].mean()),
        "peak_hz_std": float(feat["peak_hz"].std(ddof=0)),
        "rms_mean": float(feat["rms"].mean()),
        "rms_std": float(feat["rms"].std(ddof=0)),
    }
    return row, scores


def evaluate(args: argparse.Namespace) -> None:
    model = joblib.load(args.model)
    rows = []
    normal_scores = []
    anomaly_scores = []
    stored_normal = np.asarray(model.get("normal_test_scores", []), dtype=float)
    if len(stored_normal):
        threshold = float(model["threshold"])
        detected = int(np.sum(stored_normal > threshold))
        low, high = wilson_interval(detected, len(stored_normal))
        ratios = stored_normal / threshold
        rows.append({
            "condition": "held_out_normal", "label": "normal", "n": len(stored_normal),
            "anomaly_count": detected, "detection_rate": detected / len(stored_normal),
            "ci95_low": low, "ci95_high": high,
            "score_ratio_median": float(np.median(ratios)),
            "score_ratio_q1": float(np.quantile(ratios, 0.25)),
            "score_ratio_q3": float(np.quantile(ratios, 0.75)),
            "peak_hz_mean": math.nan, "peak_hz_std": math.nan,
            "rms_mean": math.nan, "rms_std": math.nan,
        })
        normal_scores.extend(stored_normal.tolist())
    for value in args.normal:
        row, scores = evaluation_row(model, Path(value), "normal")
        rows.append(row)
        normal_scores.extend(scores.tolist())
    for value in args.anomaly:
        row, scores = evaluation_row(model, Path(value), "anomaly")
        rows.append(row)
        anomaly_scores.extend(scores.tolist())
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    if len(result):
        print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"결과 저장: {args.output}")
    if normal_scores and anomaly_scores:
        threshold = float(model["threshold"])
        normal_pred = np.asarray(normal_scores) > threshold
        anomaly_pred = np.asarray(anomaly_scores) > threshold
        tn, fp = int(np.sum(~normal_pred)), int(np.sum(normal_pred))
        tp, fn = int(np.sum(anomaly_pred)), int(np.sum(~anomaly_pred))
        precision = tp / (tp + fp) if tp + fp else math.nan
        recall = tp / (tp + fn) if tp + fn else math.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
        far = fp / (fp + tn) if fp + tn else math.nan
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"FAR={far:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")


def tail_dataframe(path: Path, rows: int) -> pd.DataFrame:
    with path.open("rb") as f:
        header = f.readline().decode("utf-8").strip()
        f.seek(0, 2)
        pos = f.tell()
        blocks = []
        count = 0
        while pos > 0 and count <= rows:
            size = min(65536, pos)
            pos -= size
            f.seek(pos)
            block = f.read(size)
            blocks.append(block)
            count += block.count(b"\n")
        truncated = pos > 0
        raw = b"".join(reversed(blocks)).decode("utf-8", errors="ignore")
    lines = raw.splitlines()
    if lines and lines[0] == header:
        lines = lines[1:]
    elif truncated and lines:
        lines = lines[1:]
    lines = lines[-rows:]
    text = header + "\n" + "\n".join(lines)
    return pd.read_csv(io.StringIO(text))


def score_event(model: dict, seg: pd.DataFrame, env: np.ndarray) -> dict:
    config = model["config"]
    vector, peak, rms, freq, power = extract_one(seg, config["fs"], config["top"], config["bands"])
    x = model_features(model, vector)
    env_filled, _ = fill_env(env.reshape(1, -1), np.asarray(model["env_medians"]))
    if model["env_model"] is not None:
        x = x - model["env_model"].predict(env_filled)
    scaled = model["scaler"].transform(x)
    recon = model["autoencoder"].predict(scaled)
    per_feature = (recon - scaled) ** 2
    score = float(np.mean(per_feature))
    threshold = float(model["threshold"])
    band_err = per_feature[0, 5:8]
    total = float(np.sum(band_err))
    bands = (band_err / total).tolist() if total > 0 else [0.0, 0.0, 0.0]
    return {"score": score, "score_ratio": score / threshold if threshold > 0 else math.nan, "peak_hz": peak, "rms": rms, "freq_x": freq.tolist(), "freq_y": power.tolist(), "bands": bands}


class Monitor:
    def __init__(self, bridge: Path, model_path: Path):
        self.bridge = bridge
        self.tap_path, _ = companion_paths(bridge)
        self.model = joblib.load(model_path)
        self.lock = threading.Lock()
        self.state = {"status_text": "대기 중", "status_class": "waiting", "score_ratio": None, "peak_hz": None, "rms": None, "temp_c": None, "humidity": None, "tap_index": None, "evaluated_count": 0, "anomaly_count": 0, "wave_x": [], "wave_y": [], "freq_x": [], "freq_y": [], "bands": [0, 0, 0], "updated_at": None, "info": "로거 데이터를 기다리는 중"}
        self.last_tap = -1
        self.evaluated_count = 0
        self.anomaly_count = 0
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self) -> None:
        while True:
            try:
                if not self.bridge.exists() or not self.tap_path.exists():
                    time.sleep(0.5)
                    continue
                config = self.model["config"]
                rows = int(config["fs"] * 15)
                data = tail_dataframe(self.bridge, rows)
                taps = read_taps(self.tap_path)
                if data.empty:
                    time.sleep(0.5)
                    continue
                t = pd.to_numeric(data["t_perf_ns"], errors="coerce").to_numpy(float)
                z = pd.to_numeric(data["az_g"], errors="coerce").to_numpy(float)
                max_t = np.nanmax(t)
                recent_start = max_t - 2e9
                recent = t >= recent_start
                wave_y = z[recent]
                wave_x = ((t[recent] - t[recent][0]) / 1e9).tolist() if np.any(recent) else []
                temp = float(pd.to_numeric(data["temp_c"], errors="coerce").dropna().iloc[-1]) if "temp_c" in data and pd.to_numeric(data["temp_c"], errors="coerce").notna().any() else None
                hum = float(pd.to_numeric(data["humidity"], errors="coerce").dropna().iloc[-1]) if "humidity" in data and pd.to_numeric(data["humidity"], errors="coerce").notna().any() else None
                unix_values = pd.to_numeric(data["t_unix_s"], errors="coerce").dropna() if "t_unix_s" in data else pd.Series(dtype=float)
                updated_at = float(unix_values.iloc[-1]) if len(unix_values) else time.time()
                update = {"wave_x": wave_x, "wave_y": wave_y.tolist(), "temp_c": temp, "humidity": hum, "updated_at": updated_at}
                end_needed = taps["tap_ns"] + (self.model["delay"] + self.model["window"]) * 1e9
                complete = taps[end_needed <= max_t]
                if len(complete):
                    tap_idx = int(complete.index[-1])
                    if tap_idx != self.last_tap:
                        tap_ns = float(complete.iloc[-1]["tap_ns"])
                        start = tap_ns + self.model["delay"] * 1e9
                        end = start + self.model["window"] * 1e9
                        m = (t >= start) & (t <= end)
                        seg = data.loc[m]
                        if len(seg) >= int(config["fs"] * self.model["window"] * 0.7):
                            env = np.array([float(pd.to_numeric(seg["temp_c"], errors="coerce").mean()) if "temp_c" in seg else math.nan, float(pd.to_numeric(seg["humidity"], errors="coerce").mean()) if "humidity" in seg else math.nan])
                            result = score_event(self.model, seg, env)
                            ratio = result["score_ratio"]
                            if not np.isfinite(ratio):
                                text, css = "측정 대기", "waiting"
                            elif ratio < 0.8:
                                text, css = "정상 범위", "normal"
                            elif ratio < 1.0:
                                text, css = "주의", "warning"
                            else:
                                text, css = "이상 후보", "anomaly"
                            self.evaluated_count += 1
                            if np.isfinite(ratio) and ratio >= 1.0:
                                self.anomaly_count += 1
                            env_text = "환경 보정 사용" if self.model.get("environment_enabled", self.model.get("env_model") is not None) else "환경 보정 미사용"
                            update.update(result)
                            update.update({"status_text": text, "status_class": css, "tap_index": tap_idx, "evaluated_count": self.evaluated_count, "anomaly_count": self.anomaly_count, "info": f"Autoencoder 8→8→3→8→8 | 로그·대역비율 피처 | {env_text} | 분석 대역 0–{config['top']:.1f} Hz | 임계값 {self.model['threshold']:.5f} | 최근 타격 {tap_idx}"})
                            self.last_tap = tap_idx
                with self.lock:
                    self.state.update(update)
            except Exception as exc:
                with self.lock:
                    self.state["info"] = f"대기/오류: {exc}"
            time.sleep(0.5)

    def get(self) -> dict:
        with self.lock:
            return dict(self.state)


def local_ip() -> str:
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True, check=False)
        for value in result.stdout.split():
            if ":" not in value and not value.startswith("127."):
                return value
    except OSError:
        pass
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        value = sock.getsockname()[0]
        sock.close()
        return value
    except OSError:
        return "라즈베리파이_IP"


def serve(bridge: Path, model: Path, host: str, port: int) -> None:
    monitor = Monitor(bridge, model)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return HTML

    @app.get("/api")
    def api():
        return jsonify(monitor.get())

    print(f"대시보드: http://{local_ip()}:{port}")
    app.run(host=host, port=port, threaded=True, use_reloader=False)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bridge = output / f"{args.condition}_bridge_log.csv"
    tap_path, meta_path = companion_paths(bridge)
    for path in (bridge, tap_path, meta_path):
        if path.exists():
            path.unlink()
    logger = Path(args.logger)
    if not logger.is_absolute() and not logger.exists():
        candidate = Path(__file__).resolve().parent / logger
        if candidate.exists():
            logger = candidate
    model = Path(args.model)
    if not model.exists():
        raise FileNotFoundError(f"모델 파일을 찾지 못했습니다: {model}")
    command = [sys.executable, str(logger), "--condition", args.condition, "--output-dir", str(output), "--sample-rate", str(args.sample_rate), "--range-g", str(args.range_g), "--pulse-ms", str(args.pulse_ms), "--period-s", str(args.period_s), "--warmup", str(args.warmup), "--n-taps", str(args.n_taps)]
    process = subprocess.Popen(command)
    try:
        deadline = time.time() + args.warmup * args.period_s + 30
        while not bridge.exists() and time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError("로거가 먼저 종료되었습니다.")
            time.sleep(0.5)
        if not bridge.exists():
            raise RuntimeError("bridge CSV가 생성되지 않았습니다.")
        serve(bridge, model, args.host, args.port)
    finally:
        if process.poll() is None:
            process.terminate()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--bridge", nargs="+", required=True)
    t.add_argument("--model", default="sentry_autoencoder.joblib")
    t.add_argument("--delay", type=float, default=0.05)
    t.add_argument("--window", type=float, default=2.0)
    t.add_argument("--train-count", type=int, default=120)
    t.add_argument("--threshold-count", type=int, default=30)
    t.add_argument("--test-count", type=int, default=30)
    t.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("evaluate")
    e.add_argument("--model", default="sentry_autoencoder.joblib")
    e.add_argument("--normal", nargs="*", default=[])
    e.add_argument("--anomaly", nargs="*", default=[])
    e.add_argument("--output", default="evaluation_summary.csv")
    d = sub.add_parser("dashboard")
    d.add_argument("--bridge", required=True)
    d.add_argument("--model", default="sentry_autoencoder.joblib")
    d.add_argument("--host", default="0.0.0.0")
    d.add_argument("--port", type=int, default=8080)
    r = sub.add_parser("run")
    r.add_argument("--logger", default="logger3_fixed.py")
    r.add_argument("--model", default="sentry_autoencoder.joblib")
    r.add_argument("--condition", default="T0-H0-M0-B0")
    r.add_argument("--output-dir", default="data/live")
    r.add_argument("--sample-rate", type=int, default=500)
    r.add_argument("--range-g", type=int, default=8)
    r.add_argument("--pulse-ms", type=float, default=40)
    r.add_argument("--period-s", type=float, default=5)
    r.add_argument("--warmup", type=int, default=10)
    r.add_argument("--n-taps", type=int, default=35)
    r.add_argument("--host", default="0.0.0.0")
    r.add_argument("--port", type=int, default=8080)
    return p


def main() -> None:
    args = parser().parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "dashboard":
        serve(Path(args.bridge), Path(args.model), args.host, args.port)
    else:
        run(args)


if __name__ == "__main__":
    main()
