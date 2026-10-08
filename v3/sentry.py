from __future__ import annotations

import argparse
import csv
import io
import json
import math
import socket
import re
import tempfile
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify


HTML = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SENTRY 보안 대시보드</title>
<style>
body{margin:0;background:#10141b;color:#eef2f7;font-family:Arial,sans-serif}.wrap{max-width:1100px;margin:auto;padding:20px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:16px}.title{font-size:28px;font-weight:700}.small{font-size:12px;color:#9ca8b8;margin-top:7px}.status{padding:11px 18px;border-radius:999px;font-weight:700;white-space:nowrap}.normal{background:#176b45}.warning{background:#916b12}.anomaly{background:#9d2c2c}.waiting{background:#465164}.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}.card{background:#1a202b;border-radius:14px;padding:16px;box-shadow:0 6px 18px #0004}.full{grid-column:span 6}.wide{grid-column:span 3}.label{font-size:13px;color:#9ca8b8}.value{font-size:27px;font-weight:700;margin-top:6px}.buttons{display:flex;gap:10px;flex-wrap:wrap}.button{border:0;border-radius:10px;padding:11px 15px;font-weight:700;color:white;background:#2f73bf;cursor:pointer}.button.warn{background:#8b6518}.button.stop{background:#6b3440}.progress{height:10px;background:#111722;border-radius:999px;overflow:hidden;margin-top:12px}.progress>div{height:100%;width:0;background:#6db7ff}.alarmbox{border:1px solid #344052}.alarmbox.active{border-color:#d64b4b;box-shadow:0 0 28px #d64b4b55}canvas{width:100%;height:280px;background:#111722;border-radius:10px}.thresholds{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:12px}.threshold{background:#111722;border-radius:10px;padding:12px}.reason{font-size:18px;font-weight:700;margin-top:8px}@media(max-width:800px){.grid{grid-template-columns:repeat(2,1fr)}.wide,.full{grid-column:span 2}.top{align-items:flex-start;flex-direction:column}.thresholds{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<div class="top"><div><div class="title">SENTRY 기체 보안 대시보드</div><div id="connection" class="small">서버 연결 확인 중</div></div><div id="securityStatus" class="status waiting">감시 해제</div></div>
<div class="grid">
<div class="card"><div class="label">진동 RMS</div><div id="vibration" class="value">-</div><div id="vibrationRatio" class="small">-</div></div>
<div class="card"><div class="label">순간 피크</div><div id="peakMotion" class="value">-</div><div id="peakRatio" class="small">-</div></div>
<div class="card"><div class="label">자세 변화</div><div id="tilt" class="value">-</div><div id="tiltThreshold" class="small">-</div></div>
<div class="card"><div class="label">표면온도 · DS18B20</div><div id="temp" class="value">-</div><div id="surfaceState" class="small">대기</div></div>
<div class="card"><div class="label">습도 · DHT22</div><div id="hum" class="value">-</div><div id="humidityState" class="small">대기</div></div>
<div class="card"><div class="label">누적 경보</div><div id="alertCount" class="value">0</div></div>
<div class="card full alarmbox" id="alarmBox"><div class="label">최근 경보</div><div id="reason" class="reason">없음</div><div id="lastAlert" class="small">-</div></div>
<div class="card full"><div class="label">CSV 저장 상태</div><div id="storage" class="small">대기</div></div>
<div class="card full"><div class="label">최근 3축 합성 진동</div><canvas id="wave"></canvas></div>
<div class="card full"><div class="label">감시 기준</div><div class="thresholds"><div class="threshold"><div class="label">진동 기준</div><div id="vibrationThreshold" class="value">-</div></div><div class="threshold"><div class="label">충격 기준</div><div id="peakThreshold" class="value">-</div></div><div class="threshold"><div class="label">자세 기준</div><div id="angleThreshold" class="value">-</div></div></div><div class="progress"><div id="calibrationBar"></div></div><div id="calibrationText" class="small">기체를 설치한 뒤 감시 시작을 누른다.</div></div>
<div class="card full"><div class="buttons"><button class="button" onclick="action('/api/security/arm')">감시 시작 · 기준 재설정</button><button class="button warn" onclick="action('/api/security/reset')">경보 해제</button><button class="button stop" onclick="action('/api/security/disarm')">감시 해제</button></div><div id="info" class="small">-</div></div>
</div>
</div>
<script>
let audioContext=null,lastAlert=false,lastConnected=true,lastArmed=false;
function enableAudio(){if(!audioContext)audioContext=new(window.AudioContext||window.webkitAudioContext)();if(audioContext.state==='suspended')audioContext.resume();if('Notification'in window&&Notification.permission==='default')Notification.requestPermission()}
function beep(){if(!audioContext)return;const o=audioContext.createOscillator(),g=audioContext.createGain();o.frequency.value=880;g.gain.setValueAtTime(.16,audioContext.currentTime);g.gain.exponentialRampToValueAtTime(.001,audioContext.currentTime+.28);o.connect(g);g.connect(audioContext.destination);o.start();o.stop(audioContext.currentTime+.3)}
function notify(title,body){if('Notification'in window&&Notification.permission==='granted')new Notification(title,{body})}
function fmt(v,d,u){return v==null?'-':Number(v).toFixed(d)+u}
function line(canvas,x,y){const c=document.getElementById(canvas),d=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*d;c.height=h*d;const g=c.getContext('2d');g.scale(d,d);g.clearRect(0,0,w,h);g.strokeStyle='#2c3544';g.beginPath();for(let i=1;i<5;i++){const yy=h*i/5;g.moveTo(0,yy);g.lineTo(w,yy)}g.stroke();if(!y||y.length<2)return;let ymin=Math.min(...y),ymax=Math.max(...y);if(ymax===ymin){ymax+=1;ymin-=1}g.strokeStyle='#67b2ff';g.lineWidth=2;g.beginPath();for(let i=0;i<y.length;i++){const px=i/(y.length-1)*w,py=h-(y[i]-ymin)/(ymax-ymin)*h;if(i===0)g.moveTo(px,py);else g.lineTo(px,py)}g.stroke()}
async function action(path){enableAudio();try{await fetch(path,{method:'POST'});await update()}catch(e){}}
async function update(){const conn=document.getElementById('connection');try{const r=await fetch('/api',{cache:'no-store'});if(!r.ok)throw new Error('api');const s=await r.json();const st=document.getElementById('securityStatus');st.textContent=s.security_status_text||'대기 중';st.className='status '+(s.security_status_class||'waiting');document.getElementById('vibration').textContent=fmt(s.vibration_rms_g,4,' g');document.getElementById('peakMotion').textContent=fmt(s.vibration_peak_g,4,' g');document.getElementById('tilt').textContent=fmt(s.orientation_change_deg,1,'°');document.getElementById('tiltThreshold').textContent=s.orientation_threshold_deg==null?'-':'기준 '+s.orientation_threshold_deg.toFixed(1)+'°';document.getElementById('temp').textContent=fmt(s.surface_temp_c,1,' °C');document.getElementById('hum').textContent=fmt(s.humidity,1,' %');document.getElementById('surfaceState').textContent=s.surface_status||'대기';document.getElementById('humidityState').textContent=s.humidity_status||'대기';document.getElementById('storage').textContent=(s.logger_status||'')+' | '+(s.storage_status||'')+' | '+(s.active_csv||'파일 대기')+' | '+(s.run_folder||'');document.getElementById('alertCount').textContent=String(s.alert_count||0);document.getElementById('vibrationThreshold').textContent=fmt(s.vibration_threshold_g,4,' g');document.getElementById('peakThreshold').textContent=fmt(s.peak_threshold_g,4,' g');document.getElementById('angleThreshold').textContent=fmt(s.orientation_threshold_deg,1,'°');document.getElementById('vibrationRatio').textContent=s.vibration_ratio==null?'-':'기준의 '+s.vibration_ratio.toFixed(2)+'배';document.getElementById('peakRatio').textContent=s.peak_ratio==null?'-':'기준의 '+s.peak_ratio.toFixed(2)+'배';const progress=Math.max(0,Math.min(100,s.calibration_progress||0));document.getElementById('calibrationBar').style.width=progress+'%';document.getElementById('calibrationText').textContent=s.calibration_text||'-';document.getElementById('reason').textContent=s.last_alert_reason||'없음';document.getElementById('lastAlert').textContent=s.last_alert_at_text||'-';document.getElementById('info').textContent=s.info||'-';document.getElementById('alarmBox').className='card full alarmbox'+(s.security_alert?' active':'');line('wave',s.wave_x,s.wave_y);if(s.updated_at==null){conn.textContent='서버 연결됨 · 센서 데이터 대기'}else{const age=Math.max(0,Date.now()/1000-s.updated_at);conn.textContent=age<3?'서버 연결됨 · 실시간 감시 중':'서버 연결됨 · 마지막 센서 데이터 '+age.toFixed(1)+'초 전'}if(s.security_alert&&!lastAlert){beep();setTimeout(beep,700);notify('SENTRY 도난·이동 의심',s.last_alert_reason||'기체 움직임이 감지됨')}lastAlert=!!s.security_alert;lastConnected=true;lastArmed=!!s.security_armed}catch(e){conn.textContent='서버 연결 끊김';const st=document.getElementById('securityStatus');st.textContent='연결 끊김';st.className='status anomaly';if(lastConnected&&lastArmed){beep();notify('SENTRY 연결 끊김','장치 전원 또는 네트워크 상태를 확인해야 한다.')}lastConnected=false}}
setInterval(update,500);update();
</script>
</body>
</html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SENTRY continuous security dashboard")
    parser.add_argument("--logger", default="logger.py")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bridge", help="기존 CSV 한 개 열기(자동 파일 전환 없음)")
    source.add_argument("--manifest", help="실행 폴더의 current.json을 따라가기")
    parser.add_argument("--condition", default="SECURITY")
    parser.add_argument("--output-dir", default="data/security")
    parser.add_argument("--sample-rate", type=int, choices=[125, 250, 500], default=125)
    parser.add_argument("--range-g", type=int, choices=[2, 4, 8], default=8)
    parser.add_argument("--no-dht", action="store_true")
    parser.add_argument("--no-surface", action="store_true")
    parser.add_argument("--surface-device", help="DS18B20 장치 ID: 28-...")
    parser.add_argument("--rotate-seconds", type=float, default=3600.0)
    parser.add_argument("--flush-seconds", type=float, default=1.0)
    parser.add_argument("--fsync-seconds", type=float, default=10.0)
    parser.add_argument("--env-stale-seconds", type=float, default=10.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--calibration-seconds", type=float, default=20.0)
    parser.add_argument("--window-seconds", type=float, default=0.5)
    parser.add_argument("--step-seconds", type=float, default=0.25)
    parser.add_argument("--vibration-multiplier", type=float, default=4.0)
    parser.add_argument("--peak-multiplier", type=float, default=4.0)
    parser.add_argument("--min-vibration-g", type=float, default=0.015)
    parser.add_argument("--min-peak-g", type=float, default=0.08)
    parser.add_argument("--tilt-deg", type=float, default=8.0)
    parser.add_argument("--tilt-hold-seconds", type=float, default=1.0)
    parser.add_argument("--vote-window", type=int, default=5)
    parser.add_argument("--required-votes", type=int, default=3)
    parser.add_argument("--stale-seconds", type=float, default=5.0)
    parser.add_argument("--auto-arm", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+", args.condition) or args.condition in {".", ".."}:
        parser.error("condition에는 문자, 숫자, _, -, .만 사용할 수 있습니다.")
    for name in ("rotate_seconds", "flush_seconds", "fsync_seconds", "env_stale_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name}은 유한한 양수여야 합니다.")
    return args


def metadata_path(bridge: Path) -> Path:
    if not bridge.name.endswith("_bridge_log.csv"):
        raise ValueError("bridge CSV 이름은 *_bridge_log.csv 형식이어야 합니다.")
    return bridge.with_name(bridge.name[:-len("_bridge_log.csv")] + "_metadata.json")


def metadata_for(bridge: Path) -> dict:
    if bridge.suffix.lower() == ".json":
        return json.loads(bridge.read_text(encoding="utf-8"))
    path = metadata_path(bridge)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"sample_rate_hz": 125, "range_g": 8}


def tail_dataframe(path: Path, rows: int) -> pd.DataFrame:
    with path.open("rb") as file:
        header = file.readline().decode("utf-8").strip()
        file.seek(0, 2)
        position = file.tell()
        blocks, count = [], 0
        while position > 0 and count <= rows + 1:
            size = min(65536, position)
            position -= size
            file.seek(position)
            block = file.read(size)
            blocks.append(block)
            count += block.count(b"\n")
        truncated = position > 0
        raw = b"".join(reversed(blocks))
    # 쓰기 중인 마지막 불완전 행은 다음 조회로 미룬다.
    raw = raw[:raw.rfind(b"\n") + 1]
    lines = raw.decode("utf-8", errors="ignore").splitlines()
    if lines and lines[0] == header:
        lines = lines[1:]
    elif truncated and lines:
        lines = lines[1:]
    lines = lines[-rows:]
    if not lines or not header:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(header + "\n" + "\n".join(lines)))


def manifest_csv(source: Path, filename: str | None) -> Path | None:
    if filename is None:
        return None
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("current.json의 CSV 이름이 올바르지 않습니다.")
    return source.parent / filename


def read_live_tail(source: Path, rows: int) -> tuple[pd.DataFrame, dict, Path]:
    metadata = metadata_for(source)
    if source.suffix.lower() != ".json":
        return tail_dataframe(source, rows), metadata, source
    current = manifest_csv(source, metadata.get("current_csv"))
    if current is None:
        raise ValueError("current.json에 current_csv가 없습니다.")
    data = tail_dataframe(current, rows)
    previous = manifest_csv(source, metadata.get("previous_csv"))
    if len(data) < rows and previous is not None and previous.exists():
        older = tail_dataframe(previous, rows - len(data))
        frames = [frame for frame in (older, data) if not frame.empty]
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return data.tail(rows), metadata, current


def last_number(data: pd.DataFrame, column: str) -> float | None:
    # dropna().iloc[-1]는 지난 파일/지난 시각의 값을 되살리므로 사용하지 않는다.
    if data.empty or column not in data:
        return None
    value = pd.to_numeric(data[column].iloc[-1], errors="coerce")
    return finite_or_none(value)


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def format_alert_time(value: float | None) -> str | None:
    if value is None:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def security_window_metrics(values: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    dynamic = values - center
    magnitude = np.linalg.norm(dynamic, axis=1)
    rms = float(np.sqrt(np.mean(magnitude * magnitude)))
    peak = float(np.max(magnitude))
    return rms, peak, center, magnitude


class SecurityMonitor:
    def __init__(self, bridge: Path, args: argparse.Namespace):
        self.bridge = bridge
        self.logger_process = getattr(args, "logger_process", None)
        self.env_stale_seconds = float(metadata_for(bridge).get("env_stale_seconds", args.env_stale_seconds))
        self.recent_data = pd.DataFrame()
        self.fs = float(metadata_for(bridge).get("sample_rate_hz", args.sample_rate))
        self.calibration_seconds = max(5.0, args.calibration_seconds)
        self.window_seconds = max(0.2, args.window_seconds)
        self.step_seconds = max(0.1, args.step_seconds)
        self.vibration_multiplier = max(1.0, args.vibration_multiplier)
        self.peak_multiplier = max(1.0, args.peak_multiplier)
        self.min_vibration_g = max(0.0001, args.min_vibration_g)
        self.min_peak_g = max(0.0001, args.min_peak_g)
        self.orientation_threshold_deg = max(1.0, args.tilt_deg)
        self.orientation_hold_seconds = max(0.2, args.tilt_hold_seconds)
        self.vote_window = max(1, args.vote_window)
        self.required_votes = min(max(1, args.required_votes), self.vote_window)
        self.stale_seconds = max(2.0, args.stale_seconds)
        self.lock = threading.Lock()
        self.baseline_vector: np.ndarray | None = None
        self.vibration_threshold_g: float | None = None
        self.peak_threshold_g: float | None = None
        self.calibration_start_ns: float | None = None
        self.calibration_requested = False
        self.armed = False
        self.alert_latched = False
        self.alert_count = 0
        self.last_alert_at: float | None = None
        self.last_alert_reason: str | None = None
        self.last_evaluated_ns = 0.0
        self.tilt_started_ns: float | None = None
        self.motion_votes: deque[bool] = deque(maxlen=self.vote_window)
        self.alert_path = (
            bridge.parent / "security_alerts.csv" if bridge.suffix.lower() == ".json"
            else bridge.with_name(bridge.name.replace("_bridge_log.csv", "_security_alerts.csv"))
        )
        self.state = {
            "security_armed": False,
            "security_alert": False,
            "security_status_text": "감시 해제",
            "security_status_class": "waiting",
            "vibration_rms_g": None,
            "vibration_peak_g": None,
            "orientation_change_deg": None,
            "vibration_threshold_g": None,
            "peak_threshold_g": None,
            "orientation_threshold_deg": self.orientation_threshold_deg,
            "vibration_ratio": None,
            "peak_ratio": None,
            "temp_c": None,
            "surface_temp_c": None,
            "surface_age_s": None,
            "dht_age_s": None,
            "humidity": None,
            "surface_enabled": True,
            "dht_enabled": True,
            "active_csv": "",
            "run_folder": str(bridge.parent),
            "storage_status": "waiting",
            "rotate_seconds": args.rotate_seconds,
            "alert_count": 0,
            "last_alert_at": None,
            "last_alert_at_text": None,
            "last_alert_reason": None,
            "calibration_progress": 0.0,
            "calibration_text": "기체를 설치한 뒤 감시 시작을 누른다.",
            "wave_x": [],
            "wave_y": [],
            "updated_at": None,
            "info": "센서 데이터를 기다리는 중",
        }
        threading.Thread(target=self.loop, daemon=True).start()
        if args.auto_arm:
            self.arm()

    def arm(self) -> dict:
        with self.lock:
            self.calibration_requested = True
            self.calibration_start_ns = None
            self.baseline_vector = None
            self.vibration_threshold_g = None
            self.peak_threshold_g = None
            self.armed = False
            self.alert_latched = False
            self.last_evaluated_ns = 0.0
            self.motion_votes.clear()
            self.tilt_started_ns = None
            self.state.update(
                {
                    "security_armed": False,
                    "security_alert": False,
                    "security_status_text": "보정 중",
                    "security_status_class": "warning",
                    "calibration_progress": 0.0,
                    "calibration_text": f"{self.calibration_seconds:.0f}초 동안 기체를 건드리지 않는다.",
                    "vibration_rms_g": None,
                    "vibration_peak_g": None,
                    "orientation_change_deg": None,
                    "vibration_ratio": None,
                    "peak_ratio": None,
                }
            )
            return dict(self.state)

    def disarm(self) -> dict:
        with self.lock:
            self.calibration_requested = False
            self.armed = False
            self.alert_latched = False
            self.motion_votes.clear()
            self.tilt_started_ns = None
            self.state.update(
                {
                    "security_armed": False,
                    "security_alert": False,
                    "security_status_text": "감시 해제",
                    "security_status_class": "waiting",
                    "calibration_progress": 0.0,
                    "calibration_text": "감시가 해제되었다.",
                }
            )
            return dict(self.state)

    def reset_alert(self) -> dict:
        with self.lock:
            self.alert_latched = False
            self.motion_votes.clear()
            self.tilt_started_ns = None
            if self.armed:
                text, style = "감시 중", "normal"
            elif self.calibration_requested:
                text, style = "보정 중", "warning"
            else:
                text, style = "감시 해제", "waiting"
            self.state.update(
                {
                    "security_alert": False,
                    "security_status_text": text,
                    "security_status_class": style,
                }
            )
            return dict(self.state)

    def calibrate(self, data: pd.DataFrame) -> None:
        values = data[["ax_g", "ay_g", "az_g"]].to_numpy(float)
        window_samples = max(16, int(self.fs * self.window_seconds))
        if len(values) < window_samples * 4:
            return
        rms_values = []
        peak_values = []
        for start in range(0, len(values) - window_samples + 1, window_samples):
            rms, peak, _, _ = security_window_metrics(values[start:start + window_samples])
            rms_values.append(rms)
            peak_values.append(peak)
        if len(rms_values) < 4:
            return
        baseline = np.median(values, axis=0)
        vibration_threshold = max(self.min_vibration_g, float(np.quantile(rms_values, 0.99)) * self.vibration_multiplier)
        peak_threshold = max(self.min_peak_g, float(np.quantile(peak_values, 0.99)) * self.peak_multiplier)
        with self.lock:
            self.baseline_vector = baseline
            self.vibration_threshold_g = vibration_threshold
            self.peak_threshold_g = peak_threshold
            self.calibration_requested = False
            self.armed = True
            self.alert_latched = False
            self.motion_votes.clear()
            self.state.update(
                {
                    "security_armed": True,
                    "security_alert": False,
                    "security_status_text": "감시 중",
                    "security_status_class": "normal",
                    "vibration_threshold_g": vibration_threshold,
                    "peak_threshold_g": peak_threshold,
                    "calibration_progress": 100.0,
                    "calibration_text": "기준 측정이 끝났으며 보안 감시가 시작되었다.",
                    "info": f"기준 RMS {np.median(rms_values):.5f} g | 기준 피크 {np.median(peak_values):.5f} g | 최근 {self.vote_window}개 중 {self.required_votes}개 초과 시 경보",
                }
            )

    def write_alert(self, reason: str, now: float, metrics: dict) -> None:
        new_file = not self.alert_path.exists()
        with self.alert_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["t_unix_s", "time", "reason", "vibration_rms_g", "vibration_peak_g", "orientation_change_deg"])
            writer.writerow(
                [
                    f"{now:.6f}",
                    format_alert_time(now),
                    reason,
                    f"{metrics['vibration_rms_g']:.8f}",
                    f"{metrics['vibration_peak_g']:.8f}",
                    f"{metrics['orientation_change_deg']:.4f}",
                ]
            )

    def trigger_alert(self, reason: str, metrics: dict) -> None:
        with self.lock:
            if self.alert_latched:
                return
            now = time.time()
            self.alert_latched = True
            self.alert_count += 1
            self.last_alert_at = now
            self.last_alert_reason = reason
            self.state.update(
                {
                    "security_alert": True,
                    "security_status_text": "도난·이동 의심",
                    "security_status_class": "anomaly",
                    "alert_count": self.alert_count,
                    "last_alert_at": now,
                    "last_alert_at_text": format_alert_time(now),
                    "last_alert_reason": reason,
                }
            )
        self.write_alert(reason, now, metrics)

    def evaluate(self, data: pd.DataFrame, max_t: float) -> None:
        with self.lock:
            baseline = None if self.baseline_vector is None else self.baseline_vector.copy()
            vibration_threshold = self.vibration_threshold_g
            peak_threshold = self.peak_threshold_g
            armed = self.armed
        if not armed or baseline is None or vibration_threshold is None or peak_threshold is None:
            return
        segment = data[data["t_perf_ns"] >= max_t - self.window_seconds * 1e9]
        if len(segment) < max(16, int(self.fs * self.window_seconds * 0.7)):
            return
        values = segment[["ax_g", "ay_g", "az_g"]].to_numpy(float)
        vibration_rms, vibration_peak, current_vector, magnitude = security_window_metrics(values)
        baseline_norm = float(np.linalg.norm(baseline))
        current_norm = float(np.linalg.norm(current_vector))
        angle = 0.0
        if baseline_norm > 1e-6 and current_norm > 1e-6:
            cosine = float(np.dot(baseline, current_vector) / (baseline_norm * current_norm))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        vibration_exceeded = vibration_rms >= vibration_threshold
        peak_exceeded = vibration_peak >= peak_threshold
        motion = vibration_exceeded or peak_exceeded
        self.motion_votes.append(motion)
        votes = sum(self.motion_votes)
        if angle >= self.orientation_threshold_deg:
            if self.tilt_started_ns is None:
                self.tilt_started_ns = max_t
        else:
            self.tilt_started_ns = None
        tilt_sustained = self.tilt_started_ns is not None and (max_t - self.tilt_started_ns) / 1e9 >= self.orientation_hold_seconds
        reason = None
        if vibration_peak >= peak_threshold * 2.0:
            reason = "강한 충격이 감지됨"
        elif tilt_sustained:
            reason = "기체 자세 변화가 지속됨"
        elif motion and angle >= self.orientation_threshold_deg * 0.5:
            reason = "진동과 자세 변화가 함께 감지됨"
        elif votes >= self.required_votes:
            reason = "진동이 반복적으로 감지됨"
        vibration_ratio = vibration_rms / vibration_threshold
        peak_ratio = vibration_peak / peak_threshold
        recent_t = segment["t_perf_ns"].to_numpy(float)
        metrics = {
            "vibration_rms_g": vibration_rms,
            "vibration_peak_g": vibration_peak,
            "orientation_change_deg": angle,
        }
        with self.lock:
            if self.alert_latched:
                text, style = "도난·이동 의심", "anomaly"
            elif motion or angle >= self.orientation_threshold_deg:
                text, style = "진동 감지", "warning"
            else:
                text, style = "감시 중", "normal"
            self.state.update(
                {
                    "security_armed": True,
                    "security_alert": self.alert_latched,
                    "security_status_text": text,
                    "security_status_class": style,
                    "vibration_rms_g": vibration_rms,
                    "vibration_peak_g": vibration_peak,
                    "orientation_change_deg": angle,
                    "vibration_threshold_g": vibration_threshold,
                    "peak_threshold_g": peak_threshold,
                    "orientation_threshold_deg": self.orientation_threshold_deg,
                    "vibration_ratio": vibration_ratio,
                    "peak_ratio": peak_ratio,
                    "wave_x": ((recent_t - recent_t[0]) / 1e9).tolist(),
                    "wave_y": magnitude.tolist(),
                    "info": f"최근 {self.vote_window}개 판정 중 움직임 {votes}개 | 자세 변화 {angle:.1f}° | {self.alert_path.name}",
                }
            )
        if reason is not None:
            self.trigger_alert(reason, metrics)

    def loop(self) -> None:
        rows = int(self.fs * max(self.calibration_seconds + 5.0, 20.0))
        while True:
            try:
                if not self.bridge.exists():
                    time.sleep(0.3)
                    continue
                data, metadata, active_csv = read_live_tail(self.bridge, rows)
                with self.lock:
                    self.state.update({
                        "active_csv": active_csv.name,
                        "storage_status": metadata.get("status", "CSV 직접 보기"),
                        "rotate_seconds": metadata.get("rotate_seconds", None),
                        "surface_enabled": metadata.get("surface_enabled", "surface_temp_c" in data),
                        "dht_enabled": metadata.get("dht_enabled", True),
                    })
                required = {"t_perf_ns", "ax_g", "ay_g", "az_g"}
                if data.empty or not required.issubset(data.columns):
                    time.sleep(0.3)
                    continue
                frames = [frame for frame in (self.recent_data, data) if not frame.empty]
                data = pd.concat(frames, ignore_index=True)
                data = data.drop_duplicates(subset=["t_perf_ns"], keep="last")
                data = data.sort_values("t_perf_ns").tail(rows).copy()
                self.recent_data = data
                for column in ["t_perf_ns", "t_unix_s", "ax_g", "ay_g", "az_g",
                               "temp_c", "humidity", "surface_temp_c", "surface_age_s", "dht_age_s"]:
                    if column in data.columns:
                        data[column] = pd.to_numeric(data[column], errors="coerce")
                data = data.dropna(subset=["t_perf_ns", "ax_g", "ay_g", "az_g"]).sort_values("t_perf_ns")
                if data.empty:
                    time.sleep(0.3)
                    continue
                max_t = float(data["t_perf_ns"].iloc[-1])
                updated_at = float(data["t_unix_s"].dropna().iloc[-1]) if "t_unix_s" in data and data["t_unix_s"].notna().any() else time.time()
                with self.lock:
                    self.state.update({
                        "updated_at": updated_at,
                        "temp_c": last_number(data, "temp_c"),
                        "surface_temp_c": last_number(data, "surface_temp_c"),
                        "surface_age_s": last_number(data, "surface_age_s"),
                        "humidity": last_number(data, "humidity"),
                        "dht_age_s": last_number(data, "dht_age_s"),
                    })
                    calibration_requested = self.calibration_requested
                    calibration_start_ns = self.calibration_start_ns
                    armed = self.armed
                if calibration_requested:
                    if calibration_start_ns is None:
                        with self.lock:
                            self.calibration_start_ns = max_t
                            calibration_start_ns = max_t
                    elapsed = max(0.0, (max_t - calibration_start_ns) / 1e9)
                    progress = min(100.0, elapsed / self.calibration_seconds * 100.0)
                    with self.lock:
                        self.state.update(
                            {
                                "security_status_text": "보정 중",
                                "security_status_class": "warning",
                                "calibration_progress": progress,
                                "calibration_text": f"기체를 건드리지 않는다. {elapsed:.1f}/{self.calibration_seconds:.0f}초",
                            }
                        )
                    if elapsed >= self.calibration_seconds:
                        self.calibrate(data[data["t_perf_ns"] >= calibration_start_ns])
                elif armed:
                    if max_t - self.last_evaluated_ns >= self.step_seconds * 1e9:
                        self.last_evaluated_ns = max_t
                        self.evaluate(data, max_t)
                    if time.time() - updated_at > self.stale_seconds:
                        metrics = {
                            "vibration_rms_g": float(self.state.get("vibration_rms_g") or 0.0),
                            "vibration_peak_g": float(self.state.get("vibration_peak_g") or 0.0),
                            "orientation_change_deg": float(self.state.get("orientation_change_deg") or 0.0),
                        }
                        self.trigger_alert("센서 데이터가 중단됨", metrics)
                else:
                    segment = data[data["t_perf_ns"] >= max_t - self.window_seconds * 1e9]
                    if len(segment) >= 2:
                        values = segment[["ax_g", "ay_g", "az_g"]].to_numpy(float)
                        _, _, _, magnitude = security_window_metrics(values)
                        recent_t = segment["t_perf_ns"].to_numpy(float)
                        with self.lock:
                            self.state.update(
                                {
                                    "wave_x": ((recent_t - recent_t[0]) / 1e9).tolist(),
                                    "wave_y": magnitude.tolist(),
                                }
                            )
            except Exception as exc:
                with self.lock:
                    self.state["info"] = f"대기/오류: {exc}"
            time.sleep(0.25)

    def get(self) -> dict:
        with self.lock:
            result = dict(self.state)
        for key in [
            "vibration_rms_g",
            "vibration_peak_g",
            "orientation_change_deg",
            "vibration_threshold_g",
            "peak_threshold_g",
            "orientation_threshold_deg",
            "vibration_ratio",
            "peak_ratio",
            "temp_c",
            "surface_temp_c",
            "surface_age_s",
            "dht_age_s",
            "humidity",
            "updated_at",
            "last_alert_at",
            "calibration_progress",
        ]:
            result[key] = finite_or_none(result.get(key))
        elapsed = max(0.0, time.time() - result["updated_at"]) if result.get("updated_at") else math.inf
        for value_col, age_col, enabled_col, status_col in (
            ("surface_temp_c", "surface_age_s", "surface_enabled", "surface_status"),
            ("humidity", "dht_age_s", "dht_enabled", "humidity_status"),
        ):
            age = result.get(age_col)
            stale = elapsed > self.env_stale_seconds or (
                age is not None and age + elapsed > self.env_stale_seconds)
            if not result[enabled_col]:
                result[value_col], result[status_col] = None, "미사용"
            elif result.get(value_col) is None or stale:
                result[value_col], result[status_col] = None, "결측 / 센서 확인"
            else:
                result[status_col] = "정상"
        if self.logger_process is not None:
            exit_code = self.logger_process.poll()
            result["logger_status"] = "수집 중" if exit_code is None else f"로거 종료 (코드 {exit_code})"
        else:
            result["logger_status"] = "기존 기록 연결"
        return result


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


def resolve_logger(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() and not path.exists():
        candidate = Path(__file__).resolve().parent / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"로거 파일을 찾지 못했습니다: {path}")
    return path


def serve(bridge: Path, args: argparse.Namespace) -> None:
    monitor = SecurityMonitor(bridge, args)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return HTML

    @app.get("/api")
    def api():
        return jsonify(monitor.get())

    @app.post("/api/security/arm")
    def arm():
        return jsonify(monitor.arm())

    @app.post("/api/security/disarm")
    def disarm():
        return jsonify(monitor.disarm())

    @app.post("/api/security/reset")
    def reset():
        return jsonify(monitor.reset_alert())

    print(f"보안 대시보드: http://{local_ip()}:{args.port}")
    print("대시보드에서 감시 시작을 누른 뒤 보정 시간 동안 기체를 건드리지 않는다.")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


def main() -> None:
    args = parse_args()
    process: subprocess.Popen | None = None
    try:
        if args.bridge or args.manifest:
            bridge = Path(args.bridge or args.manifest).resolve()
            if not bridge.exists():
                raise FileNotFoundError(f"기록 파일을 찾지 못했습니다: {bridge}")
        else:
            logger = resolve_logger(args.logger).resolve()
            output = Path(args.output_dir).resolve()
            output.mkdir(parents=True, exist_ok=True)
            run_dir = Path(tempfile.mkdtemp(
                prefix=f"{args.condition}_{time.strftime('%Y%m%dT%H%M%S')}_", dir=output))
            bridge = run_dir / "current.json"
            command = [
                sys.executable, "-u", str(logger),
                "--condition", args.condition, "--run-dir", str(run_dir),
                "--sample-rate", str(args.sample_rate), "--range-g", str(args.range_g),
                "--rotate-seconds", str(args.rotate_seconds),
                "--flush-seconds", str(args.flush_seconds),
                "--fsync-seconds", str(args.fsync_seconds),
                "--env-stale-seconds", str(args.env_stale_seconds),
            ]
            if args.no_dht:
                command.append("--no-dht")
            if args.no_surface:
                command.append("--no-surface")
            if args.surface_device:
                command.extend(["--surface-device", args.surface_device])
            process = subprocess.Popen(command)
            args.logger_process = process
            deadline = time.monotonic() + 45.0
            while not bridge.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("로거가 먼저 종료되었습니다. 센서/라이브러리/디스크 오류를 확인하세요.")
                time.sleep(0.2)
            if not bridge.exists():
                raise RuntimeError("로거 시작 시간 초과: current.json이 생성되지 않았습니다.")
            print(f"[저장] 실행 폴더: {run_dir}")
            print(f"[저장] {args.rotate_seconds:g}초마다 새 CSV. 기존 파일은 유지됩니다.")
        serve(bridge, args)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=25.0)
            except subprocess.TimeoutExpired:
                print("[경고] 로거 종료 지연으로 강제 종료합니다. 마지막 파일을 점검하세요.")
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
