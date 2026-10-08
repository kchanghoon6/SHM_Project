from __future__ import annotations

from pathlib import Path

import argparse
import numpy as np
import pandas as pd


DEFAULT_BAND_EDGES = "0,3,10,20"

def parse_band_edges(text: str) -> list[float]:
    edges = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(edges) < 2:
        raise ValueError("--band-edges는 최소 2개 이상의 숫자가 필요합니다. 예: 0,3,10,20")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("--band-edges는 오름차순이어야 합니다. 예: 0,3,10,20")
    return edges


def band_names_from_edges(edges: list[float]) -> list[str]:
    names = []
    for lo, hi in zip(edges, edges[1:]):
        lo_text = str(lo).replace(".", "p")
        hi_text = str(hi).replace(".", "p")
        names.append(f"band_{lo_text}_{hi_text}hz")
    return names


def generate_sample_data(path: Path, sample_rate: int = 100, minutes: float = 6.0) -> None:
    rng = np.random.default_rng(48)
    n = int(minutes * 60 * sample_rate)
    t = np.arange(n) / sample_rate

    temp = 22 + 5 * np.sin(2 * np.pi * t / (minutes * 60)) + rng.normal(0, 0.05, n)
    humidity = 48 + 9 * np.cos(2 * np.pi * t / (minutes * 60) + 0.7) + rng.normal(0, 0.3, n)

    f1 = 18.0 - 0.055 * (temp - 22) + 0.018 * (humidity - 48)
    damaged = t > minutes * 60 * 0.68
    f1 = np.where(damaged, f1 - 2.2, f1)

    phase = 2 * np.pi * np.cumsum(f1) / sample_rate
    ax = 0.025 * np.sin(phase) + 0.006 * np.sin(2 * phase) + rng.normal(0, 0.004, n)
    ay = 0.012 * np.sin(phase * 0.95) + rng.normal(0, 0.004, n)
    az = 1.0 + 0.018 * np.sin(phase * 1.03) + rng.normal(0, 0.004, n)
    ax[damaged] += 0.016 * np.sin(2 * np.pi * 58.0 * t[damaged])

    df = pd.DataFrame(
        {
            "ms": (t * 1000).astype(int),
            "temp_c": temp,
            "humidity": humidity,
            "adxl_temp_c": temp + rng.normal(0.7, 0.08, n),
            "ax_g": ax,
            "ay_g": ay,
            "az_g": az,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def infer_sample_rate(df: pd.DataFrame) -> float:
    ms = df["ms"].to_numpy(dtype=float)
    dt = np.diff(ms)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("ms 열이 증가하지 않아 샘플링 주파수를 계산할 수 없습니다.")
    return 1000.0 / np.median(dt)


def spectral_entropy(power: np.ndarray) -> float:
    total = np.sum(power)
    if total <= 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(power)))

def extract_axis_features(
        signal:np.ndarray,
        sample_rate: float,
        band_edges: list[float],
        prefix: str,
) -> dict[str, float]:
    signal = signal.astype(float)
    signal = signal - np.mean(signal)
    window = np.hanning(len(signal))
    fft = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    power = np.abs(fft) ** 2

    nyquist = sample_rate / 2.0
    usable = (freqs >= max(0.5, band_edges[0])) & (freqs <= min(band_edges[-1], nyquist))
    if not np.any(usable):
        usable = freqs > 0
    
    peak_i = np.argmax(power[usable])
    usable_freq = freqs[usable]
    usable_power = power[usable]

    row = {
        f"{prefix}_rms": float(np.sqrt(np.mean(signal ** 2))),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_peak_to_peak": float(np.max(signal) - np.min(signal)),
        f"{prefix}_spectral_entropy": spectral_entropy(usable_power),
        f"{prefix}_peak_freq_hz": float(usable_freq[peak_i]),
        f"{prefix}_peak_power": float(usable_power[peak_i]),
    }

    for name, lo, hi in zip(band_names_from_edges(band_edges), band_edges, band_edges[1:]):
        idx = (freqs >= lo) & (freqs < hi)
        row[f"{prefix}_{name}_energy"] = float(np.sum(power[idx]))

    return row


def extract_segment_features(
        segment: pd.DataFrame, 
        sample_rate: float, 
        segment_id: int,
        band_edges: list[float],
        axes: list[str],
) -> dict[str, float]:

    row: dict[str, float] = {
        "segment_id": float(segment_id),
        "start_ms": float(segment["ms"].iloc[0]),
        "end_ms": float(segment["ms"].iloc[-1]),
        "temp_c": float(segment["temp_c"].interpolate().bfill().mean()),
        "humidity": float(segment["humidity"].interpolate().bfill().mean()),
    }
    if "adxl_temp_c" in segment.columns:
        row["adxl_temp_c"] = float(segment["adxl_temp_c"].interpolate().bfill().ffill().mean())

    for axis in axes:
        col = f"{axis}_g"
        row.update(extract_axis_features(segment[col].to_numpy(dtype=float), sample_rate, band_edges, axis))

    return row


def make_features(df: pd.DataFrame, window_seconds: float, band_edges: list[float], axes: list[str]) -> pd.DataFrame:
    sample_rate = infer_sample_rate(df)
    window_size = int(round(sample_rate * window_seconds))
    if window_size < 32:
        raise ValueError("분석 기간 창이 너무 짧습니다. 최소 약 1초 이상으로 설정하세요.")
    
    rows = []
    for segment_id, start in enumerate(range(0, len(df) - window_size + 1, window_size)):
        segment = df.iloc[start : start + window_size].copy()
        rows.append(extract_segment_features(segment, sample_rate, segment_id, band_edges, axes))
    return pd.DataFrame(rows)

def environmental_design_matrix(features: pd.DataFrame, degree: int) -> np.ndarray:
    temp = features["temp_c"].to_numpy(dtype=float)
    humidity = features["humidity"].to_numpy(dtype=float)
    cols = [temp, humidity, np.ones_like(temp)]
    if degree >= 2:
        cols = [temp, humidity, temp**2, humidity**2, temp * humidity, np.ones_like(temp)]
    return np.column_stack(cols)


def fit_temperature_models(normal: pd.DataFrame, columns: list[str], degree: int) -> dict[str, np.ndarray]:
    models = {}
    x = environmental_design_matrix(normal, degree)
    models = {}
    for col in columns:
        y = normal[col].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        models[col] = coef
    return models


def apply_temperature_compensation(
        features: pd.DataFrame, 
        models: dict[str, np.ndarray],
        degree: int,
) -> pd.DataFrame:
    compensated = features.copy()
    x = environmental_design_matrix(compensated, degree)
    for col, coef in models.items():
        predicted = x @ coef
        compensated[f"{col}_env_pred"] = predicted
        compensated[f"{col}_residual"] = compensated[col] - predicted
    return compensated


def fit_pca_constructor(X: np.ndarray, keep_components: int) -> dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    z = (X - mean) / std
    _, _, vh = np.linalg.svd(z, full_matrices=False)
    components = vh[:keep_components]
    return {"mean": mean, "std": std, "components": components}


def reconstruction_error(X: np.ndarray, model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    z = (X - model["mean"]) / model["std"]
    components = model["components"]
    projected = z @ components.T @ components
    residual = z - projected
    return np.mean(residual**2, axis=1), residual**2


def band_contribution_columns(residual_cols: list[str], band_edges: list[float], axes: list[str]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for name in band_names_from_edges(band_edges):
        cols = []
        for axis in axes:
            col = f"{axis}_{name}_energy_residual"
            if col in residual_cols:
                cols.append(col)
        mapping[name] = cols
    return mapping


def normalize_axes(text: str) -> list[str]:
    aliases = {"x" : "ax", "y": "ay", "z": "az", "ax": "ax", "ay": "ay", "az": "az"}
    axes = []
    for raw_axis in text.split(","):
        raw_axis = raw_axis.strip().lower()
        if not raw_axis:
            continue
        if raw_axis not in aliases:
            raise ValueError(f"알 수 없는 분석 축입니다: {raw_axis}. 사용가능: ax, ay, az 또는 x, y, z")
        axes.append(aliases[raw_axis])
    return axes


def run_pipeline(args: argparse.Namespace) -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path

    if args.generate_sample:
        generate_sample_data(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"입력 CSV 파일을 찾지 못했습니다: {input_path}\n\n"
            "처음 테스트라면 아래처럼 샘플 데이터를 먼저 생성해서 실행하세요.\n"
            "  python bridge_anomaly_demo.py --generate-sample\n\n"
            "실제 센서 데이터를 분석하려면 라즈베리파이 수집 코드로 CSV를 만든 뒤 아래처럼 실행하세요.\n"
            " python bridge_anomaly_demo.py --input data/real_bridge_log.csv"
        )
    df = pd.read_csv(input_path)
    required = {"ms", "temp_c", "humidity", "ax_g", "ay_g", "az_g"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 CSV 열이 없습니다: {sorted(missing)}")
    
    axes = normalize_axes(args.axes)
    for axis in axes:
        if f"{axis}_g" not in df.columns:
            raise ValueError(f"분석 축 {axis!r}에 필요한 열이 없습니다: {axis}_g")
        
    band_edges = parse_band_edges(args.band_edges)
    features = make_features(df, args.window_seconds, band_edges, axes)

    feature_cols = [
        col
        for col in features.columns
        if col
        not in {
            "segment_id",
            "start_ms",
            "end_ms",
            "temp_c",
            "humidity",
            "adxl_temp_c",
        }
    ]

    normal_count = max(4, int(len(features) * args.normal_fraction))
    normal = features.iloc[:normal_count].copy()
    
    temp_models = fit_temperature_models(normal, feature_cols, args.env_degree)
    compensated = apply_temperature_compensation(features, temp_models, args.env_degree)
    residual_cols = [f"{col}_residual" for col in feature_cols]

    raw_train = compensated.iloc[:normal_count][feature_cols].to_numpy(dtype=float)
    raw_all = compensated[feature_cols].to_numpy(dtype=float)
    residual_train = compensated.iloc[:normal_count][residual_cols].to_numpy(dtype=float)
    residual_all = compensated[residual_cols].to_numpy(dtype=float)

    raw_model = fit_pca_constructor(raw_train, keep_components=args.components)
    residual_model = fit_pca_constructor(residual_train, keep_components=args.components)

    raw_score, _ = reconstruction_error(raw_all, raw_model)
    residual_score, per_feature_error = reconstruction_error(residual_all, residual_model)
    raw_threshold = float(np.quantile(raw_score[:normal_count], args.threshold_quantile))
    threshold = float(np.quantile(residual_score[:normal_count], args.threshold_quantile))

    results = compensated.copy()
    results["raw_anomaly_score"] = raw_score
    results["raw_threshold"] = raw_threshold
    results["raw_score_norm"] = raw_score / raw_threshold if raw_threshold != 0 else raw_score
    results["anomaly_score"] = residual_score
    results["threshold"] = threshold
    results["anomaly_score_norm"] = residual_score / threshold if threshold != 0 else residual_score
    results["is_anomaly"] = results["anomaly_score"] > threshold
    results["score_reduction_after_env_comp"] = results["raw_score_norm"] - results["anomaly_score_norm"]
    results["env_comp_improved"] = results["score_reduction_after_env_comp"] > 0

    residual_index = {name: i for i, name in enumerate(residual_cols)}
    band_errors = {}
    for band, cols in band_contribution_columns(residual_cols, band_edges, axes).items():
        indices = [residual_index[col] for col in cols]
        if indices:
            band_errors[band] = per_feature_error[:, indices].sum(axis=1)
    band_error_df = pd.DataFrame(band_errors)
    if not band_error_df.empty:
        band_sum = band_error_df.sum(axis=1).replace(0, np.nan)
        for col in band_error_df.columns:
            results[f"{col}_contribution"] = band_error_df[col] / band_sum
        results["main_band"] = band_error_df.idxmax(axis=1)
    else:
        results["main_band"] = ""

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "segment_features.csv", index=False)
    results.to_csv(output_dir / "anomaly_results.csv", index=False)

    print("분석 완료.")
    print(f"입력 파일: {input_path}")
    print(f"전체 분석 구간 수: {len(results)}")
    print(f"정상 학습 구간 수: {normal_count}")
    print(f"온습도 보정 차수: {args.env_degree}")
    print(f"보정 전 이상 판단 기준선: {raw_threshold:.6f}")
    print(f"이상 판단 기준선: {threshold:.6f}")
    print(f"이상으로 판단된 구간 수: {int(results["is_anomaly"].sum())}")
    print(f"피처 CSV: {output_dir / "segment_features.csv"}")
    print(f"결과 CSV: {output_dir / "anomaly_results.csv"}")
    print()
    print("최근 구간 요약: ")
    print(
        results[
            [
                "segment_id", 
                "temp_c",
                "humidity",
                "raw_anomaly_score",
                "raw_score_norm",
                "anomaly_score",
                "anomaly_score_norm", 
                "is_anomaly",
                "main_band",
            ]
        ]
        .tail(12)
        .to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_bridge_log.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--generate-sample", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--normal-fraction", type=float, default=0.45)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--env-degree", type=int, choices=[1, 2], default=1)
    parser.add_argument("--band-edges", default=DEFAULT_BAND_EDGES)
    parser.add_argument("--axes", default="ax,ay,az")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())