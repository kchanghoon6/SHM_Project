from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../SHM_Project/Z24 Benchmark")
    parser.add_argument("--output-dir", default="outputs_z24")
    parser.add_argument("--normal-label", type=int, default=0)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def spectral_entropy(power: np.ndarray) -> float:
    total = np.sum(power)
    if total <= 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(power)))


def band_energy(power: np.ndarray, start_frac: float, end_frac: float) -> float:
    n = len(power)
    start = max(1, int(n * start_frac))
    end = max(start + 1, int(n * end_frac))
    end = min(end, n)
    return float(np.sum(power[start:end]))


def extract_features_one(sample: np.ndarray) -> dict[str, float]:
    centered = sample - sample.mean(axis=1, keepdims=True)
    sensor_rms = np.sqrt(np.mean(centered**2, axis=1))
    sensor_std = np.std(centered, axis=1)
    sensor_ptp = np.ptp(centered, axis=1)

    window = np.hanning(centered.shape[1]).astype(centered.dtype)
    spectrum = np.fft.rfft(centered * window, axis=1)
    power_by_sensor = np.abs(spectrum) ** 2
    mean_power = power_by_sensor.mean(axis=0)

    usable_power = mean_power[1:]
    peak_bin = int(np.argmax(usable_power) + 1)
    total_power = float(np.sum(usable_power))

    return {
        "rms_mean": float(sensor_rms.mean()),
        "rms_std_across_sensors": float(sensor_rms.std()),
        "rms_max": float(sensor_rms.max()),
        "std_mean": float(sensor_std.mean()),
        "ptp_mean": float(sensor_ptp.mean()),
        "ptp_max": float(sensor_ptp.max()),
        "peak_bin": float(peak_bin),
        "peak_power": float(mean_power[peak_bin]),
        "total_power": total_power,
        "spectral_entropy": spectral_entropy(usable_power),
        "band_very_low": band_energy(mean_power, 0.00, 0.02),
        "band_low": band_energy(mean_power, 0.00, 0.08),
        "band_mid": band_energy(mean_power, 0.08, 0.25),
        "band_high": band_energy(mean_power, 0.25, 0.50),
    }


def build_feature_table(inputs: np.ndarray, labels: np.ndarray, max_samples: int) -> pd.DataFrame:
    n = len(labels) if max_samples <= 0 else min(max_samples, len(labels))
    rows = []
    for i in range(n):
        row = extract_features_one(np.asarray(inputs[i], dtype=np.float32))
        row["sample_id"] = i
        row["label"] = int(labels[i])
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f"피처 추출 중: {i + 1}/{n}")
    return pd.DataFrame(rows)


def fit_pca_reconstructor(X: np.ndarray, keep_components: int) -> dict[str, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    z = (X - mean) / std
    _, _, vh = np.linalg.svd(z, full_matrices=False)
    return {"mean": mean, "std": std, "components": vh[:keep_components]}


def reconstruction_error(X: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    z = (X - model["mean"]) / model["std"]
    components = model["components"]
    projected = z @ components.T @ components
    residual = z - projected
    return np.mean(residual**2, axis=1)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs_path = data_dir / "inputs.npy"
    labels_path = data_dir / "labels.npy"
    if not inputs_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"inputs.npy 또는 labels.npy를 찾지 못했습니다: {data_dir}")
    
    print("Z24 데이터 로딩 중...")
    inputs = np.load(inputs_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")
    print(f"inputs shape: {inputs.shape}")
    print(f"labels shape: {labels.shape}")

    features = build_feature_table(inputs, labels, args.max_samples)
    feature_cols = [c for c in features.columns if c not in {"sample_id", "label"}]

    normal_mask = features["label"] == args.normal_label
    if normal_mask.sum() < args.components + 2:
        raise ValueError("정상 학습 샘플이 너무 적습니다. --normal-label 또는 components 값을 확인하세요.")
    
    x_train = features.loc[normal_mask, feature_cols].to_numpy(dtype=float)
    x_all = features[feature_cols].to_numpy(dtype=float)

    model = fit_pca_reconstructor(x_train, args.components)
    train_score = reconstruction_error(x_train, model)
    threshold = float(np.quantile(train_score, args.threshold_quantile))
    score = reconstruction_error(x_all, model)

    results = features[["sample_id", "label"]].copy()
    results["anomaly_score"] = score
    results["threshold"] = threshold
    results["is_anomaly"] = results["anomaly_score"] > threshold
    results = pd.concat([results, features[feature_cols]], axis=1)

    summary = (
        results.groupby("label")
        .agg(
            count=("sample_id", "count"),
            anomaly_count=("is_anomaly", "sum"),
            anomaly_rate=("is_anomaly", "mean"),
            mean_score=("anomaly_score", "mean"),
            median_score=("anomaly_score", "median"),
            max_score=("anomaly_score", "max"),
        )
        .reset_index()
    )

    results.to_csv(output_dir / "z24_sample_scores.csv", index=False)
    summary.to_csv(output_dir / "z24_label_summary.csv", index=False)

    print()
    print("분석 완료")
    print(f"정상 기준 label: {args.normal_label}")
    print(f"정상 학습 샘플 수: {int(normal_mask.sum())}")
    print(f"이상 판단 기준선: {threshold:.6f}")
    print(f"샘플별 결과: {output_dir / 'z24_sample_scores.csv'}")
    print(f"label별 요약: {output_dir / 'z24_label_summary.csv'}")
    print()
    print("label별 요약")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()