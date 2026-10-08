from __future__ import annotations

import argparse
import struct
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MI_KNOWN = {1, 2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18}
MX_CELL = 1
MX_STRUCT = 2
MX_CHAR = 4
MX_DOUBLE = 6


class MatV5Reader:
    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()
        self.endian = "<" if self.data[126:128] == b"IM" else ">"
        self.dtype_map = {
            1: ("i1", 1),
            2: ("u1", 1),
            3: (self.endian + "i2", 2),
            4: (self.endian + "u2", 2),
            5: (self.endian + "i4", 4),
            6: (self.endian + "u4", 4),
            7: (self.endian + "f4", 4),
            9: (self.endian + "f8", 8),
            12: (self.endian + "i8", 8),
            13: (self.endian + "u8", 8),
        }

    @staticmethod
    def align8(offset: int) -> int:
        return (offset + 7) // 8 * 8

    def read_tag(self, buf: bytes, offset: int) -> tuple[int, int, int, int, bool, int]:
        raw = struct.unpack(self.endian + "I", buf[offset : offset + 4])[0]
        small_type = raw & 0xFFFF
        small_size = raw >> 16
        if small_size and small_type in MI_KNOWN:
            return small_type, small_size, offset + 4, offset + 4 + small_size, True, offset + 8

        data_type, size = struct.unpack(self.endian + "II", buf[offset : offset + 8])
        return data_type, size, offset + 8, offset + 8 + size, False, offset + 8 + size

    def read_numeric(self, buf: bytes, start: int, end: int, data_type: int, dims: list[int]) -> np.ndarray:
        dtype_text, _ = self.dtype_map[data_type]
        arr = np.frombuffer(buf[start:end], dtype=np.dtype(dtype_text)).copy()
        if dims and int(np.prod(dims)) == arr.size:
            return arr.reshape(tuple(dims), order="F")
        return arr

    def parse_matrix(self, buf: bytes, offset: int = 0) -> tuple[dict[str, Any], int]:
        data_type, _size, start, end, _small, _next_pos = self.read_tag(buf, offset)
        if data_type != 14:
            raise ValueError(f"Expected miMATRIX at {offset}, got type {data_type}")

        pos = start

        _, _, flag_start, flag_end, small, next_pos = self.read_tag(buf, pos)
        pos = next_pos if small else self.align8(flag_end)
        matlab_class = buf[flag_start]

        _, dim_size, dim_start, dim_end, small, next_pos = self.read_tag(buf, pos)
        pos = next_pos if small else self.align8(dim_end)
        dims = list(struct.unpack(self.endian + "i" * (dim_size // 4), buf[dim_start:dim_end])) if dim_size else []

        _, name_size, name_start, name_end, small, next_pos = self.read_tag(buf, pos)
        pos = next_pos if small else self.align8(name_end)
        name = buf[name_start:name_end].decode("utf-8", "replace") if name_size else ""

        if matlab_class == MX_DOUBLE:
            data_type, _, data_start, data_end, _small, _next_pos = self.read_tag(buf, pos)
            value = self.read_numeric(buf, data_start, data_end, data_type, dims)
            return {"name": name, "class": "double", "dims": dims, "value": value}, self.align8(end)

        if matlab_class == MX_CHAR:
            _, _, text_start, text_end, _small, _next_pos = self.read_tag(buf, pos)
            text = buf[text_start:text_end].decode("utf-8", "replace")
            return {"name": name, "class": "char", "dims": dims, "value": text}, self.align8(end)

        if matlab_class == MX_CELL:
            values = []
            for _ in range(int(np.prod(dims))):
                child, pos = self.parse_matrix(buf, pos)
                values.append(child)
            return {"name": name, "class": "cell", "dims": dims, "value": values}, self.align8(end)

        if matlab_class == MX_STRUCT:
            _, _, fn_len_start, fn_len_end, small, next_pos = self.read_tag(buf, pos)
            pos = next_pos if small else self.align8(fn_len_end)
            field_name_length = int(np.frombuffer(buf[fn_len_start:fn_len_end], dtype=np.dtype(self.endian + "i4"))[0])

            _, _, fields_start, fields_end, small, next_pos = self.read_tag(buf, pos)
            pos = next_pos if small else self.align8(fields_end)
            raw_fields = buf[fields_start:fields_end]

            fields = []
            for idx in range(0, len(raw_fields), field_name_length):
                field = raw_fields[idx : idx + field_name_length].split(b"\x00", 1)[0].decode("utf-8", "replace")
                if field:
                    fields.append(field)

            values = {field: [] for field in fields}
            for _ in range(int(np.prod(dims))):
                for field in fields:
                    child, pos = self.parse_matrix(buf, pos)
                    values[field].append(child)

            return {"name": name, "class": "struct", "dims": dims, "fields": fields, "values": values}, self.align8(end)

        raise ValueError(f"Unsupported MATLAB class {matlab_class} for variable {name!r}")

    def load_first_variable(self) -> dict[str, Any]:
        data_type, _size, start, end, _small, _next_pos = self.read_tag(self.data, 128)
        payload = self.data[start:end]
        if data_type == 15:
            payload = zlib.decompress(payload)
            variable, _ = self.parse_matrix(payload, 0)
            return variable
        if data_type == 14:
            variable, _ = self.parse_matrix(self.data, 128)
            return variable
        raise ValueError(f"Unsupported top-level MAT element type: {data_type}")


def matlab_datenum_to_datetime(value: float) -> datetime:
    days = int(value)
    fraction = value - days
    return datetime.fromordinal(days - 366) + timedelta(days=fraction)


def cell_texts(cell_value: list[dict[str, Any]]) -> list[str]:
    return [str(item["value"]) for item in cell_value]


def load_kw51_dataframe(mat_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    root = MatV5Reader(mat_path).load_first_variable()
    values = root["values"]

    sdn = values["sdn"][0]["value"].ravel()
    frequencies = values["f"][0]["value"]
    damping = values["xi"][0]["value"]
    environment = values["env"][0]["value"]
    env_labels = cell_texts(values["labels_env"][0]["value"])
    sensor_labels = cell_texts(values["labels_m"][0]["value"])

    df = pd.DataFrame({"datetime": [matlab_datenum_to_datetime(x) for x in sdn], "sdn": sdn})

    for idx in range(frequencies.shape[1]):
        df[f"f_mode_{idx + 1}"] = frequencies[:, idx]
        df[f"xi_mode_{idx + 1}"] = damping[:, idx]

    for idx, label in enumerate(env_labels):
        df[label] = environment[:, idx]

    if "tBD31A" in df and "tVL" in df:
        df["temp_c"] = df["tBD31A"].combine_first(df["tVL"])
    elif "tBD31A" in df:
        df["temp_c"] = df["tBD31A"]
    else:
        df["temp_c"] = df["tVL"]

    if "rhBD31A" in df and "rhVL" in df:
        df["humidity"] = df["rhBD31A"].combine_first(df["rhVL"])
    elif "rhBD31A" in df:
        df["humidity"] = df["rhBD31A"]
    else:
        df["humidity"] = df["rhVL"]

    return df, env_labels, sensor_labels


def select_frequency_columns(df: pd.DataFrame, min_valid_ratio: float) -> list[str]:
    frequency_cols = [col for col in df.columns if col.startswith("f_mode_")]
    selected = [col for col in frequency_cols if df[col].notna().mean() >= min_valid_ratio]
    if not selected:
        selected = sorted(frequency_cols, key=lambda col: df[col].notna().mean(), reverse=True)[:5]
    return selected


def prepare_analysis_frame(df: pd.DataFrame, frequency_cols: list[str], interpolation_limit: int) -> pd.DataFrame:
    out = df[["datetime", "sdn", "temp_c", "humidity", *frequency_cols]].copy()
    out = out.sort_values("datetime")
    out[["temp_c", "humidity", *frequency_cols]] = out[["temp_c", "humidity", *frequency_cols]].interpolate(
        limit=interpolation_limit,
        limit_direction="both",
    )
    out = out.dropna(subset=["temp_c", "humidity", *frequency_cols]).reset_index(drop=True)
    return out


def design_matrix(frame: pd.DataFrame, variables: list[str]) -> np.ndarray:
    if not variables:
        return np.ones((len(frame), 1))
    cols = [frame[var].to_numpy(dtype=float) for var in variables]
    cols.append(np.ones(len(frame)))
    return np.column_stack(cols)


def compensate_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
    env_vars: list[str],
    train_mask: np.ndarray,
) -> pd.DataFrame:
    compensated = frame.copy()
    if not env_vars:
        for col in feature_cols:
            compensated[f"{col}_residual"] = compensated[col]
        return compensated

    x_train = design_matrix(compensated.loc[train_mask], env_vars)
    x_all = design_matrix(compensated, env_vars)
    for col in feature_cols:
        y_train = compensated.loc[train_mask, col].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
        compensated[f"{col}_env_pred"] = x_all @ coef
        compensated[f"{col}_residual"] = compensated[col] - compensated[f"{col}_env_pred"]
    return compensated


def fit_autoencoder(
    x: np.ndarray,
    latent_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    z = (x - mean) / std

    input_dim = z.shape[1]
    latent_dim = max(1, min(latent_dim, input_dim, len(z)))
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.08, size=(input_dim, latent_dim))
    b1 = np.zeros(latent_dim)
    w2 = rng.normal(0, 0.08, size=(latent_dim, input_dim))
    b2 = np.zeros(input_dim)
    n = max(1, len(z))

    for _ in range(max(1, epochs)):
        hidden_linear = z @ w1 + b1
        hidden = np.tanh(hidden_linear)
        reconstructed = hidden @ w2 + b2
        error = reconstructed - z

        grad_reconstructed = (2.0 / n) * error
        grad_w2 = hidden.T @ grad_reconstructed
        grad_b2 = grad_reconstructed.sum(axis=0)
        grad_hidden = grad_reconstructed @ w2.T
        grad_hidden_linear = grad_hidden * (1.0 - hidden**2)
        grad_w1 = z.T @ grad_hidden_linear
        grad_b1 = grad_hidden_linear.sum(axis=0)

        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2

    return {"mean": mean, "std": std, "w1": w1, "b1": b1, "w2": w2, "b2": b2}


def reconstruction_error(x: np.ndarray, model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    z = (x - model["mean"]) / model["std"]
    hidden = np.tanh(z @ model["w1"] + model["b1"])
    reconstructed = hidden @ model["w2"] + model["b2"]
    residual = z - reconstructed
    return np.mean(residual**2, axis=1), residual**2


def make_change_candidate_label(frame: pd.DataFrame, feature_cols: list[str], train_mask: np.ndarray, quantile: float) -> np.ndarray:
    baseline = frame.loc[train_mask, feature_cols].median()
    scale = frame.loc[train_mask, feature_cols].std().replace(0, np.nan)
    deviation = ((frame[feature_cols] - baseline).abs() / scale).mean(axis=1).replace([np.inf, -np.inf], np.nan)
    deviation = deviation.fillna(0.0)
    threshold = float(deviation.quantile(quantile))
    return deviation.to_numpy() >= threshold


def classification_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    predicted = predicted.astype(bool)
    actual = actual.astype(bool)
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "False Alarm Rate": false_alarm_rate,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
    }


def run_model(
    name: str,
    env_vars: list[str],
    frame: pd.DataFrame,
    feature_cols: list[str],
    train_mask: np.ndarray,
    change_label: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    compensated = compensate_features(frame, feature_cols, env_vars, train_mask)
    residual_cols = [f"{col}_residual" for col in feature_cols]

    x_train = compensated.loc[train_mask, residual_cols].to_numpy(dtype=float)
    x_all = compensated[residual_cols].to_numpy(dtype=float)
    model = fit_autoencoder(x_train, args.latent_dim, args.ae_epochs, args.ae_learning_rate, args.seed)
    scores, per_feature_error = reconstruction_error(x_all, model)
    threshold = float(np.quantile(scores[train_mask], args.threshold_quantile))
    predicted = scores > threshold

    metrics = classification_metrics(predicted, change_label)
    normal_period_alarm_rate = float(predicted[train_mask].mean())

    score_frame = pd.DataFrame(
        {
            "datetime": frame["datetime"],
            "model": name,
            "anomaly_score": scores,
            "threshold": threshold,
            "is_anomaly": predicted,
            "change_candidate": change_label,
        }
    )
    for col, errors in zip(feature_cols, per_feature_error.T):
        score_frame[f"{col}_reconstruction_error"] = errors

    summary = {
        "모델": name,
        "입력 환경 변수": "없음" if not env_vars else " + ".join(env_vars),
        "False Alarm Rate": metrics["False Alarm Rate"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-score": metrics["F1-score"],
        "normal_period_alarm_rate": normal_period_alarm_rate,
        "threshold": threshold,
        "anomaly_ratio": float(predicted.mean()),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
    }
    return score_frame, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../kw51/trackedmodes.mat")
    parser.add_argument("--output-dir", default="outputs_kw51")
    parser.add_argument("--min-valid-ratio", type=float, default=0.75)
    parser.add_argument("--interpolation-limit", type=int, default=24)
    parser.add_argument("--normal-fraction", type=float, default=0.5)
    parser.add_argument("--change-quantile", type=float, default=0.85)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--ae-epochs", type=int, default=800)
    parser.add_argument("--ae-learning-rate", type=float, default=0.01)
    parser.add_argument("--threshold-quantile", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    mat_path = Path(args.input)
    if not mat_path.is_absolute():
        mat_path = (script_dir / mat_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df, env_labels, sensor_labels = load_kw51_dataframe(mat_path)
    frequency_cols = select_frequency_columns(raw_df, args.min_valid_ratio)
    frame = prepare_analysis_frame(raw_df, frequency_cols, args.interpolation_limit)

    normal_count = max(20, int(len(frame) * args.normal_fraction))
    train_mask = np.zeros(len(frame), dtype=bool)
    train_mask[:normal_count] = True
    change_label = make_change_candidate_label(frame, frequency_cols, train_mask, args.change_quantile)

    model_specs = [
        ("무보정 AE", []),
        ("온도 보정 AE", ["temp_c"]),
        ("온도·습도 보정 AE", ["temp_c", "humidity"]),
    ]

    all_scores = []
    summaries = []
    for offset, (name, env_vars) in enumerate(model_specs):
        args.seed += offset
        scores, summary = run_model(name, env_vars, frame, frequency_cols, train_mask, change_label, args)
        all_scores.append(scores)
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    score_df = pd.concat(all_scores, ignore_index=True)

    raw_df.to_csv(output_dir / "kw51_extracted_raw.csv", index=False)
    frame.assign(change_candidate=change_label, is_train_period=train_mask).to_csv(output_dir / "kw51_analysis_frame.csv", index=False)
    score_df.to_csv(output_dir / "kw51_autoencoder_scores.csv", index=False)
    summary_df.to_csv(output_dir / "kw51_model_summary.csv", index=False)

    print("KW51 분석 완료.")
    print(f"입력 파일: {mat_path}")
    print(f"분석 기간: {frame['datetime'].min()} ~ {frame['datetime'].max()}")
    print(f"분석 행 수: {len(frame)}")
    print(f"정상 기준 학습 행 수: {normal_count}")
    print(f"선택 주파수 피처: {', '.join(frequency_cols)}")
    print(f"결과 폴더: {output_dir}")
    print()
    print(summary_df[["모델", "입력 환경 변수", "False Alarm Rate", "Precision", "Recall", "F1-score"]].to_string(index=False))


if __name__ == "__main__":
    main()
