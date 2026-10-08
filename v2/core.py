from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import math
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

MODEL_VERSION = 2
REQUIRED_RAW_COLUMNS = {"ms", "temp_c", "humidity", "ax_g", "ay_g", "az_g"}


class DenseAutoencoder:
    def __init__(
            self,
            *,
            hidden_dim: int = 8,
            latent_dim: int = 3,
            learning_rate: float = 1e-3,
            epochs: int = 500,
            batch_size: int = 128,
            patience: int = 50,
            weight_decay: float = 1e-5,
            validation_fraction: float = 0.15,
            min_delta: float = 1e-7,
            gradient_clip: float = 5.0,
            random_state: int = 42,
    ) -> None:
        if hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")
        if latent_dim < 1:
            raise ValueError("latent_dim must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs < 1 or batch_size < 1 or patience < 1:
            raise ValueError("epochs, batch_size, and patience must be positive")
        if not 0.0 <= validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.weight_decay = float(weight_decay)
        self.validation_fraction = float(validation_fraction)
        self.min_delta = float(min_delta)
        self.gradient_clip = float(gradient_clip)
        self.random_state = int(random_state)

        self.input_dim_: int | None = None
        self.hidden_dim_: int | None = None
        self.latent_dim_: int | None = None
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.training_history_: list[dict[str, float]] = []
        self.n_iter_: int = 0
        self.best_validation_loss_: float = math.inf

    @staticmethod
    def _tanh_grad(activated: np.ndarray) -> np.ndarray:
        return 1.0 - activated**2
    
    def _initialize(self, input_dim: int, rng: np.random.Generator) -> None:
        latent = min(self.latent_dim, max(1, input_dim - 1))
        hidden = max(latent + 1, self.hidden_dim)
        dims = [input_dim, hidden, latent, hidden, input_dim]
        self.input_dim_ = input_dim
        self.hidden_dim_ = hidden
        self.latent_dim_ = latent
        self.weights_ = []
        self.biases_ = []
        for fan_in, fan_out in zip(dims, dims[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.weights_.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float64))
            self.biases_.append(np.zeros(fan_out, dtype=np.float64))

    def _check_fitted(self) -> None:
        if self.input_dim_ is None or len(self.weights_) != 4:
            raise RuntimeError("Autoencoder has not been fitted")
        
    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        self._check_fitted()
        activations = [X]
        preactivations: list[np.ndarray] = []
        current = X
        for layer in range(3):
            z = current @ self.weights_[layer] + self.biases_[layer]
            current = np.tanh(z)
            preactivations.append(z)
            activations.append(current)
        z_out = current @ self.weights_[3] + self.biases_[3]
        preactivations.append(z_out)
        activations.append(z_out)
        return activations, preactivations
    
    def _mse(self, X: np.ndarray) -> float:
        reconstructed = self._forward(X)[0][-1]
        mse = float(np.mean((reconstructed - X) ** 2))
        penalty = self.weight_decay * sum(float(np.sum(w**2)) for w in self.weights_)
        return mse + penalty
    
    def fit(self, X: np.ndarray) -> "DenseAutoencoder":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array")
        if len(X) < 8:
            raise ValueError("at least 8 training rows are required for the Autoencoder")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values")
        
        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(X))
        if len(X) >= 12 and self.validation_fraction > 0:
            n_valid = max(2, int(round(len(X) * self.validation_fraction)))
            n_valid = min(n_valid, len(X) - 4)
            valid_idx = order[:n_valid]
            train_idx = order[n_valid:]
            X_valid = X[valid_idx]
            X_train = X[train_idx]
        else:
            X_train = X
            X_valid = X

        self._initialize(X.shape[1], rng)
        m_w = [np.zeros_like(w) for w in self.weights_]
        v_w = [np.zeros_like(w) for w in self.weights_]
        m_b = [np.zeros_like(b) for b in self.biases_]
        v_b = [np.zeros_like(b) for b in self.biases_]
        beta1, beta2, eps = 0.9, 0.9999, 1e-8
        step = 0
        best_weights = [w.copy() for w in self.weights_]
        best_biases = [b.copy() for b in self.biases_]
        best_loss = math.inf
        epochs_without_improvement = 0
        self.training_history_ = []

        for epoch in range(1, self.epochs + 1):
            shuffled = rng.permutation(len(X_train))
            for start in range(0, len(X_train), self.batch_size):
                batch = X_train[shuffled[start : start + self.batch_size]]
                activations, _ = self._forward(batch)
                reconstructed = activations[-1]
                scale = 2.0 / (len(batch) * batch.shape[1])
                delta = (reconstructed - batch) * scale

                grad_w = [np.zeros_like(w) for w in self.weights_]
                grad_b = [np.zeros_like(b) for b in self.biases_]
                grad_w[3] = activations[3].T @ delta + 2.0 * self.weight_decay * self.weights_[3]
                grad_b[3] = np.sum(delta, axis=0)

                back = delta @ self.weights_[3].T
                for layer in (2, 1, 0):
                    back = back * self._tanh_grad(activations[layer + 1])
                    grad_w[layer] = activations[layer].T @ back + 2.0 * self.weight_decay * self.weights_[layer]
                    grad_b[layer] = np.sum(back, axis=0)
                    if layer > 0:
                        back = back @ self.weights_[layer].T

                total_norm_sq = sum(float(np.sum(g**2)) for g in grad_w + grad_b)
                total_norm = math.sqrt(total_norm_sq)
                if self.gradient_clip > 0 and total_norm > self.gradient_clip:
                    factor = self.gradient_clip / (total_norm + 1e-12)
                    grad_w = [g * factor for g in grad_w]
                    grad_b = [g * factor for g in grad_b]

                step += 1
                for i in range(4):
                    m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * grad_w[i]
                    v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * (grad_w[i] ** 2)
                    m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * grad_b[i]
                    v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (grad_b[i] ** 2)
                    m_w_hat = m_w[i] / (1.0 - beta1**step)
                    v_w_hat = v_w[i] / (1.0 - beta2**step)
                    m_b_hat = m_b[i] / (1.0 - beta1**step)
                    v_b_hat = v_b[i] / (1.0 - beta2**step)
                    self.weights_[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
                    self.biases_[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

            train_loss = self._mse(X_train)
            valid_loss = self._mse(X_valid)
            self.training_history_.append(
                {"epoch": float(epoch), "train_loss": train_loss, "validation_loss": valid_loss}
            )
            if valid_loss < best_loss - self.min_delta:
                best_loss = valid_loss
                best_weights = [w.copy() for w in self.weights_]
                best_biases = [b.copy() for b in self.biases_]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.patience:
                break

        self.weights_ = best_weights
        self.biases_ = best_biases
        self.n_iter_ = len(self.training_history_)
        self.best_validation_loss_ = float(best_loss)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._check_fitted()
        if X.shape[1] != self.input_dim_:
            raise ValueError(f"expected {self.input_dim_} features, received {X.shape[1]}")
        return self._forward(X)[0][-1]
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._check_fitted()
        return self._forward(X)[0][2]
    
    def summary(self) -> dict[str, Any]:
        self._check_fitted()
        return {
            "architecture": [self.input_dim_, self.hidden_dim_, self.latent_dim_, self.hidden_dim_, self.input_dim_],
            "activation": "tanh",
            "output_activation": "linear",
            "optimizer": "Adam",
            "epochs_run": self.n_iter_,
            "best_validation_loss": self.best_validation_loss_,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
        }
    

@dataclass(frozen=True)
class FeatureConfig:
    window_seconds: float = 2.0
    post_tap_delay_seconds: float = 0.05
    hop_seconds: float = 1.0
    band_edges: tuple[float, ...] = (0.0, 6.0, 12.0, 18.0, 25.0, 50.0)
    axes: tuple[str, ...] = ("ax", "ay", "az")
    min_peak_hz: float = 0.5

    def validate(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        if len(self.band_edges) < 2:
            raise ValueError("band_edges must contain at least two values")
        if any(b <= a for a, b in zip(self.band_edges, self.band_edges[1:])):
            raise ValueError("band_edges must be strictly increasing")
        if not self.axes:
            raise ValueError("at least one analysis axis is required")
        for axis in self.axes:
            if axis not in {"ax", "ay", "az"}:
                raise ValueError(f"unsupported axis: {axis}")
            

@dataclass
class Prediction:
    raw_score: float
    raw_score_norm: float
    compensated_score: float
    compensated_score_norm: float
    status: str
    main_band: str
    band_contributions: dict[str, float]
    feature_errors: dict[str, float]
    env_compensation_active: bool


def parse_band_edges(text: str) -> tuple[float, ...]:
    edges = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    FeatureConfig(band_edges=edges).validate()
    return edges


def normalize_axes(text: str) -> tuple[str, ...]:
    aliases = {"x": "ax", "y": "ay", "z": "az", "ax": "ax", "ay": "ay", "az": "az"}
    axes: list[str] = []
    for item in text.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError(f"unsupported axis: {item}")
        axis = aliases[key]
        if axis not in axes:
            axes.append(axis)
    return tuple(axes)


def band_name(lo: float, hi: float) -> str:
    return f"band_{lo:g}_{hi:g}hz"


def infer_sample_rate(df: pd.DataFrame) -> float:
    ms = pd.to_numeric(df["ms"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(ms)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) < 2:
        raise ValueError("cannot infer sample rate from the ms column")
    return 1000.0 / float(np.median(dt))


def _spectral_entropy(power: np.ndarray) -> float:
    power = np.asarray(power, dtype=float)
    total = float(np.sum(power))
    if total <= 0 or len(power) <= 1:
        return 0.0
    p = power / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / math.log2(len(power)))


def _fill_env(series: pd.Series, fallback: float) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().any():
        return float(values.interpolate(limit_direction="both").mean())
    return float(fallback)


def extract_axis_features(
        signal: np.ndarray,
        sample_rate: float,
        band_edges: tuple[float, ...],
        prefix: str,
        min_peak_hz: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 32:
        raise ValueError("analysis window is too short")
    signal = signal - float(np.mean(signal))
    window = np.hanning(len(signal))
    fft = np.fft.rfft(signal * window)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    power = np.abs(fft) ** 2

    upper = min(float(band_edges[-1]), sample_rate / 2.0)
    usable = (freqs >= max(min_peak_hz, float(band_edges[0]))) & (freqs <= upper)
    if not np.any(usable):
        usable = freqs > 0
    usable_freqs = freqs[usable]
    usable_power = power[usable]
    peak_idx = int(np.argmax(usable_power))

    row: dict[str, float] = {
        f"{prefix}_rms": float(np.sqrt(np.mean(signal**2))),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_peak_to_peak": float(np.ptp(signal)),
        f"{prefix}_spectral_entropy": _spectral_entropy(usable_power),
        f"{prefix}_peak_freq_hz": float(usable_freqs[peak_idx]),
        f"{prefix}_peak_power": float(usable_power[peak_idx]),
    }

    for lo, hi in zip(band_edges, band_edges[1:]):
        mask = (freqs >= lo) & (freqs < min(hi, sample_rate / 2.0 + 1e-9))
        row[f"{prefix}_{band_name(lo, hi)}_energy"] = float(np.sum(power[mask]))
    return row, freqs, power


def extract_feature_row(
        df: pd.DataFrame,
        config: FeatureConfig,
        *,
        sample_rate: float | None = None,
        fallback_temp_c: float = 25.0,
        fallback_humidity: float = 50.0,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    config.validate()
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing raw columns: {sorted(missing)}")
    fs = float(sample_rate or infer_sample_rate(df))
    row: dict[str, float] = {
        "temp_c": _fill_env(df["temp_c"], fallback_temp_c),
        "humidity": _fill_env(df["humidity"], fallback_humidity),
        "sample_rate_hz": fs,
    }
    primary_freqs = np.array([], dtype=float)
    primary_power = np.array([], dtype=float)
    for axis in config.axes:
        features, freqs, power = extract_axis_features(
            pd.to_numeric(df[f"{axis}_g"], errors="coerce").interpolate(limit_direction="both").to_numpy(dtype=float),
            fs,
            config.band_edges,
            axis,
            config.min_peak_hz,
        )
        row.update(features)
        if axis == "az" or primary_freqs.size == 0:
            primary_freqs, primary_power = freqs, power
    return row, primary_freqs, primary_power


def _event_windows(df: pd.DataFrame, config: FeatureConfig, sample_rate: float) -> list[pd.DataFrame]:
    event_col = "tapper_event"
    if event_col not in df.columns:
        return []
    events = np.flatnonzero(pd.to_numeric(df[event_col], errors="coerce").fillna(0).to_numpy() > 0)
    if len(events) == 0:
        return []
    delay = int(round(config.post_tap_delay_seconds * sample_rate))
    length = int(round(config.window_seconds * sample_rate))
    windows: list[pd.DataFrame] = []
    for idx in events:
        start = idx + delay
        end = start + length
        if start >= 0 and end <= len(df):
            windows.append(df.iloc[start:end].copy())
    return windows


def extract_feature_table_from_dataframe(
        df: pd.DataFrame,
        config: FeatureConfig,
        *,
        source_file: str = "",
        prefer_tapper_events: bool = True,
) -> pd.DataFrame:
    config.validate()
    missing = REQUIRED_RAW_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing raw columns: {sorted(missing)}")
    fs = infer_sample_rate(df)
    windows = _event_windows(df, config, fs) if prefer_tapper_events else []
    if not windows:
        window_size = int(round(config.window_seconds * fs))
        hop_size = max(1, int(round(config.hop_seconds * fs)))
        windows = [df.iloc[start : start + window_size].copy() for start in range(0, len(df) - window_size + 1, hop_size)]

    rows: list[dict[str, Any]] = []
    for window_id, window in enumerate(windows):
        row, _, _ = extract_feature_row(window, config, sample_rate=fs)
        row.update(
            window_id=window_id,
            source_file=source_file,
            start_ms=float(window["ms"].iloc[0]),
            end_ms=float(window["ms"].iloc[-1]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(table: pd.DataFrame) -> list[str]:
    excluded = {"window_id", "source_file", "start_ms", "end_ms", "temp_c", "humidity", "sample_rate_hz"}
    return [c for c in table.columns if c not in excluded and pd.api.types.is_numeric_dtype(table[c])]


def _fit_reconstructor(
        X: np.ndarray,
        *,
        hidden_dim: int,
        latent_dim: int,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        patience: int,
        weight_decay: float,
        random_state: int,
) -> DenseAutoencoder:
    model = DenseAutoencoder(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        weight_decay=weight_decay,
        random_state=random_state,
    )
    return model.fit(X)


def _reconstruct(model: DenseAutoencoder, X: np.ndarray, model_type: str) -> np.ndarray:
    if model_type != "autoencoder":
        raise ValueError(f"This package only supports the nonlinear Autoencoder, received: {model_type}")
    return np.asarray(model.predict(X), dtype=float)


def _reconstruction_errors(
        model: DenseAutoencoder,
        X: np.ndarray,
        model_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    reconstructed = _reconstruct(model, X, model_type)
    per_feature = (X - reconstructed) ** 2
    return np.mean(per_feature, axis=1), per_feature

def _environment_matrix(poly: PolynomialFeatures, temp: np.ndarray, humidity: np.ndarray) -> np.ndarray:
    base = np.column_stack([temp, humidity])
    return poly.transform(base)


def _fit_environment_models(
        train: pd.DataFrame,
        names: list[str],
        degree: int,
        min_temp_range: float,
        min_humidity_range: float,
) -> tuple[bool, PolynomialFeatures | None, dict[str, LinearRegression], dict[str, float]]:
    temp = train["temp_c"].to_numpy(dtype=float)
    humidity = train["humidity"].to_numpy(dtype=float)
    ranges = {
        "temp_range_c": float(np.nanmax(temp) - np.nanmin(temp)),
        "humidity_range_pct": float(np.nanmax(humidity) - np.nanmin(humidity)),
    }
    active = ranges["temp_range_c"] >= min_temp_range or ranges["humidity_range_pct"] >= min_humidity_range
    if not active:
        return False, None, {}, ranges
    poly = PolynomialFeatures(degree=degree, include_bias=True)
    X_env = poly.fit_transform(np.column_stack([temp, humidity]))
    models: dict[str, LinearRegression] = {}
    for name in names:
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_env, train[name].to_numpy(dtype=float))
        models[name] = reg
    return True, poly, models, ranges


def _apply_environment(
        table: pd.DataFrame,
        names: list[str],
        active: bool,
        poly: PolynomialFeatures | None,
        models: dict[str, LinearRegression],
        feature_means: dict[str, float],
) -> np.ndarray:
    raw = table[names].to_numpy(dtype=float)
    if not active or poly is None:
        return raw - np.array([feature_means[name] for name in names], dtype=float)
    X_env = _environment_matrix(poly, table["temp_c"].to_numpy(dtype=float), table["humidity"].to_numpy(dtype=float))
    predicted = np.column_stack([models[name].predict(X_env) for name in names])
    return raw - predicted


def _split_train_validation(table: pd.DataFrame, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = [s for s in table["source_file"].dropna().unique().tolist() if s]
    if len(sources) >= 2:
        n_valid_sources = max(1, int(round(len(sources) * validation_fraction)))
        valid_sources = set(sources[-n_valid_sources:])
        valid = table[table["source_file"].isin(valid_sources)].copy()
        train = table[~table["source_file"].isin(valid_sources)].copy()
    else:
        cut = max(1, int(round(len(table) * (1.0 - validation_fraction))))
        cut = min(cut, len(table) - 1) if len(table) > 1 else 1
        train = table.iloc[:cut].copy()
        valid = table.iloc[cut:].copy()
    if valid.empty:
        valid = train.copy()
    return train, valid


def train_sentry_model(
        feature_table: pd.DataFrame,
        config: FeatureConfig,
        *,
        model_type: str = "autoencoder",
        threshold_quantile: float = 0.99,
        environment_degree: int = 2,
        min_temp_range_c: float = 2.0,
        min_humidity_range_pct: float = 5.0,
        validation_fraction: float = 0.15,
        hidden_dim: int = 8,
        latent_dim: int = 3,
        learning_rate: float = 1e-3,
        epochs: int = 500,
        batch_size: int = 128,
        patience: int = 50,
        weight_decay: float = 1e-5,
        random_state: int = 42,
) -> dict[str, Any]:
    config.validate()
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1")
    names = feature_columns(feature_table)
    if not names:
        raise ValueError("feature table has no trainable feature columns")
    if len(feature_table) < 30:
        raise ValueError("training requires at least 30 feature windows")
    table = feature_table.reset_index(drop=True).copy()
    if "source_file" not in table.columns:
        table["source_file"] = ""
    train, valid = _split_train_validation(table, validation_fraction)
    feature_means = {name: float(train[name].mean()) for name in names}
    env_active, poly, env_models, env_ranges = _fit_environment_models(
        train,
        names,
        environment_degree,
        min_temp_range_c,
        min_humidity_range_pct,
    )

    branches: dict[str, dict[str, Any]] = {}
    for branch, active in (("raw", False), ("compensated", env_active)):
        train_matrix = _apply_environment(train, names, active, poly, env_models, feature_means)
        valid_matrix = _apply_environment(valid, names, active, poly, env_models, feature_means)
        scaler = StandardScaler().fit(train_matrix)
        model = _fit_reconstructor(
            scaler.transform(train_matrix),
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            weight_decay=weight_decay,
            random_state=random_state,
        )
        errors, _ = _reconstruction_errors(model, scaler.transform(valid_matrix), model_type)
        branches[branch] = {
            "scaler": scaler,
            "model": model,
            "threshold": float(np.quantile(errors, threshold_quantile)),
        }

    return {
        "model_version": MODEL_VERSION,
        "model_type": model_type,
        "config": asdict(config),
        "feature_names": names,
        "feature_means": feature_means,
        "environment": {
            "active": env_active,
            "degree": environment_degree,
            "poly": poly,
            "models": env_models,
            "ranges": env_ranges,
        },
        "branches": branches,
        "training": {
            "n_windows": int(len(table)),
            "n_train": int(len(train)),
            "n_validation": int(len(valid)),
            "threshold_quantile": threshold_quantile,
            "random_state": random_state,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def model_config(model: dict[str, Any]) -> FeatureConfig:
    data = dict(model["config"])
    data["band_edges"] = tuple(float(value) for value in data["band_edges"])
    data["axes"] = tuple(str(value) for value in data["axes"])
    return FeatureConfig(**data)


def save_model(model: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    card = {
        "model_version": model["model_version"],
        "model_type": model["model_type"],
        "config": model["config"],
        "feature_names": model["feature_names"],
        "environment": {
            "active": model["environment"]["active"],
            "degree": model["environment"]["degree"],
            "ranges": model["environment"]["ranges"],
        },
        "thresholds": {branch: values["threshold"] for branch, values in model["branches"].items()},
        "training": model["training"],
        "runtime": model["runtime"],
    }
    path.with_suffix(".json").write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_model(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, dict):
        raise ValueError("model file does not contain a SENTRY model bundle")
    version = model.get("model_version")
    if version != MODEL_VERSION:
        raise ValueError(f"unsupported model version: {version} (expected {MODEL_VERSION})")
    return model


def _status_from_norm(norm: float) -> str:
    if not math.isfinite(norm):
        return "unknown"
    if norm < 0.8:
        return "normal"
    if norm < 1.0:
        return "warning"
    return "anomaly"


def _band_contributions(feature_errors: dict[str, float], config: FeatureConfig) -> dict[str, float]:
    totals: dict[str, float] = {}
    for lo, hi in zip(config.band_edges, config.band_edges[1:]):
        name = band_name(lo, hi)
        totals[name] = float(sum(value for key, value in feature_errors.items() if f"_{name}_" in key))
    grand = float(sum(totals.values()))
    if grand <= 0:
        return {name: 0.0 for name in totals}
    return {name: value / grand for name, value in totals.items()}


def _score_branch(
        model: dict[str, Any],
        branch: str,
        table: pd.DataFrame,
        active: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    env = model["environment"]
    matrix = _apply_environment(table, model["feature_names"], active, env["poly"], env["models"], model["feature_means"])
    bundle = model["branches"][branch]
    scaled = bundle["scaler"].transform(matrix)
    errors, per_feature = _reconstruction_errors(bundle["model"], scaled, model["model_type"])
    return errors, per_feature, float(bundle["threshold"])


def predict_feature_table(model: dict[str, Any], feature_table: pd.DataFrame) -> list[Prediction]:
    config = model_config(model)
    names = model["feature_names"]
    missing = set(names) - set(feature_table.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")
    env_active = bool(model["environment"]["active"])
    if env_active and not {"temp_c", "humidity"} <= set(feature_table.columns):
        raise ValueError("environment compensation requires temp_c and humidity columns")
    table = feature_table.reset_index(drop=True).copy()
    raw_errors, _, raw_threshold = _score_branch(model, "raw", table, False)
    comp_errors, comp_per_feature, comp_threshold = _score_branch(model, "compensated", table, env_active)

    predictions: list[Prediction] = []
    for index in range(len(table)):
        feature_errors = {name: float(comp_per_feature[index, j]) for j, name in enumerate(names)}
        contributions = _band_contributions(feature_errors, config)
        main_band = max(contributions, key=contributions.get) if contributions else ""
        raw_norm = float(raw_errors[index] / raw_threshold) if raw_threshold > 0 else math.nan
        comp_norm = float(comp_errors[index] / comp_threshold) if comp_threshold > 0 else math.nan
        predictions.append(
            Prediction(
                raw_score=float(raw_errors[index]),
                raw_score_norm=raw_norm,
                compensated_score=float(comp_errors[index]),
                compensated_score_norm=comp_norm,
                status=_status_from_norm(comp_norm),
                main_band=main_band,
                band_contributions=contributions,
                feature_errors=feature_errors,
                env_compensation_active=env_active,
            )
        )
    return predictions


def predict_dataframe(
        model: dict[str, Any],
        df: pd.DataFrame,
        *,
        source_file: str = "",
        prefer_tapper_events: bool = True,
) -> tuple[pd.DataFrame, list[Prediction]]:
    config = model_config(model)
    table = extract_feature_table_from_dataframe(
        df,
        config,
        source_file=source_file,
        prefer_tapper_events=prefer_tapper_events,
    )
    if table.empty:
        raise ValueError("no analysis windows could be extracted from the raw data")
    return table, predict_feature_table(model, table)
