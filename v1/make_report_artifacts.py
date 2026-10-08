from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


DEFAULT_CONDITIONS = ["normal", "env_change", "damage", "damage_env"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/anomaly_results.csv", help="anomaly_results.csv path")
    parser.add_argument("--output-dir", default="", help="Output folders")
    parser.add_argument("--condition-column", default="", help="Optional condition column name in the input CSV")
    parser.add_argument(
        "--condition-names",
        default=",".join(DEFAULT_CONDITIONS),
        help="Names used when no condition column exists. The data is split into equal phases."
    )
    return parser.parse_args()


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base_dir / path


def add_condition_column(df: pd.DataFrame, condition_column: str, condition_name: list[str]) -> pd.DataFrame:
    out = df.copy()
    if condition_column and condition_column in out.columns:
        out["condition"] = out[condition_column].astype(str)
        return out
    
    n = len(out)
    if n == 0:
        out["condition"] = []
        return out
    
    names = condition_name or DEFAULT_CONDITIONS
    condition = []
    for i in range(n):
        idx = min(int(i * len(names) / n), len(names) - 1)
        condition.append(names[idx])
    out["condition"] = condition
    return out


def make_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("condition", sort=False)
        .agg(
            count=("segment_id", "count"),
            anomaly_count=("is_anomaly", "sum"),
            anomaly_rate=("is_anomaly", "mean"),
            mean_raw_score_norm=("raw_score_norm", "mean"),
            mean_anomaly_score_norm=("anomaly_score_norm", "mean"),
            mean_score_reduction=("score_reduction_after_env_comp", "mean"),
            mean_temp_c=("temp_c", "mean"),
            mean_humidity=("humidity", "mean"),
            main_band_mode=("main_band", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
        )
        .reset_index()
    )
    return grouped
    

def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(points: list[tuple[float, float]], color: str, width: int = 2) -> str:
    text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700">{html.escape(title)}</text>'
    ]


def add_axes(parts: list[str], left: int, top: int, right: int, bottom: int) -> None:
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333" stroke-width="1"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333" stroke-width="1"/>')


def save_anomaly_score_svg(df: pd.DataFrame, path: Path) -> None:
    width, height = 980, 420
    left, right, top, bottom = 70, 940, 55, 300
    x_values = list(range(len(df)))
    max_y = max(float(df["raw_score_norm"].max()), float(df["anomaly_score_norm"].max()), 1.0)
    max_y *= 1.08

    parts = svg_header(width, height, "Anomaly Score Before/After Temperature-Humidity Compensation")
    add_axes(parts, left, top, right, bottom)

    raw_points = [
        (scale(i, 0, max(len(df) - 1, 1), left, right), scale(v, 0, max_y, bottom, top))
        for i, v in zip(x_values, df["raw_score_norm"])
    ]
    comp_points = [
        (scale(i, 0, max(len(df) - 1, 1), left, right), scale(v, 0, max_y, bottom, top))
        for i, v in zip(x_values, df["anomaly_score_norm"])
    ]
    parts.append(polyline(raw_points, "#9b5de5", 2))
    parts.append(polyline(comp_points, "#0077b6", 2))
    threshold_y = scale(1.0, 0, max_y, bottom, top)
    parts.append(f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{right}" y2="{threshold_y:.1f}" stroke="#d62828" stroke-width="1.5" stroke-dasharray="6 5"/>')
    parts.append('<text x="760" y="30" font-family="Arial, sans-serif" font-size="13" fill="#9b5de5">raw normalized</text>')
    parts.append('<text x="760" y="50" font-family="Arial, sans-serif" font-size="13" fill="#0077b6">compensated normalized</text>')
    parts.append('<text x="760" y="70" font-family="Arial, sans-serif" font-size="13" fill="#d62828">threshold = 1.0</text>')
    parts.append(f'<text x="{left}" y="392" font-family="Arial, sans-serif" font-size="12">segment</text>')
    parts.append(f'<text x="18" y="{top + 16}" font-family="Arial, sans-serif" font-size="12">score</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def save_environment_svg(df: pd.DataFrame, path: Path) -> None:
    width, height = 980, 420
    left, right, top, bottom = 70, 940, 55, 360
    x_values = list(range(len(df)))
    min_y = min(float(df["temp_c"].min()), float(df["humidity"].min()))
    max_y = max(float(df["temp_c"].max()), float(df["humidity"].max()))
    pad = max((max_y - min_y) * 0.08, 1.0)
    min_y -= pad
    max_y += pad

    parts = svg_header(width, height, "Temperature and Humidity During Measurement")
    add_axes(parts, left, top, right, bottom)
    temp_points = [
        (scale(i, 0, max(len(df) - 1, 1), left, right), scale(v, min_y, max_y, bottom, top))
        for i, v in zip(x_values, df["temp_c"])
    ]
    hum_points = [
        (scale(i, 0, max(len(df) - 1, 1), left, right), scale(v, min_y, max_y, bottom, top))
        for i, v in zip(x_values, df["humidity"])
    ]
    parts.append(polyline(temp_points, "#e76f51", 2))
    parts.append(polyline(hum_points, "#2a9d8f", 2))
    parts.append('<text x="760" y="30" font-family="Arial, sans-serif" font-size="13" fill="#e76f51">temperature C</text>')
    parts.append('<text x="760" y="50" font-family="Arial, sans-serif" font-size="13" fill="#2a9d8f">humidity %RH C</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding='utf-8')


def save_band_contribution_svg(df: pd.DataFrame, path: Path) -> None:
    contribution_cols = [c for c in df.columns if c.endswith("_contribution")]
    width, height = 980, 420
    left, right, top, bottom = 80, 940, 65, 350
    parts = svg_header(width, height, "Average Frequency-Band Contribution")
    add_axes(parts, left, top, right, bottom)

    if not contribution_cols:
        parts.append('<text x="90" y="110" font-family="Arial, sans-serif" font-size="14">No contribution columns found.</text>')
        parts.append('</svg>')
        path.write_text("\n".join(parts), encoding='utf-8')
        return

    means = df[contribution_cols].mean().sort_values(ascending=False)
    bar_gap = 16
    bar_width = (right - left - bar_gap * (len(means) - 1)) / max(len(means), 1)
    colors = ["#0077b6", "#f77f00", "#9b5de5", "#d62828"]
    for i, (name, value) in enumerate(means.items()):
        x = left + i * (bar_width + bar_gap)
        y = scale(float(value), 0, max(float(means.max()), 1.0), bottom, top)
        h = bottom - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{h:.1f}" fill="{colors[i % len(colors)]}"/>')
        label = name.replace("_contribution", "").replace("band_", "")
        parts.append(f'<text x="{x:.1f}" y="{bottom + 24}" font-family="Arial, sans-serif" font-size="11">{html.escape(label)}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y - 6:.1f}" font-family="Arial, sans-serif" font-size="11">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_summary_markdown(df: pd.DataFrame, condition_summary: pd.DataFrame, output_path: Path) -> None:
    anomaly_count = int(df["is_anomaly"].sum())
    total = len(df)
    improvement_rate = float(df["env_comp_improved"].mean()) if "env_comp_improved" in df else 0.0
    top_band = df["main_band"].mode().iloc[0] if "main_band" in df and not df["main_band"].mode().empty else ""

    lines = [
        "# 전체 결과 요약",
        "",
        f"- 전체 구간 수: {total}",
        f"- 이상 판단 구간 수: {anomaly_count}",
        f"- 이상 판단 비율: {anomaly_count / total:.2f}" if total else "- 이상 판단 비율: N/A",
        f"- 온습도 보정 후 정규화 이상점수가 감소한 구간 비율: {improvement_rate:.2%}",
        f"- 가장 자주 나타난 주여 기여 대역: `{top_band}`",
        "",
        "## 조건별 요약",
        "",
        dataframe_to_markdown(condition_summary),
        "",
        "## 생성된 그래프",
        "",
        "- `anomaly_scores.svg`",
        "- `environment.svg`",
        "- `band_contributions.svg`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    columns = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(str(c) for c in columns) + " |")
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = resolve_path(args.input, script_dir)
    output_dir = resolve_path(args.output_dir, script_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    condition_names = [part.strip() for part in args.condition_names.split(",") if part.strip()]
    df = add_condition_column(df, args.condition_column, condition_names)
    condition_summary = make_condition_summary(df)

    condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)
    save_anomaly_score_svg(df, output_dir / "anomaly_scores.svg")
    save_environment_svg(df, output_dir / "environment.svg")
    save_band_contribution_svg(df, output_dir / "band_contributions.svg")
    write_summary_markdown(df, condition_summary, output_dir / "summary.md")

    print("Report artifacts created.")
    print(f"Input: {input_path}")
    print(f"Output folder: {output_dir}")
    print(f"Condition summary: {output_dir / 'condition_summary.csv'}")
    print(f"Summary markdown: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()