from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import core


def tap_events_path(bridge_csv: Path) -> Path:
    name = bridge_csv.name
    if name.endswith("_bridge_log.csv"):
        return bridge_csv.with_name(name[: -len("_bridge_log.csv")] + "_tap_events.csv")
    return bridge_csv.with_name(bridge_csv.stem + "_tap_events.csv")


def load_raw(bridge_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(bridge_csv)

    # logger3 형식(ms 열 없음) 지원: t_perf_ns / t_unix_s 로부터 ms 생성
    if "ms" not in df.columns:
        if "t_perf_ns" in df.columns:
            t = pd.to_numeric(df["t_perf_ns"], errors="coerce").to_numpy(dtype=float)
            df["ms"] = (t - t[0]) / 1e6
        elif "t_unix_s" in df.columns:
            t = pd.to_numeric(df["t_unix_s"], errors="coerce").to_numpy(dtype=float)
            df["ms"] = (t - t[0]) * 1000.0
        else:
            raise ValueError(f"{bridge_csv.name}: ms / t_perf_ns / t_unix_s 열이 없어 시간축을 만들 수 없습니다")

    # 별도 tap_events CSV -> core.py가 기대하는 tapper_event 열 주입
    tap_csv = tap_events_path(bridge_csv)
    df["tapper_event"] = 0
    if tap_csv.exists():
        taps = pd.read_csv(tap_csv)
        pairs = [
            ("pulse_start_perf_ns", "t_perf_ns"),
            ("t_perf_ns", "t_perf_ns"),
            ("pulse_start_unix_s", "t_unix_s"),
            ("t_unix_s", "t_unix_s"),
        ]
        injected = False
        for tap_col, log_col in pairs:
            if tap_col in taps.columns and log_col in df.columns:
                tap_times = pd.to_numeric(taps[tap_col], errors="coerce").dropna().to_numpy(dtype=float)
                log_times = pd.to_numeric(df[log_col], errors="coerce").to_numpy(dtype=float)
                idx = np.searchsorted(log_times, tap_times)
                idx = idx[(idx >= 0) & (idx < len(df))]
                df.loc[df.index[idx], "tapper_event"] = 1
                print(f"  {bridge_csv.name}: 타격 {len(idx)}회 매칭 ({tap_col})")
                injected = True
                break
        if not injected:
            print(f"  경고: {tap_csv.name}의 시각 열을 매칭하지 못해 슬라이딩 윈도우로 분석합니다")
    else:
        print(f"  참고: {tap_csv.name} 없음 -> 슬라이딩 윈도우로 분석합니다")
    return df


def build_table(files: list[str], config: core.FeatureConfig) -> pd.DataFrame:
    parts = []
    for path_text in files:
        path = Path(path_text)
        df = load_raw(path)
        table = core.extract_feature_table_from_dataframe(df, config, source_file=path.name)
        print(f"  {path.name}: 분석 창 {len(table)}개 추출")
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def cmd_train(args: argparse.Namespace) -> None:
    config = core.FeatureConfig(
        window_seconds=args.window,
        post_tap_delay_seconds=args.delay,
        band_edges=core.parse_band_edges(args.band_edges),
        axes=core.normalize_axes(args.axes),
    )
    print(f"특징 설정: 축 {config.axes}, 대역 경계 {config.band_edges} Hz")
    print("[1/3] 정상 데이터 특징 추출")
    table = build_table(args.normal, config)
    print(f"[2/3] 학습 시작 (총 {len(table)}개 창)")
    model = core.train_sentry_model(table, config, epochs=args.epochs, random_state=args.seed)
    print("[3/3] 모델 저장")
    path = core.save_model(model, args.model)
    env = model["environment"]
    print(f"모델 저장: {path}")
    print(f"환경 보정: {'사용' if env['active'] else '미사용 (온습도 변화 폭 부족)'} "
          f"(온도 폭 {env['ranges']['temp_range_c']:.1f}C, 습도 폭 {env['ranges']['humidity_range_pct']:.1f}%)")
    for branch, values in model["branches"].items():
        print(f"임계값[{branch}]: {values['threshold']:.5f}")


def cmd_score(args: argparse.Namespace) -> None:
    model = core.load_model(args.model)
    config = core.model_config(model)
    for path_text in args.input:
        path = Path(path_text)
        print(f"\n=== {path.name} ===")
        df = load_raw(path)
        table, preds = core.predict_dataframe(model, df, source_file=path.name)
        statuses = [p.status for p in preds]
        norms = np.array([p.compensated_score_norm for p in preds], dtype=float)
        bands = pd.Series([p.main_band for p in preds])
        print(f"분석 창 {len(preds)}개 | normal {statuses.count('normal')} / "
              f"warning {statuses.count('warning')} / anomaly {statuses.count('anomaly')}")
        print(f"점수/임계값 비율: 중앙값 {np.median(norms):.2f}, 최대 {np.max(norms):.2f} "
              f"(1.0 이상 = 이상 후보)")
        print(f"주요 기여 대역: {bands.mode().iloc[0]}")
        if args.out:
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            result = table[["window_id", "source_file", "start_ms", "end_ms", "temp_c", "humidity"]].copy()
            result["raw_score_norm"] = [p.raw_score_norm for p in preds]
            result["compensated_score_norm"] = [p.compensated_score_norm for p in preds]
            result["status"] = statuses
            result["main_band"] = [p.main_band for p in preds]
            out_path = out_dir / (path.stem + "_scores.csv")
            result.to_csv(out_path, index=False)
            print(f"결과 저장: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="core.py 실행용 스크립트 (학습/판정)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="정상 데이터로 모델 학습")
    p_train.add_argument("--normal", nargs="+", required=True, help="정상 bridge_log CSV (여러 개 가능)")
    p_train.add_argument("--model", default="sentry_core_model.joblib")
    p_train.add_argument("--window", type=float, default=2.0)
    p_train.add_argument("--delay", type=float, default=0.05)
    p_train.add_argument("--band-edges", default="0,6,12,18,25,50")
    p_train.add_argument("--axes", default="ax,ay,az")
    p_train.add_argument("--epochs", type=int, default=500)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_score = sub.add_parser("score", help="학습된 모델로 조건 파일 판정")
    p_score.add_argument("--model", default="sentry_core_model.joblib")
    p_score.add_argument("--input", nargs="+", required=True, help="판정할 bridge_log CSV (여러 개 가능)")
    p_score.add_argument("--out", default="", help="창별 점수 CSV 저장 폴더 (선택)")
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
