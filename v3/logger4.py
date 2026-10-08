from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path
from typing import Final

import adafruit_dht
import board
from smbus2 import SMBus


ADXL355_ADDR_LOW: Final = 0x1D
ADXL355_ADDR_HIGH: Final = 0x53

REG_DEVID_AD: Final = 0x00
REG_TEMP2: Final = 0x06
REG_XDATA3: Final = 0x08
REG_FILTER: Final = 0x28
REG_RANGE: Final = 0x2C
REG_POWER_CTL: Final = 0x2D
REG_RESET: Final = 0x2F

EXPECTED_ID: Final = (0xAD, 0x1D, 0xED)

RANGE_TO_CODE = {2: 0x01, 4: 0x02, 8: 0x03}
RANGE_TO_SCALE_G_PER_LSB = {
    2: 2.048 / (2**19),
    4: 4.096 / (2**19),
    8: 8.192 / (2**19),
}
RANGE_TO_FULL_SCALE_G = {2: 2.048, 4: 4.096, 8: 8.192}

SAMPLE_RATE_TO_FILTER = {125: 0x05, 250: 0x04, 500: 0x03}

TEMP_BIAS_LSB: Final = 1852.0
TEMP_SLOPE_LSB_PER_C: Final = -9.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SENTRY 스크리닝 로거: ADXL355 + DHT22 연속 수집 (태퍼 없음)"
    )
    parser.add_argument(
        "--condition",
        default="B31-SCR-A",
        help="파일명 코드. 예: B31-SCR-A(스크리닝 후보 A), B31-Day0-QC",
    )
    parser.add_argument("--output-dir", default="data", help="CSV 저장 폴더")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="수집 시간(초). 0이면 Ctrl+C까지 무제한. 30분 스크리닝은 1800",
    )
    parser.add_argument("--bus", type=int, default=1, help="I2C 버스 번호")
    parser.add_argument(
        "--address",
        choices=["auto", "0x1d", "0x53"],
        default="auto",
        help="ADXL355 I2C 주소",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=sorted(SAMPLE_RATE_TO_FILTER),
        default=500,
        help="수집 fs(Hz). 스크리닝은 500 유지 — 분석용 다운샘플링은 후처리에서(5-D 규칙)",
    )
    parser.add_argument(
        "--range-g",
        type=int,
        choices=[2, 4, 8],
        default=2,
        help="측정 범위. 보행·바람 상시 미진동은 mg급이므로 분해능이 가장 좋은 ±2 g 기본. "
        "z축에는 중력 약 1 g이 실려 있으므로 종료 시 클리핑 QC로 여유 확인",
    )
    parser.add_argument("--dht-interval", type=float, default=2.0, help="DHT22 측정 간격(초)")
    parser.add_argument(
        "--adxl-temp-interval", type=float, default=1.0, help="ADXL355 내부 온도 측정 간격(초)"
    )
    parser.add_argument("--flush-seconds", type=float, default=1.0, help="CSV 디스크 flush 간격")
    parser.add_argument(
        "--status-interval",
        type=float,
        default=60.0,
        help="터미널 상태 로그 출력 간격(초). 0이면 출력 안 함",
    )
    parser.add_argument("--no-dht", action="store_true", help="DHT22 없이 가속도만 수집")
    parser.add_argument(
        "--surface-interval",
        type=float,
        default=2.0,
        help="DS18B20 표면온도 측정 간격(초). 12-bit 변환에 ~750 ms 소요되므로 1초 미만은 비권장",
    )
    parser.add_argument("--no-surface", action="store_true", help="DS18B20 표면온도 없이 수집")
    return parser.parse_args()


def twos_complement(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def detect_address(bus: SMBus, requested: str) -> int:
    candidates = (
        [ADXL355_ADDR_LOW, ADXL355_ADDR_HIGH]
        if requested == "auto"
        else [int(requested, 16)]
    )
    for address in candidates:
        try:
            device_id = tuple(bus.read_i2c_block_data(address, REG_DEVID_AD, 3))
        except OSError:
            continue
        if device_id == EXPECTED_ID:
            return address
    raise RuntimeError(
        "ADXL355를 찾지 못했습니다. i2cdetect -y 1에서 1d 또는 53이 보이는지 확인하세요."
    )


def configure_adxl355(bus: SMBus, address: int, range_g: int, sample_rate: int) -> None:
    bus.write_byte_data(address, REG_RESET, 0x52)
    time.sleep(0.1)
    bus.write_byte_data(address, REG_RANGE, 0x80 | RANGE_TO_CODE[range_g])
    bus.write_byte_data(address, REG_FILTER, SAMPLE_RATE_TO_FILTER[sample_rate])
    bus.write_byte_data(address, REG_POWER_CTL, 0x00)
    time.sleep(0.1)


def read_adxl_temperature_c(bus: SMBus, address: int) -> float:
    temp2, temp1 = bus.read_i2c_block_data(address, REG_TEMP2, 2)
    raw = ((temp2 & 0x0F) << 8) | temp1
    return 25.0 + (raw - TEMP_BIAS_LSB) / TEMP_SLOPE_LSB_PER_C


def read_acceleration_g(
    bus: SMBus, address: int, scale_g_per_lsb: float
) -> tuple[float, float, float]:
    data = bus.read_i2c_block_data(address, REG_XDATA3, 9)
    values: list[float] = []
    for i in range(0, 9, 3):
        raw20 = (data[i] << 12) | (data[i + 1] << 4) | (data[i + 2] >> 4)
        values.append(twos_complement(raw20, 20) * scale_g_per_lsb)
    return values[0], values[1], values[2]


def wait_until_ns(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 2_000_000:
            time.sleep((remaining_ns - 1_000_000) / 1e9)
        else:
            time.sleep(0)


def unix_from_perf_ns(perf_ns: int, anchor_perf_ns: int, anchor_unix_s: float) -> float:
    return anchor_unix_s + (perf_ns - anchor_perf_ns) / 1e9


def dht_worker(output_queue: mp.Queue, stop_event: mp.Event, interval_s: float) -> None:
    # DHT22 판독이 수집 루프를 막지 않도록 별도 프로세스에서 실행한다.
    dht = adafruit_dht.DHT22(board.D4)
    try:
        while not stop_event.is_set():
            measured_ns = time.perf_counter_ns()
            temp_c = math.nan
            humidity = math.nan
            try:
                temp = dht.temperature
                hum = dht.humidity
                if temp is not None:
                    temp_c = float(temp)
                if hum is not None:
                    humidity = float(hum)
            except RuntimeError:
                pass

            item = (measured_ns, temp_c, humidity)
            try:
                while True:
                    output_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                output_queue.put_nowait(item)
            except queue.Full:
                pass

            stop_event.wait(interval_s)
    finally:
        dht.exit()


DS18B20_DEVICES_DIR = Path("/sys/bus/w1/devices")


def find_ds18b20_device() -> Path:
    candidates = sorted(DS18B20_DEVICES_DIR.glob("28-*/w1_slave"))
    if not candidates:
        raise RuntimeError(
            "DS18B20을 찾지 못했습니다. /boot/firmware/config.txt에 "
            "dtoverlay=w1-gpio,gpiopin=17 을 추가하고 재부팅한 뒤 "
            "ls /sys/bus/w1/devices/ 에서 28-* 폴더를 확인하세요. "
            "(표면온도 없이 수집하려면 --no-surface)"
        )
    return candidates[0]


def read_ds18b20_c(device_path: Path) -> float:
    # 판독에 약 750 ms가 걸리므로 별도 프로세스에서 호출한다.
    try:
        text = device_path.read_text()
    except OSError:
        return math.nan
    lines = text.strip().splitlines()
    if len(lines) < 2 or not lines[0].strip().endswith("YES"):
        return math.nan
    _, _, raw = lines[1].partition("t=")
    try:
        value = int(raw.strip()) / 1000.0
    except ValueError:
        return math.nan
    if value == 85.0:
        return math.nan  # 파워온 리셋 기본값(85.0 C) -> 무효 처리
    return value


def surface_worker(output_queue: mp.Queue, stop_event: mp.Event, interval_s: float) -> None:
    device_path = find_ds18b20_device()
    while not stop_event.is_set():
        measured_ns = time.perf_counter_ns()
        surface_c = read_ds18b20_c(device_path)
        item = (measured_ns, surface_c)
        try:
            while True:
                output_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            output_queue.put_nowait(item)
        except queue.Full:
            pass
        stop_event.wait(interval_s)


def format_optional(value: float, digits: int = 3) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accel_path = output_dir / f"{args.condition}_bridge_log.csv"
    metadata_path = output_dir / f"{args.condition}_metadata.json"

    scale = RANGE_TO_SCALE_G_PER_LSB[args.range_g]
    full_scale_g = RANGE_TO_FULL_SCALE_G[args.range_g]
    clip_limit_g = 0.995 * full_scale_g
    sample_interval_ns = int(1e9 / args.sample_rate)

    dht_process: mp.Process | None = None
    dht_queue: mp.Queue | None = None
    dht_stop: mp.Event | None = None
    surface_process: mp.Process | None = None
    surface_queue: mp.Queue | None = None
    surface_stop: mp.Event | None = None

    latest_dht_temp = math.nan
    latest_humidity = math.nan
    latest_dht_perf_ns = 0
    latest_surface_temp = math.nan
    latest_surface_perf_ns = 0
    latest_adxl_temp = math.nan
    last_adxl_temp_ns = 0

    total_samples = 0
    missed_slots = 0
    clip_counts = {"ax": 0, "ay": 0, "az": 0}

    status_prev_samples = 0
    status_prev_missed = 0
    status_az_sum = 0.0
    status_az_sq_sum = 0.0

    try:
        if not args.no_dht:
            ctx = mp.get_context("spawn")
            dht_queue = ctx.Queue(maxsize=1)
            dht_stop = ctx.Event()
            dht_process = ctx.Process(
                target=dht_worker,
                args=(dht_queue, dht_stop, args.dht_interval),
                daemon=True,
            )
            dht_process.start()

        if not args.no_surface:
            find_ds18b20_device()
            ctx = mp.get_context("spawn")
            surface_queue = ctx.Queue(maxsize=1)
            surface_stop = ctx.Event()
            surface_process = ctx.Process(
                target=surface_worker,
                args=(surface_queue, surface_stop, args.surface_interval),
                daemon=True,
            )
            surface_process.start()

        with SMBus(args.bus) as bus:
            address = detect_address(bus, args.address)
            configure_adxl355(bus, address, args.range_g, args.sample_rate)

            anchor_perf_ns = time.perf_counter_ns()
            anchor_unix_s = time.time()
            end_ns = (
                None
                if args.duration_s <= 0
                else anchor_perf_ns + int(args.duration_s * 1e9)
            )

            metadata = {
                "condition": args.condition,
                "address_hex": f"0x{address:02X}",
                "sample_rate_hz": args.sample_rate,
                "filter_register_hex": f"0x{SAMPLE_RATE_TO_FILTER[args.sample_rate]:02X}",
                "range_g": args.range_g,
                "duration_s": args.duration_s,
                "tapper": False,
                "surface_sensor": not args.no_surface,
                "start_unix_s": anchor_unix_s,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(f"[수집] 조건: {args.condition} (태퍼 없음 — 상시 진동 연속 수집)")
            print(f"[수집] ADXL355 0x{address:02X}, ±{args.range_g} g, {args.sample_rate} Hz")
            if end_ns is None:
                print("[수집] 수집 시간: 무제한 (Ctrl+C로 종료)")
            else:
                print(f"[수집] 수집 시간: {args.duration_s:.0f} s 후 자동 종료")
            print(f"[수집] 가속도 CSV: {accel_path}")
            print(f"[수집] 표면온도(DS18B20): {'미사용' if args.no_surface else '사용 (1-Wire GPIO17)'}")
            if args.status_interval > 0:
                print(f"[수집] 상태 로그: {args.status_interval:.0f}초 간격으로 출력")

            next_sample_ns = time.perf_counter_ns()
            next_flush_ns = next_sample_ns + int(args.flush_seconds * 1e9)
            next_status_ns = (
                None
                if args.status_interval <= 0
                else next_sample_ns + int(args.status_interval * 1e9)
            )
            status_prev_ns = next_sample_ns

            with accel_path.open(
                "w", newline="", encoding="utf-8", buffering=1024 * 1024
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "sample_index",
                        "t_perf_ns",
                        "t_unix_s",
                        "scheduled_perf_ns",
                        "lateness_us",
                        "missed_slots_total",
                        "temp_c",
                        "humidity",
                        "dht_age_s",
                        "surface_temp_c",
                        "surface_age_s",
                        "adxl_temp_c",
                        "ax_g",
                        "ay_g",
                        "az_g",
                    ]
                )

                while True:
                    wait_until_ns(next_sample_ns)
                    now_ns = time.perf_counter_ns()

                    if end_ns is not None and now_ns >= end_ns:
                        print("[수집] 지정한 수집 시간 도달 — 자동 종료")
                        break

                    if now_ns - next_sample_ns >= sample_interval_ns:
                        skipped = int((now_ns - next_sample_ns) // sample_interval_ns)
                        missed_slots += skipped
                        next_sample_ns += skipped * sample_interval_ns

                    scheduled_ns = next_sample_ns

                    if dht_queue is not None:
                        try:
                            while True:
                                (
                                    latest_dht_perf_ns,
                                    temp_value,
                                    humidity_value,
                                ) = dht_queue.get_nowait()
                                if not math.isnan(temp_value):
                                    latest_dht_temp = temp_value
                                if not math.isnan(humidity_value):
                                    latest_humidity = humidity_value
                        except queue.Empty:
                            pass

                    if surface_queue is not None:
                        try:
                            while True:
                                (
                                    latest_surface_perf_ns,
                                    surface_value,
                                ) = surface_queue.get_nowait()
                                if not math.isnan(surface_value):
                                    latest_surface_temp = surface_value
                        except queue.Empty:
                            pass

                    if now_ns - last_adxl_temp_ns >= int(args.adxl_temp_interval * 1e9):
                        try:
                            latest_adxl_temp = read_adxl_temperature_c(bus, address)
                        except OSError:
                            latest_adxl_temp = math.nan
                        last_adxl_temp_ns = now_ns

                    read_start_ns = time.perf_counter_ns()
                    ax_g, ay_g, az_g = read_acceleration_g(bus, address, scale)
                    read_end_ns = time.perf_counter_ns()
                    sample_perf_ns = (read_start_ns + read_end_ns) // 2

                    for key, value in (("ax", ax_g), ("ay", ay_g), ("az", az_g)):
                        if abs(value) >= clip_limit_g:
                            clip_counts[key] += 1

                    status_az_sum += az_g
                    status_az_sq_sum += az_g * az_g

                    dht_age_s = (
                        math.nan
                        if latest_dht_perf_ns == 0
                        else (sample_perf_ns - latest_dht_perf_ns) / 1e9
                    )

                    surface_age_s = (
                        math.nan
                        if latest_surface_perf_ns == 0
                        else (sample_perf_ns - latest_surface_perf_ns) / 1e9
                    )

                    writer.writerow(
                        [
                            total_samples,
                            sample_perf_ns,
                            f"{unix_from_perf_ns(sample_perf_ns, anchor_perf_ns, anchor_unix_s):.9f}",
                            scheduled_ns,
                            f"{(sample_perf_ns - scheduled_ns) / 1000.0:.3f}",
                            missed_slots,
                            format_optional(latest_dht_temp),
                            format_optional(latest_humidity),
                            format_optional(dht_age_s, 3),
                            format_optional(latest_surface_temp),
                            format_optional(surface_age_s, 3),
                            format_optional(latest_adxl_temp),
                            f"{ax_g:.8f}",
                            f"{ay_g:.8f}",
                            f"{az_g:.8f}",
                        ]
                    )
                    total_samples += 1
                    next_sample_ns += sample_interval_ns

                    if read_end_ns >= next_flush_ns:
                        file.flush()
                        next_flush_ns = read_end_ns + int(args.flush_seconds * 1e9)

                    if next_status_ns is not None and read_end_ns >= next_status_ns:
                        span_s = (read_end_ns - status_prev_ns) / 1e9
                        n_new = total_samples - status_prev_samples
                        eff_hz = n_new / span_s if span_s > 0 else 0.0
                        if n_new > 0:
                            az_mean = status_az_sum / n_new
                            az_var = max(status_az_sq_sum / n_new - az_mean * az_mean, 0.0)
                            az_std_mg = (az_var**0.5) * 1000.0
                        else:
                            az_std_mg = 0.0
                        temp_str = format_optional(latest_dht_temp, 1) or "NA"
                        hum_str = format_optional(latest_humidity, 1) or "NA"
                        surf_str = format_optional(latest_surface_temp, 1) or "NA"
                        print(
                            f"[상태 {time.strftime('%H:%M:%S')}] "
                            f"경과 {(read_end_ns - anchor_perf_ns) / 6e10:.1f}분 | "
                            f"샘플 {total_samples:,} (구간 실효 {eff_hz:.1f} Hz) | "
                            f"누락 +{missed_slots - status_prev_missed} (누적 {missed_slots:,}) | "
                            f"클리핑 ax {clip_counts['ax']} / ay {clip_counts['ay']} / az {clip_counts['az']} | "
                            f"az 변동 {az_std_mg:.2f} mg | "
                            f"기온 {temp_str} C · 표면 {surf_str} C · 습도 {hum_str} % | "
                            f"파일 {accel_path.stat().st_size / 1e6:.1f} MB"
                        )
                        status_prev_ns = read_end_ns
                        status_prev_samples = total_samples
                        status_prev_missed = missed_slots
                        status_az_sum = 0.0
                        status_az_sq_sum = 0.0
                        next_status_ns = read_end_ns + int(args.status_interval * 1e9)

    except KeyboardInterrupt:
        print("\n[수집] Ctrl+C로 중단됨")
    finally:
        if dht_stop is not None:
            dht_stop.set()
        if dht_process is not None:
            dht_process.join(timeout=3.0)
            if dht_process.is_alive():
                dht_process.terminate()

        if surface_stop is not None:
            surface_stop.set()
        if surface_process is not None:
            surface_process.join(timeout=3.0)
            if surface_process.is_alive():
                surface_process.terminate()

        elapsed_s = total_samples / args.sample_rate if args.sample_rate else 0.0
        print("[수집] 종료")
        print(f"[QC] 저장 샘플: {total_samples:,} (약 {elapsed_s / 60.0:.1f}분)")
        print(f"[QC] 건너뛴 샘플 슬롯: {missed_slots:,}")
        print(
            "[QC] 포화(클리핑) 샘플 수: "
            f"ax={clip_counts['ax']}, ay={clip_counts['ay']}, az={clip_counts['az']}"
        )
        if any(clip_counts.values()):
            print("[경고] 클리핑 발생 — range-g를 한 단계 올려 재수집하세요 (해당 창은 분석 제외).")


if __name__ == "__main__":
    main()
