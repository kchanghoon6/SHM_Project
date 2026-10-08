from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import queue
import threading
import time
from pathlib import Path
from typing import Final

import adafruit_dht
import board
from gpiozero import OutputDevice
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
        description="SENTRY ADXL355 + DHT22 + solenoid tapper logger"
    )
    parser.add_argument(
        "--condition",
        default="Tx-Hx-M0-B0",
        help="조건 코드. 예: T0-H0-M500-B0",
    )
    parser.add_argument("--output-dir", default="data", help="CSV 저장 폴더")
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
        help="ADXL355 ODR와 로거 목표 샘플링 주파수. 최종 실험 전체에서 고정",
    )
    parser.add_argument(
        "--range-g",
        type=int,
        choices=[2, 4, 8],
        default=8,
        help="가속도 측정 범위",
    )
    parser.add_argument(
        "--dht-interval",
        type=float,
        default=2.0,
        help="DHT22 측정 간격(초)",
    )
    parser.add_argument(
        "--adxl-temp-interval",
        type=float,
        default=1.0,
        help="ADXL355 내부 온도 측정 간격(초)",
    )
    parser.add_argument(
        "--pulse-ms",
        type=float,
        default=40.0,
        help="태퍼 펄스폭(ms). 예비 실험에서 포화가 없도록 확정",
    )
    parser.add_argument(
        "--period-s",
        type=float,
        default=5.0,
        help="타격 시작 시점 간 주기(초)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="기록 전에 같은 펄스·같은 주기로 수행할 워밍업 타격 횟수",
    )
    parser.add_argument(
        "--pre-tap-seconds",
        type=float,
        default=2.0,
        help="기록 시작 후 첫 본 타격까지의 대기 시간",
    )
    parser.add_argument("--n-taps", type=int, default=35, help="본 타격 횟수")
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=5.0,
        help="마지막 타격 후 추가 수집 시간",
    )
    parser.add_argument(
        "--flush-seconds",
        type=float,
        default=1.0,
        help="CSV 디스크 flush 간격",
    )
    parser.add_argument(
        "--tapper-gpio",
        type=int,
        default=17,
        help="솔레노이드 드라이버 제어 BCM GPIO",
    )
    parser.add_argument(
        "--no-tapper",
        action="store_true",
        help="태퍼 없이 연속 수집. Ctrl+C로 종료",
    )
    parser.add_argument(
        "--no-dht",
        action="store_true",
        help="DHT22 없이 가속도만 수집",
    )
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
        "ADXL355를 찾지 못했습니다. i2cdetect -y 1에서 1d 또는 53이 "
        "보이는지 확인하세요."
    )


def configure_adxl355(
    bus: SMBus, address: int, range_g: int, sample_rate: int
) -> None:
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


def pulse_tapper(tap: OutputDevice, pulse_seconds: float) -> tuple[int, int]:
    start_ns = time.perf_counter_ns()
    tap.on()
    time.sleep(pulse_seconds)
    tap.off()
    end_ns = time.perf_counter_ns()
    return start_ns, end_ns


def run_warmup(
    tap: OutputDevice,
    warmup_count: int,
    period_s: float,
    pulse_s: float,
) -> None:
    if warmup_count <= 0:
        return
    print(
        f"[태퍼] 워밍업 {warmup_count}회: 본 실험과 같은 "
        f"{period_s:.3f} s 주기, {pulse_s * 1000:.1f} ms 펄스"
    )
    start_ns = time.perf_counter_ns() + 1_000_000_000
    period_ns = int(period_s * 1e9)
    for i in range(warmup_count):
        wait_until_ns(start_ns + i * period_ns)
        pulse_tapper(tap, pulse_s)


def unix_from_perf_ns(
    perf_ns: int, anchor_perf_ns: int, anchor_unix_s: float
) -> float:
    return anchor_unix_s + (perf_ns - anchor_perf_ns) / 1e9


def dht_worker(
    output_queue: mp.Queue,
    stop_event: mp.Event,
    interval_s: float,
) -> None:
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


def tapper_worker(
    args: argparse.Namespace,
    tap: OutputDevice,
    tap_path: Path,
    stop_event: threading.Event,
    anchor_perf_ns: int,
    anchor_unix_s: float,
    first_tap_ns: int,
) -> None:
    pulse_s = args.pulse_ms / 1000.0
    period_ns = int(args.period_s * 1e9)

    try:
        with tap_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "tap_index",
                    "scheduled_perf_ns",
                    "pulse_start_perf_ns",
                    "pulse_end_perf_ns",
                    "pulse_start_unix_s",
                    "start_delay_ms",
                    "condition",
                ]
            )

            print(
                f"[태퍼] 본 타격 {args.n_taps}회 시작 "
                f"({args.period_s:.3f} s start-to-start)"
            )
            for index in range(args.n_taps):
                scheduled_ns = first_tap_ns + index * period_ns
                wait_until_ns(scheduled_ns)
                start_ns, end_ns = pulse_tapper(tap, pulse_s)
                writer.writerow(
                    [
                        index,
                        scheduled_ns,
                        start_ns,
                        end_ns,
                        f"{unix_from_perf_ns(start_ns, anchor_perf_ns, anchor_unix_s):.9f}",
                        f"{(start_ns - scheduled_ns) / 1e6:.3f}",
                        args.condition,
                    ]
                )
                file.flush()

        print(
            f"[태퍼] 완료. 마지막 타격 후 {args.tail_seconds:.2f} s 추가 수집"
        )
        time.sleep(args.tail_seconds)
    finally:
        stop_event.set()


def format_optional(value: float, digits: int = 3) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    if args.period_s <= args.pulse_ms / 1000.0:
        raise ValueError("period-s는 pulse-ms보다 길어야 합니다.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accel_path = output_dir / f"{args.condition}_bridge_log.csv"
    tap_path = output_dir / f"{args.condition}_tap_events.csv"
    metadata_path = output_dir / f"{args.condition}_metadata.json"

    scale = RANGE_TO_SCALE_G_PER_LSB[args.range_g]
    full_scale_g = RANGE_TO_FULL_SCALE_G[args.range_g]
    clip_limit_g = 0.995 * full_scale_g
    sample_interval_ns = int(1e9 / args.sample_rate)

    tap: OutputDevice | None = None
    tapper_thread: threading.Thread | None = None
    dht_process: mp.Process | None = None
    dht_queue: mp.Queue | None = None
    dht_stop: mp.Event | None = None
    stop_event = threading.Event()

    latest_dht_temp = math.nan
    latest_humidity = math.nan
    latest_dht_perf_ns = 0
    latest_adxl_temp = math.nan
    last_adxl_temp_ns = 0

    total_samples = 0
    missed_slots = 0
    clip_counts = {"ax": 0, "ay": 0, "az": 0}

    try:
        if not args.no_tapper:
            tap = OutputDevice(
                args.tapper_gpio, active_high=True, initial_value=False
            )

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

        with SMBus(args.bus) as bus:
            address = detect_address(bus, args.address)
            configure_adxl355(
                bus, address, args.range_g, args.sample_rate
            )

            if not args.no_tapper:
                if tap is None:
                    raise RuntimeError("태퍼 GPIO 초기화에 실패했습니다.")
                run_warmup(
                    tap,
                    args.warmup,
                    args.period_s,
                    args.pulse_ms / 1000.0,
                )

            anchor_perf_ns = time.perf_counter_ns()
            anchor_unix_s = time.time()
            first_tap_ns = (
                anchor_perf_ns + int(args.pre_tap_seconds * 1e9)
            )

            metadata = {
                "condition": args.condition,
                "address_hex": f"0x{address:02X}",
                "sample_rate_hz": args.sample_rate,
                "filter_register_hex": (
                    f"0x{SAMPLE_RATE_TO_FILTER[args.sample_rate]:02X}"
                ),
                "range_g": args.range_g,
                "pulse_ms": args.pulse_ms,
                "period_s": args.period_s,
                "warmup_taps": args.warmup,
                "recorded_taps": 0 if args.no_tapper else args.n_taps,
                "pre_tap_seconds": args.pre_tap_seconds,
                "tail_seconds": args.tail_seconds,
                "start_unix_s": anchor_unix_s,
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(f"[수집] 조건: {args.condition}")
            print(
                f"[수집] ADXL355 0x{address:02X}, "
                f"±{args.range_g} g, {args.sample_rate} Hz"
            )
            print(f"[수집] 가속도 CSV: {accel_path}")
            if not args.no_tapper:
                print(f"[수집] 타격 CSV:  {tap_path}")

            if not args.no_tapper:
                tapper_thread = threading.Thread(
                    target=tapper_worker,
                    args=(
                        args,
                        tap,
                        tap_path,
                        stop_event,
                        anchor_perf_ns,
                        anchor_unix_s,
                        first_tap_ns,
                    ),
                    daemon=True,
                )
                tapper_thread.start()

            next_sample_ns = time.perf_counter_ns()
            next_flush_ns = (
                next_sample_ns + int(args.flush_seconds * 1e9)
            )

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
                        "adxl_temp_c",
                        "ax_g",
                        "ay_g",
                        "az_g",
                    ]
                )

                while not stop_event.is_set():
                    wait_until_ns(next_sample_ns)
                    now_ns = time.perf_counter_ns()


                    if now_ns - next_sample_ns >= sample_interval_ns:
                        skipped = int(
                            (now_ns - next_sample_ns) // sample_interval_ns
                        )
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

                    if (
                        now_ns - last_adxl_temp_ns
                        >= int(args.adxl_temp_interval * 1e9)
                    ):
                        try:
                            latest_adxl_temp = read_adxl_temperature_c(
                                bus, address
                            )
                        except OSError:
                            latest_adxl_temp = math.nan
                        last_adxl_temp_ns = now_ns

                    read_start_ns = time.perf_counter_ns()
                    ax_g, ay_g, az_g = read_acceleration_g(
                        bus, address, scale
                    )
                    read_end_ns = time.perf_counter_ns()
                    sample_perf_ns = (read_start_ns + read_end_ns) // 2

                    for key, value in (
                        ("ax", ax_g),
                        ("ay", ay_g),
                        ("az", az_g),
                    ):
                        if abs(value) >= clip_limit_g:
                            clip_counts[key] += 1

                    dht_age_s = (
                        math.nan
                        if latest_dht_perf_ns == 0
                        else (sample_perf_ns - latest_dht_perf_ns) / 1e9
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
                        next_flush_ns = (
                            read_end_ns + int(args.flush_seconds * 1e9)
                        )

    except KeyboardInterrupt:
        print("\n[수집] Ctrl+C로 중단됨")
    finally:
        stop_event.set()

        if tapper_thread is not None:
            tapper_thread.join(timeout=3.0)

        if dht_stop is not None:
            dht_stop.set()
        if dht_process is not None:
            dht_process.join(timeout=3.0)
            if dht_process.is_alive():
                dht_process.terminate()

        if tap is not None:
            tap.off()
            tap.close()

        print("[수집] 종료")
        print(f"[QC] 저장 샘플: {total_samples}")
        print(f"[QC] 건너뛴 샘플 슬롯: {missed_slots}")
        print(
            "[QC] 포화 샘플 수: "
            f"ax={clip_counts['ax']}, "
            f"ay={clip_counts['ay']}, "
            f"az={clip_counts['az']}"
        )
        if any(clip_counts.values()):
            print(
                "[경고] 포화가 발생했습니다. pulse-ms를 줄이거나 "
                "range-g를 높인 뒤 재수집하세요."
            )


if __name__ == "__main__":
    main()
