from __future__ import annotations

# 3차 실험: ADXL355 + DS18B20 표면온도 + DHT22 습도.
# 기본: 실행 시작 기준 3600초마다 CSV 교체. 수집 중 계속 기록/flush한다.
# temp_c는 호환성을 위해 DHT22 기온으로 유지하며 표면온도는 surface_temp_c이다.
# DS18B20: logger4.py와 같은 1-Wire 설정(GPIO17). 태퍼 코드는 사용하지 않는다.
import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import queue
import re
import signal
import tempfile
import threading
import time
from pathlib import Path
from typing import Final

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


def twos_complement(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def detect_address(bus: SMBus, requested: str) -> int:
    candidates = [ADXL355_ADDR_LOW, ADXL355_ADDR_HIGH] if requested == "auto" else [int(requested, 16)]
    for address in candidates:
        try:
            device_id = tuple(bus.read_i2c_block_data(address, REG_DEVID_AD, 3))
        except OSError:
            continue
        if device_id == EXPECTED_ID:
            return address
    raise RuntimeError("ADXL355를 찾지 못했습니다. i2cdetect -y 1에서 1d 또는 53을 확인해야 합니다.")


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


def read_acceleration_g(bus: SMBus, address: int, scale_g_per_lsb: float) -> tuple[float, float, float]:
    data = bus.read_i2c_block_data(address, REG_XDATA3, 9)
    values: list[float] = []
    for index in range(0, 9, 3):
        raw20 = (data[index] << 12) | (data[index + 1] << 4) | (data[index + 2] >> 4)
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



CSV_COLUMNS = [
    "sample_index", "t_perf_ns", "t_unix_s", "scheduled_perf_ns",
    "lateness_us", "missed_slots_total",
    "temp_c", "humidity", "dht_age_s", "dht_temp_age_s",
    "surface_temp_c", "surface_age_s", "surface_ok", "humidity_ok",
    "adxl_temp_c", "ax_g", "ay_g", "az_g",
]
DS18B20_DEVICES_DIR = Path("/sys/bus/w1/devices")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SENTRY hourly CSV + DS18B20/DHT22 logger")
    parser.add_argument("--condition", default="drift-check")
    parser.add_argument("--output-dir", default="data/check")
    parser.add_argument("--run-dir", help="대시보드가 생성한 빈 실행 폴더(내부용)")
    parser.add_argument("--bus", type=int, default=1)
    parser.add_argument("--address", choices=["auto", "0x1d", "0x53"], default="auto")
    parser.add_argument("--sample-rate", type=int, choices=[125, 250, 500], default=125)
    parser.add_argument("--range-g", type=int, choices=[2, 4, 8], default=8)
    parser.add_argument("--rotate-seconds", type=float, default=3600.0,
                        help="실행 시작 기준 CSV 분할 간격. 기본 1시간")
    parser.add_argument("--flush-seconds", type=float, default=1.0)
    parser.add_argument("--fsync-seconds", type=float, default=10.0,
                        help="OS에 파일 동기화 요청 간격. 정전 무손실 보장은 아님")
    parser.add_argument("--dht-interval", type=float, default=2.0)
    parser.add_argument("--surface-interval", type=float, default=2.0)
    parser.add_argument("--adxl-temp-interval", type=float, default=1.0)
    parser.add_argument("--env-stale-seconds", type=float, default=10.0,
                        help="마지막 성공한 환경 측정값이 이 시간보다 오래되면 빈 칸")
    parser.add_argument("--surface-device", help="여러 DS18B20 사용 시 28-... 장치 ID")
    parser.add_argument("--no-dht", action="store_true")
    parser.add_argument("--no-surface", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+", args.condition) or args.condition in {".", ".."}:
        parser.error("condition에는 문자, 숫자, _, -, .만 사용할 수 있습니다.")
    for name in ("rotate_seconds", "flush_seconds", "fsync_seconds",
                 "dht_interval", "surface_interval", "adxl_temp_interval", "env_stale_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name}은 유한한 양수여야 합니다.")
    if args.dht_interval < 2.0:
        parser.error("dht-interval은 2초 이상이어야 합니다.")
    if args.surface_interval < 1.0:
        parser.error("surface-interval은 1초 이상이어야 합니다.")
    return args


def create_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run = Path(args.run_dir).resolve()
        run.mkdir(parents=True, exist_ok=True)
        if any(run.iterdir()):
            raise FileExistsError(f"실행 폴더가 비어 있지 않습니다. 덮어쓰지 않습니다: {run}")
        return run
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(
        prefix=f"{args.condition}_{time.strftime('%Y%m%dT%H%M%S')}_", dir=output))


def sync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: dict) -> None:
    # 동일 파일시스템에서 원자적 교체: 대시보드가 반쪽짜리 JSON을 읽지 않게 한다.
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def find_ds18b20_device(device_id: str | None = None) -> Path:
    if device_id:
        if not re.fullmatch(r"28-[0-9a-fA-F]+", device_id):
            raise ValueError("surface-device에는 28-... 형식의 장치 ID를 지정하세요.")
        candidates = [DS18B20_DEVICES_DIR / device_id / "w1_slave"]
        candidates = [path for path in candidates if path.exists()]
    else:
        candidates = sorted(DS18B20_DEVICES_DIR.glob("28-*/w1_slave"))
    if not candidates:
        raise RuntimeError(
            "DS18B20 없음: /boot/firmware/config.txt의 "
            "dtoverlay=w1-gpio,gpiopin=17 설정과 재부팅, 28-* 장치를 확인하세요.")
    if len(candidates) > 1:
        raise RuntimeError("DS18B20이 여러 개입니다. --surface-device 28-... 로 지정하세요.")
    return candidates[0]


def read_ds18b20_c(path: Path) -> float:
    # logger4.py의 CRC 판독 방식. 블로킹 판독은 별도 프로세스에서만 수행.
    try:
        lines = path.read_text().strip().splitlines()
        if len(lines) < 2 or not lines[0].strip().endswith("YES"):
            return math.nan
        value = int(lines[1].partition("t=")[2].strip()) / 1000.0
    except (OSError, ValueError):
        return math.nan
    # logger4.py와 동일하게 파워온 기본값 85°C는 무효로 처리한다.
    return value if -55 <= value <= 125 and value != 85.0 else math.nan


def publish_sensor(output_queue: mp.Queue, item: tuple) -> None:
    try:
        output_queue.put_nowait(item)
    except queue.Full:
        # 최신 측정만 필요. Queue feeder와의 경합 시 다음 측정에서 다시 전달.
        try:
            output_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            output_queue.put_nowait(item)
        except queue.Full:
            pass


def dht_worker(output_queue: mp.Queue, stop_event: mp.Event, interval_s: float) -> None:
    import adafruit_dht
    import board
    dht = adafruit_dht.DHT22(board.D4)
    try:
        while not stop_event.is_set():
            temp, hum = math.nan, math.nan
            try:
                t, h = dht.temperature, dht.humidity
                if t is not None and math.isfinite(float(t)):
                    temp = float(t)
                if h is not None and math.isfinite(float(h)) and 0 <= float(h) <= 100:
                    hum = float(h)
            except (RuntimeError, OSError):
                pass
            publish_sensor(output_queue, (time.perf_counter_ns(), temp, hum))
            stop_event.wait(interval_s)
    finally:
        dht.exit()


def surface_worker(output_queue: mp.Queue, stop_event: mp.Event,
                   interval_s: float, device_path: str) -> None:
    while not stop_event.is_set():
        value = read_ds18b20_c(Path(device_path))
        publish_sensor(output_queue, (time.perf_counter_ns(), value))
        stop_event.wait(interval_s)


class SensorCache:
    def __init__(self) -> None:
        self.values: dict[str, tuple[int, float]] = {}

    def update(self, name: str, measured_ns: int, value: float) -> None:
        # 실패한 측정 시도가 아닌, 성공한 값에 대해서만 시각을 갱신한다.
        if math.isfinite(value):
            self.values[name] = (measured_ns, value)

    def read(self, name: str, now_ns: int, max_age_s: float) -> tuple[float, float, bool]:
        if name not in self.values:
            return math.nan, math.nan, False
        measured_ns, value = self.values[name]
        age = max(0.0, (now_ns - measured_ns) / 1e9)
        valid = age <= max_age_s
        return (value if valid else math.nan), age, valid


def format_optional(value: float, digits: int = 3) -> str:
    return "" if not math.isfinite(value) else f"{value:.{digits}f}"


class RotatingCSVWriter:
    """Bounded queue separates CSV flush/fsync/rotation from the sensor loop."""
    def __init__(self, run_dir: Path, metadata: dict, *, queue_size: int = 1000):
        self.run_dir = run_dir
        self.metadata = metadata
        self.manifest = run_dir / "current.json"
        self.pending: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.saved_rows = 0
        self.file = None
        self.writer = None
        self.current_path: Path | None = None
        self.previous_path: Path | None = None
        self.part = 0
        self.thread = threading.Thread(target=self._run, name="csv-writer", daemon=True)
        self.thread.start()
        if not self.ready.wait(30.0):
            self.stop_event.set()
            raise TimeoutError("CSV 저장 스레드 시작 시간 초과")
        self.check()

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"CSV 저장 실패: {self.error}") from self.error

    def put(self, row: list, sample_ns: int) -> None:
        self.check()
        try:
            self.pending.put_nowait((row, sample_ns))
        except queue.Full as exc:
            raise RuntimeError("디스크 기록이 밀려 저장 큐가 가득 찼습니다. 수집을 중단합니다.") from exc

    def publish(self, status: str) -> None:
        atomic_json(self.manifest, {
            **self.metadata,
            "current_csv": self.current_path.name if self.current_path else None,
            "previous_csv": self.previous_path.name if self.previous_path else None,
            "part": self.part, "status": status, "saved_rows": self.saved_rows,
            "manifest_updated_unix_s": time.time(),
        })

    def _sync(self) -> None:
        self.file.flush()
        os.fsync(self.file.fileno())

    def _open_part(self, part: int) -> None:
        if self.file is not None:
            self._sync()
            self.file.close()
        self.previous_path = self.current_path
        self.part = part
        self.current_path = self.run_dir / f"part_{part:06d}_bridge_log.csv"
        # x 모드: 기존 CSV는 어떤 경우에도 덮어쓰지 않는다.
        self.file = self.current_path.open("x", newline="", encoding="utf-8",
                                          buffering=1024 * 1024)
        self.writer = csv.writer(self.file)
        self.writer.writerow(CSV_COLUMNS)
        self._sync()
        atomic_json(self.current_path.with_name(f"part_{part:06d}_metadata.json"), {
            **self.metadata, "part": part,
            "segment_start_elapsed_s": (part - 1) * self.metadata["rotate_seconds"],
        })
        self.publish("running")
        print(f"[저장] 현재 CSV: {self.current_path}", flush=True)

    def _run(self) -> None:
        try:
            self._open_part(1)
            self.ready.set()
            next_flush = time.monotonic() + self.metadata["flush_seconds"]
            next_sync = time.monotonic() + self.metadata["fsync_seconds"]
            rotate_ns = int(self.metadata["rotate_seconds"] * 1e9)
            while not self.stop_event.is_set() or not self.pending.empty():
                try:
                    row, sample_ns = self.pending.get(timeout=0.1)
                except queue.Empty:
                    row = None
                if row is not None:
                    elapsed = max(0, sample_ns - self.metadata["anchor_perf_ns"])
                    part = 1 + elapsed // rotate_ns
                    if part != self.part:
                        self._open_part(part)
                    self.writer.writerow(row)
                    self.saved_rows += 1
                now = time.monotonic()
                if now >= next_flush:
                    self.file.flush()
                    next_flush = now + self.metadata["flush_seconds"]
                if now >= next_sync:
                    self._sync()
                    self.publish("running")
                    next_sync = now + self.metadata["fsync_seconds"]
            self._sync()
            self.publish("stopped")
        except BaseException as exc:
            self.error = exc
            print(f"[저장 오류] {exc}", flush=True)
        finally:
            self.ready.set()
            if self.file is not None:
                try:
                    self.file.close()
                except OSError as exc:
                    if self.error is None:
                        self.error = exc

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=15.0)
        if self.thread.is_alive():
            raise TimeoutError("CSV 마감이 지연되고 있습니다. 디스크 상태를 확인하세요.")
        self.check()


def main() -> None:
    args = parse_args()
    # 하드웨어 모듈은 실제 수집을 시작할 때만 필요하다.
    from smbus2 import SMBus
    run_dir = create_run_dir(args)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: stop.set())
    children = []
    queues = {}
    writer = None
    cache = SensorCache()
    total_samples = missed_slots = read_errors = 0
    clip_counts = {"ax": 0, "ay": 0, "az": 0}
    latest_adxl_temp = math.nan
    last_adxl_temp_ns = 0
    sample_interval_ns = int(1e9 / args.sample_rate)
    scale = RANGE_TO_SCALE_G_PER_LSB[args.range_g]
    clip_limit = 0.995 * RANGE_TO_FULL_SCALE_G[args.range_g]
    surface_path = None
    try:
        if not args.no_surface:
            surface_path = find_ds18b20_device(args.surface_device)
        context = mp.get_context("spawn")
        specs = []
        if not args.no_dht:
            specs.append(("dht", dht_worker, (args.dht_interval,)))
        if surface_path is not None:
            specs.append(("surface", surface_worker, (args.surface_interval, str(surface_path))))
        for name, target, extra in specs:
            sensor_queue = context.Queue(maxsize=4)
            sensor_stop = context.Event()
            process = context.Process(target=target, args=(sensor_queue, sensor_stop, *extra),
                                      daemon=True)
            process.start()
            queues[name] = sensor_queue
            children.append((process, sensor_stop, sensor_queue))

        with SMBus(args.bus) as bus:
            address = detect_address(bus, args.address)
            configure_adxl355(bus, address, args.range_g, args.sample_rate)
            anchor_ns, anchor_unix = time.perf_counter_ns(), time.time()
            metadata = {
                "schema_version": 3, "condition": args.condition,
                "run_id": run_dir.name, "sample_rate_hz": args.sample_rate,
                "range_g": args.range_g, "address_hex": f"0x{address:02X}",
                "filter_register_hex": f"0x{SAMPLE_RATE_TO_FILTER[args.sample_rate]:02X}",
                "anchor_perf_ns": anchor_ns, "start_unix_s": anchor_unix,
                "rotate_seconds": args.rotate_seconds,
                "flush_seconds": args.flush_seconds, "fsync_seconds": args.fsync_seconds,
                "env_stale_seconds": args.env_stale_seconds,
                "temperature_source": "DS18B20", "humidity_source": "DHT22",
                "surface_device": str(surface_path) if surface_path else None,
                "surface_enabled": not args.no_surface, "dht_enabled": not args.no_dht,
                "temp_c_meaning": "DHT22 air temperature (reference only)",
                "dht_age_s_meaning": "age since last successful humidity reading",
            }
            atomic_json(run_dir / "session_metadata.json", metadata)
            writer = RotatingCSVWriter(run_dir, metadata, queue_size=args.sample_rate * 5)
            print(f"[수집] {args.sample_rate} Hz / ±{args.range_g} g / {args.rotate_seconds:g}초 분할")
            print(f"[수집] 실행 폴더: {run_dir}\n[수집] Ctrl+C로 안전 종료", flush=True)
            next_sample_ns = time.perf_counter_ns()
            while not stop.is_set():
                writer.check()
                wait_until_ns(next_sample_ns)
                if stop.is_set():
                    break
                now_ns = time.perf_counter_ns()
                if now_ns - next_sample_ns >= sample_interval_ns:
                    skipped = (now_ns - next_sample_ns) // sample_interval_ns
                    missed_slots += skipped
                    next_sample_ns += skipped * sample_interval_ns
                scheduled_ns = next_sample_ns
                for name, sensor_queue in queues.items():
                    try:
                        while True:
                            item = sensor_queue.get_nowait()
                            if name == "dht":
                                measured_ns, temp, hum = item
                                cache.update("air", measured_ns, temp)
                                cache.update("humidity", measured_ns, hum)
                            else:
                                measured_ns, surface = item
                                cache.update("surface", measured_ns, surface)
                    except queue.Empty:
                        pass
                if now_ns - last_adxl_temp_ns >= int(args.adxl_temp_interval * 1e9):
                    try:
                        latest_adxl_temp = read_adxl_temperature_c(bus, address)
                    except OSError:
                        latest_adxl_temp = math.nan
                    last_adxl_temp_ns = now_ns
                read_start = time.perf_counter_ns()
                try:
                    ax, ay, az = read_acceleration_g(bus, address, scale)
                except OSError:
                    read_errors += 1
                    next_sample_ns += sample_interval_ns
                    continue
                sample_ns = (read_start + time.perf_counter_ns()) // 2
                for axis, value in (("ax", ax), ("ay", ay), ("az", az)):
                    if abs(value) >= clip_limit:
                        clip_counts[axis] += 1
                air, air_age, _ = cache.read("air", sample_ns, args.env_stale_seconds)
                hum, hum_age, hum_ok = cache.read("humidity", sample_ns, args.env_stale_seconds)
                surface, surface_age, surface_ok = cache.read("surface", sample_ns, args.env_stale_seconds)
                row = [
                    total_samples, sample_ns,
                    f"{unix_from_perf_ns(sample_ns, anchor_ns, anchor_unix):.9f}",
                    scheduled_ns, f"{(sample_ns - scheduled_ns) / 1000:.3f}", missed_slots,
                    format_optional(air), format_optional(hum), format_optional(hum_age),
                    format_optional(air_age), format_optional(surface), format_optional(surface_age),
                    int(surface_ok), int(hum_ok), format_optional(latest_adxl_temp),
                    f"{ax:.8f}", f"{ay:.8f}", f"{az:.8f}",
                ]
                writer.put(row, sample_ns)
                total_samples += 1
                next_sample_ns += sample_interval_ns
    finally:
        stop.set()
        for process, sensor_stop, sensor_queue in children:
            sensor_stop.set()
        try:
            if writer is not None:
                writer.close()
        finally:
            for process, sensor_stop, sensor_queue in children:
                process.join(timeout=3.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=2.0)
                sensor_queue.close()
            qc = {
                "queued_samples": total_samples,
                "saved_rows": writer.saved_rows if writer else 0,
                "missed_slots": missed_slots, "read_errors": read_errors,
                "clip_counts": clip_counts,
            }
            try:
                atomic_json(run_dir / "qc_summary.json", qc)
            except OSError as exc:
                print(f"[QC 저장 실패] {exc}", flush=True)
            print(f"[종료/QC] {qc}", flush=True)


if __name__ == "__main__":
    main()