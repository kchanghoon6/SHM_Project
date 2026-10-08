from __future__ import annotations

import argparse
import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

import adafruit_dht
import board
from smbus2 import SMBus


ADXL355_ADDR_LOW = 0x1D
ADXL355_ADDR_HIGH = 0x53

REG_DEVID_AD = 0x00
REG_TEMP2 = 0x06
REG_XDATA3 = 0x08
REG_FILTER = 0x28
REG_RANGE = 0x2C
REG_POWER_CTL = 0x2D
REG_RESET = 0x2F

EXPECTED_ID = (0xAD, 0x1D, 0xED)

RANGE_TO_CODE = {2: 0x01, 4: 0x02, 8: 0x03}
RANGE_TO_SCALE_G_PER_LSB = {
    2: 2.048 / (2**19),
    4: 4.096 / (2**19),
    8: 8.192 / (2**19),
}

# ADXL355 nominal temperature conversion: 1852 LSB at 25 C, -9.05 LSB/C.
TEMP_BIAS_LSB = 1852.0
TEMP_SLOPE_LSB_PER_C = -9.05


def scan_i2c_addresses(bus: SMBus) -> list[int]:
    found = []
    for address in range(0x03, 0x78):
        try:
            bus.read_byte(address)
        except OSError:
            continue
        found.append(address)
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="../python/data/real_bridge_log.csv", help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append to the output CSV instead of overwriting it")
    parser.add_argument("--bus", type=int, default=1, help="I2C bus number")
    parser.add_argument("--address", choices=["auto", "0x1d", "0x53"], default="auto", help="ADXL355 I2C address")
    parser.add_argument("--sample-rate", type=float, default=200.0, help="Acceleration sampling rate in Hz")
    parser.add_argument("--range-g", type=int, choices=[2, 4, 8], default=2, help="Acceleration range")
    parser.add_argument("--dht-interval", type=float, default=2.0, help="DHT22 read interval in seconds")
    parser.add_argument("--adxl-temp-interval", type=float, default=1.0, help="ADXL355 temperature read interval in seconds")
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=60.0,
        help="Force CSV data to disk every N seconds. Use 0 to sync every row.",
    )
    return parser.parse_args()


def twos_complement(value: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return value - (1 << bits) if value & sign_bit else value


def read_block(bus: SMBus, address: int, register: int, length: int) -> list[int]:
    return bus.read_i2c_block_data(address, register, length)


def write_register(bus: SMBus, address: int, register: int, value: int) -> None:
    bus.write_byte_data(address, register, value)


def detect_address(bus: SMBus, requested: str) -> int:
    candidates = [ADXL355_ADDR_LOW, ADXL355_ADDR_HIGH] if requested == "auto" else [int(requested, 16)]
    for address in candidates:
        try:
            device_id = tuple(read_block(bus, address, REG_DEVID_AD, 3))
        except OSError:
            continue
        if device_id == EXPECTED_ID:
            return address

    found = scan_i2c_addresses(bus)
    found_text = ", ".join(f"0x{addr:02X}" for addr in found) if found else "none"
    raise RuntimeError(
        "ADXL355 not found.\n"
        f"I2C addresses currently visible: {found_text}\n"
        "Run this command and check whether 1d or 53 appears:\n"
        "  i2cdetect -y 1\n"
        "If nothing appears, check VDD, VDDIO, GND, SDA, SCL, and SCLK/VSSIO wiring.\n"
        "If 0x1D appears, run with --address 0x1d. If 0x53 appears, run with --address 0x53."
    )


def configure_adxl355(bus: SMBus, address: int, range_g: int) -> None:
    write_register(bus, address, REG_RESET, 0x52)
    time.sleep(0.1)
    write_register(bus, address, REG_RANGE, 0x80 | RANGE_TO_CODE[range_g])
    write_register(bus, address, REG_FILTER, 0x00)
    write_register(bus, address, REG_POWER_CTL, 0x00)
    time.sleep(0.1)


def read_adxl_temperature_c(bus: SMBus, address: int) -> float:
    temp2, temp1 = read_block(bus, address, REG_TEMP2, 2)
    raw = ((temp2 & 0x0F) << 8) | temp1
    return 25.0 + (raw - TEMP_BIAS_LSB) / TEMP_SLOPE_LSB_PER_C


def read_acceleration_g(bus: SMBus, address: int, scale_g_per_lsb: float) -> tuple[float, float, float]:
    data = read_block(bus, address, REG_XDATA3, 9)
    values = []
    for i in range(0, 9, 3):
        raw20 = (data[i] << 12) | (data[i + 1] << 4) | (data[i + 2] >> 4)
        values.append(twos_complement(raw20, 20) * scale_g_per_lsb)
    return values[0], values[1], values[2]


def read_dht22_safely(dht: adafruit_dht.DHT22) -> tuple[float, float]:
    try:
        temp_c = dht.temperature
        humidity = dht.humidity
    except RuntimeError:
        return math.nan, math.nan

    return (
        math.nan if temp_c is None else float(temp_c),
        math.nan if humidity is None else float(humidity),
    )


def format_optional(value: float, digits: int = 2) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def resolve_output_path(output: str) -> Path:
    return Path(output).expanduser().resolve()


def open_output_csv(output_path: Path, append: bool):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        return output_path.open(mode, newline="", encoding="utf-8")
    except PermissionError as exc:
        message = (
            f"Cannot write CSV file: {output_path}\n"
            "Try one of these fixes:\n"
            "  1) Close Excel or any program that may have opened the CSV file.\n"
            "  2) Save to your home folder instead:\n"
            "     python3 raspi_3a_eval_adxl355z_dht22_logger.py --output ~/bridge_data/real_bridge_log.csv\n"
            "  3) If the file was created with sudo before, run:\n"
            f"     sudo chown $USER:$USER {output_path}\n"
            "  4) If the folder was created with sudo before, run:\n"
            f"     sudo chown -R $USER:$USER {output_path.parent}\n"
        )
        raise PermissionError(message) from exc


def csv_header() -> list[str]:
    return ["timestamp_iso", "unix_time_s", "ms", "temp_c", "humidity", "adxl_temp_c", "ax_g", "ay_g", "az_g"]


def sync_output_file(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def main() -> None:
    args = parse_args()
    output_path = resolve_output_path(args.output)

    sample_interval = 1.0 / args.sample_rate
    scale = RANGE_TO_SCALE_G_PER_LSB[args.range_g]

    dht = adafruit_dht.DHT22(board.D4)
    latest_dht_temp_c = math.nan
    latest_humidity = math.nan
    latest_adxl_temp_c = math.nan
    last_dht_time = 0.0
    last_adxl_temp_time = 0.0

    try:
        with SMBus(args.bus) as bus:
            address = detect_address(bus, args.address)
            configure_adxl355(bus, address, args.range_g)

            print("Starting data collection. Press Ctrl+C to stop.")
            print(f"ADXL355 I2C address: 0x{address:02X}")
            print(f"Acceleration range: +/-{args.range_g} g")
            print(f"Acceleration sample rate: {args.sample_rate:.1f} Hz")
            print(f"DHT22 interval: {args.dht_interval:.1f} s")
            print(f"ADXL355 temperature interval: {args.adxl_temp_interval:.1f} s")
            print(f"Disk sync interval: {args.sync_interval:.1f} s")
            print(f"Output CSV: {output_path}")

            start_time = time.monotonic()
            next_sample_time = start_time
            last_sync_time = start_time

            with open_output_csv(output_path, args.append) as f:
                writer = csv.writer(f)
                if not args.append or output_path.stat().st_size == 0:
                    writer.writerow(csv_header())
                    sync_output_file(f)

                while True:
                    now = time.monotonic()

                    if now - last_dht_time >= args.dht_interval:
                        last_dht_time = now
                        dht_temp_c, humidity = read_dht22_safely(dht)
                        if not math.isnan(dht_temp_c):
                            latest_dht_temp_c = dht_temp_c
                        if not math.isnan(humidity):
                            latest_humidity = humidity

                    if now - last_adxl_temp_time >= args.adxl_temp_interval:
                        last_adxl_temp_time = now
                        latest_adxl_temp_c = read_adxl_temperature_c(bus, address)

                    if now >= next_sample_time:
                        sample_time = datetime.now().astimezone()
                        unix_time_s = time.time()
                        elapsed_ms = int((now - start_time) * 1000)
                        ax_g, ay_g, az_g = read_acceleration_g(bus, address, scale)

                        # Use ADXL355 internal temperature only until the first valid DHT22 value arrives.
                        temp_for_analysis = latest_dht_temp_c
                        if math.isnan(temp_for_analysis):
                            temp_for_analysis = latest_adxl_temp_c

                        writer.writerow(
                            [
                                sample_time.isoformat(timespec="milliseconds"),
                                f"{unix_time_s:.3f}",
                                elapsed_ms,
                                format_optional(temp_for_analysis),
                                format_optional(latest_humidity),
                                format_optional(latest_adxl_temp_c),
                                f"{ax_g:.8f}",
                                f"{ay_g:.8f}",
                                f"{az_g:.8f}",
                            ]
                        )
                        f.flush()
                        if args.sync_interval <= 0 or now - last_sync_time >= args.sync_interval:
                            sync_output_file(f)
                            last_sync_time = now
                        next_sample_time += sample_interval
                    else:
                        time.sleep(min(0.001, next_sample_time - now))

    except KeyboardInterrupt:
        print()
        print("Data collection stopped.")
    finally:
        dht.exit()


if __name__ == "__main__":
    main()
