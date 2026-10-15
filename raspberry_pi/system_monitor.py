"""Raspberry Pi CPU / temperature monitoring from /proc and /sys (Linux only, no extra packages).

Used by the streaming script to report how much Pi CPU the visualization path
(this Python process + the ffmpeg encoder) costs. Parsing functions are pure so
they can be unit-tested on Windows.
"""

import os
import time

CLOCK_TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
THERMAL_FILE = "/sys/class/thermal/thermal_zone0/temp"


def parse_total_cpu(proc_stat_text: str) -> tuple[int, int]:
    """(busy, total) jiffies from the aggregate 'cpu' line of /proc/stat."""
    fields = proc_stat_text.splitlines()[0].split()
    if fields[0] != "cpu":
        raise ValueError("first /proc/stat line is not the aggregate 'cpu' line")
    # user nice system idle iowait irq softirq steal (guest time is already counted in user)
    values = [int(v) for v in fields[1:9]]
    idle = values[3] + values[4]
    total = sum(values)
    return total - idle, total


def parse_process_ticks(proc_pid_stat_text: str) -> int:
    """utime + stime jiffies from /proc/<pid>/stat. The command name may contain spaces, so split after ')'."""
    rest = proc_pid_stat_text[proc_pid_stat_text.rindex(")") + 2:].split()
    return int(rest[11]) + int(rest[12])  # fields 14 and 15 of the full line


def percent(delta_busy: float, delta_total: float) -> float:
    return 100.0 * delta_busy / delta_total if delta_total > 0 else 0.0


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


class CpuMonitor:
    """Each `sample()` reports usage since the previous call."""

    def __init__(self):
        self._last_total = self._total()
        self._last_wall = time.monotonic()
        self._last_ticks: dict[int, int] = {}

    @staticmethod
    def _total() -> tuple[int, int] | None:
        text = _read("/proc/stat")
        return parse_total_cpu(text) if text else None

    @staticmethod
    def _ticks(pid: int) -> int | None:
        text = _read(f"/proc/{pid}/stat")
        return parse_process_ticks(text) if text else None

    def sample(self, pids: dict[str, int | None]) -> dict[str, float | None]:
        """{'total': % of all cores, <name>: % of one core per pid (can exceed 100), 'temp_c': SoC temperature}."""
        now = time.monotonic()
        wall = now - self._last_wall
        result: dict[str, float | None] = {}

        total = self._total()
        if total and self._last_total:
            result["total"] = percent(total[0] - self._last_total[0], total[1] - self._last_total[1])
        else:
            result["total"] = None
        self._last_total = total

        for name, pid in pids.items():
            ticks = self._ticks(pid) if pid else None
            previous = self._last_ticks.get(pid) if pid else None
            result[name] = percent((ticks - previous) / CLOCK_TICKS, wall) if ticks is not None and previous is not None else None
            if ticks is not None:
                self._last_ticks[pid] = ticks

        temp = _read(THERMAL_FILE)
        result["temp_c"] = int(temp) / 1000 if temp else None
        self._last_wall = now
        return result


def format_sample(sample: dict[str, float | None]) -> str:
    parts = []
    for name, value in sample.items():
        if name == "temp_c":
            parts.append(f"temp={value:.1f}C" if value is not None else "temp=-")
        else:
            parts.append(f"cpu_{name}={value:.0f}%" if value is not None else f"cpu_{name}=-")
    return " ".join(parts)
