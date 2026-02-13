#!/usr/bin/env python3
"""
Auto-generate a unified bind JSON file for msrun/set_cpu_affinity.

Rules implemented:
- bind_cpu_mode=cpu, bind_memory_mode=numa
- actor_thread_fix_bind=True
- auto-detect device count and NPU-NUMA affinity
- per-device CPU selection uses affinity first, fallback to equal distribution
- module CPU ranges use relative positions within each device CPU list:
  runtime 4-8, minddata 9-12, main 13-19 (pynative optional if provided)
- scheduler CPU range uses relative positions 20-23
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

DEFAULT_RUNTIME = (4, 8)
DEFAULT_MINDDATA = (9, 12)
DEFAULT_MAIN = (13, 19)
DEFAULT_SCHEDULER = (20, 23)


class DeviceInfo:
    """Represent a device entry parsed from `npu-smi info -m`."""

    def __init__(self, info_line: str):
        self.npu_id = 0
        self.chip_id = 0
        self.chip_logic_id = 0
        self.chip_name = ""
        self._parse_info_line(info_line)

    def _parse_info_line(self, info_line: str) -> None:
        parts = info_line.strip().split(None, 3)
        if len(parts) < 3:
            raise ValueError(f"Invalid device info line: {info_line}")
        npu_id, chip_id, chip_logic_id = parts[:3]
        chip_name = parts[3] if len(parts) > 3 else ""
        self.npu_id = int(npu_id)
        self.chip_id = int(chip_id)
        self.chip_logic_id = int(chip_logic_id) if chip_logic_id.isnumeric() else chip_logic_id
        self.chip_name = chip_name


def _log(msg: str) -> None:
    """Print log message to standard error stream."""
    print(msg, file=sys.stderr)


def _execute_command(cmd_list: List[str], timeout: float = 1000.0) -> str:
    """Execute external command and return its stdout output."""
    cmd_str = " ".join(cmd_list)
    try:
        with subprocess.Popen(
            cmd_list,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
        ) as proc:
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                proc.kill()
                raise RuntimeError(f"Command '{cmd_str}' timed out after {timeout}s") from exc
            if proc.returncode != 0:
                raise RuntimeError(f"Command '{cmd_str}' failed (return code {proc.returncode}): {err.strip()}")
        return out
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command '{cmd_str}' not found") from exc


def _get_available_cpus() -> List[int]:
    """Get list of available CPUs from cgroup cpuset file."""
    cpu_set_path = "/sys/fs/cgroup/cpuset/cpuset.cpus"
    cpu_pattern = re.compile(r"(\d+)(?:-(\d+))?")
    try:
        with open(cpu_set_path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to read {cpu_set_path}: {exc}") from exc
    if not content:
        return []
    cpus: List[int] = []
    for segment in content.split(","):
        segment = segment.strip()
        if not segment:
            continue
        match = cpu_pattern.fullmatch(segment)
        if not match:
            raise RuntimeError(f"Invalid cpu segment '{segment}' in {cpu_set_path}")
        start, end = match.groups()
        start_val = int(start)
        end_val = int(end) if end else start_val
        if start_val > end_val:
            raise RuntimeError(f"Invalid cpu range '{segment}' in {cpu_set_path}")
        cpus.extend(range(start_val, end_val + 1))
    return sorted(set(cpus))


def _get_available_numas() -> List[int]:
    """Get list of available NUMA nodes from system directory."""
    numa_node_path = "/sys/devices/system/node/"
    if not (os.path.isdir(numa_node_path) and os.access(numa_node_path, os.R_OK)):
        return []
    numas: List[int] = []
    try:
        for dir_name in os.listdir(numa_node_path):
            if re.fullmatch(r"node\d+", dir_name):
                node_id = int(dir_name.replace("node", ""))
                numas.append(node_id)
    except OSError as exc:
        raise RuntimeError(f"Failed to read NUMA node directory {numa_node_path}: {exc}") from exc
    return sorted(numas)


def _get_device_map_info() -> Tuple[Dict[int, DeviceInfo], List[int]]:
    """Parse npu-smi output to get device mapping and available device IDs."""
    device_map_info: Dict[int, DeviceInfo] = {}
    device_ids: List[int] = []
    raw = _execute_command(["npu-smi", "info", "-m"]).strip().splitlines()
    for line in raw[1:]:
        if not line.strip():
            continue
        info = DeviceInfo(line)
        if isinstance(info.chip_logic_id, int):
            device_map_info[info.chip_logic_id] = info
            device_ids.append(info.chip_logic_id)
    device_ids = sorted(set(device_ids))
    if not device_ids:
        raise RuntimeError("No devices found from 'npu-smi info -m'.")
    return device_map_info, device_ids


def _get_pcie_info(
    device_map_info: Dict[int, DeviceInfo],
    device_ids: List[int],
    keyword: str = "PCIeBusInfo",
) -> Dict[int, str]:
    """Get PCIe bus information for each device using npu-smi."""
    device_to_pcie: Dict[int, str] = {}
    for device in device_ids:
        info = device_map_info.get(device)
        if info is None:
            raise RuntimeError("Failed to get device PCIe info (missing device map).")
        pcie_info = _execute_command(
            ["npu-smi", "info", "-t", "board", "-i", f"{info.npu_id}", "-c", f"{info.chip_id}"]
        ).strip().splitlines()
        for line in pcie_info:
            cleaned = "".join(line.split())
            if cleaned.startswith(keyword):
                device_to_pcie[device] = cleaned[len(keyword) + 1 :]
                break
    return device_to_pcie


def _get_numa_info(
    device_to_pcie: Dict[int, str], keyword: str = "NUMAnode"
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Map devices to NUMA nodes using lspci on PCIe information."""
    device_to_numa: Dict[int, int] = {}
    numa_to_device: Dict[int, List[int]] = {}
    for device, pcie_no in device_to_pcie.items():
        numa_info = _execute_command(["lspci", "-s", f"{pcie_no}", "-vvv"]).strip().splitlines()
        for line in numa_info:
            cleaned = "".join(line.split())
            if cleaned.startswith(keyword):
                numa_id = int(cleaned[len(keyword) + 1 :])
                device_to_numa[device] = numa_id
                numa_to_device.setdefault(numa_id, []).append(device)
                break
    return device_to_numa, numa_to_device


def _get_cpu_info(
    numa_ids: List[int],
    available_cpus: List[int],
    keyword1: str = "NUMAnode",
    keyword2: str = "CPU(s)",
) -> Dict[int, List[int]]:
    """Map NUMA nodes to associated CPUs using lscpu output."""
    numa_to_cpu: Dict[int, List[int]] = {}
    cpu_info = _execute_command(["lscpu"]).strip().splitlines()
    for line in cpu_info:
        cleaned = "".join(line.split())
        if cleaned.startswith(keyword1):
            pattern = re.escape(keyword1) + r"(\d+)" + re.escape(keyword2)
            match = re.search(pattern, cleaned)
            if not match:
                continue
            numa_id = int(match.group(1))
            split_info = cleaned.split(":")
            cpu_id_ranges = split_info[-1].split(",")
            ranges: List[int] = []
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    raise RuntimeError("Failed to parse 'lscpu' output.")
                start_val = int(endpoints[0])
                end_val = int(endpoints[1])
                ranges.extend([cid for cid in range(start_val, end_val + 1) if cid in available_cpus])
            if numa_id not in numa_ids:
                numa_id = -1
            numa_to_cpu.setdefault(numa_id, []).extend(ranges)
    return numa_to_cpu


def _equal_distribute_cpu_strategy(
    device_ids: List[int], available_cpus: List[int]
) -> Dict[int, List[int]]:
    """Distribute available CPUs equally across devices."""
    device_to_cpu: Dict[int, List[int]] = {}
    total_cpus = len(available_cpus)
    cpu_num_per_device = total_cpus // len(device_ids)
    if cpu_num_per_device < 1:
        return {}
    for idx, device_id in enumerate(device_ids):
        cpu_start = idx * cpu_num_per_device
        cpu_end = (idx + 1) * cpu_num_per_device if idx != len(device_ids) - 1 else total_cpus
        device_to_cpu[device_id] = available_cpus[cpu_start:cpu_end]
    return device_to_cpu


def _equal_distribute_numa_strategy(
    device_ids: List[int], available_numas: List[int]
) -> Dict[int, int]:
    """Distribute available NUMA nodes equally across devices."""
    if len(available_numas) < len(device_ids):
        return {}
    return {device_id: available_numas[idx] for idx, device_id in enumerate(device_ids)}


def _assemble_env_info(
    device_ids: List[int],
    available_cpus: List[int],
    affinity_flag: bool,
    numa_to_cpu_map: Dict[int, List[int]],
    device_to_numa_map: Dict[int, int],
) -> Dict[int, List[int]]:
    """Assemble CPU list for each device (affinity first, fallback to equal distribution)."""
    device_to_cpu_map: Dict[int, List[int]] = {device_id: [] for device_id in device_ids}
    cpu_num_per_device = len(available_cpus) // len(device_ids)
    if cpu_num_per_device < 1:
        return {}

    if affinity_flag:
        device_to_cpu_idx = {numa_id: 0 for numa_id in numa_to_cpu_map}
        for device_id in device_ids:
            numa_id = device_to_numa_map.get(device_id)
            if numa_id is None or numa_id not in numa_to_cpu_map:
                return {}
            affinity_cpu_start_idx = device_to_cpu_idx[numa_id]
            affinity_cpu = numa_to_cpu_map[numa_id][affinity_cpu_start_idx: affinity_cpu_start_idx + cpu_num_per_device]
            device_to_cpu_map[device_id].extend(affinity_cpu)
            device_to_cpu_idx[numa_id] = affinity_cpu_start_idx + len(affinity_cpu)

            if -1 in device_to_cpu_idx and len(affinity_cpu) < cpu_num_per_device:
                unaffinity_cpu_num = cpu_num_per_device - len(affinity_cpu)
                unaffinity_cpu_start_idx = device_to_cpu_idx[-1]
                unaffinity_cpu = numa_to_cpu_map[-1][
                    unaffinity_cpu_start_idx: unaffinity_cpu_start_idx + unaffinity_cpu_num
                ]
                device_to_cpu_map[device_id].extend(unaffinity_cpu)
                device_to_cpu_idx[-1] = unaffinity_cpu_start_idx + unaffinity_cpu_num
    else:
        for idx, device_id in enumerate(device_ids):
            cpu_start = idx * cpu_num_per_device
            device_to_cpu_map[device_id] = available_cpus[cpu_start: cpu_start + cpu_num_per_device]

    return device_to_cpu_map


def _int_list_to_range_str(num_list: List[int]) -> str:
    """Convert list of integers to compact range string (e.g. [1,2,3] -> "1-3", [1,3] -> "1,3")."""
    if not num_list:
        return ""
    num_list = sorted(num_list)
    ranges: List[Tuple[int, int]] = []
    start = num_list[0]
    for i in range(1, len(num_list)):
        if num_list[i] != num_list[i - 1] + 1:
            ranges.append((start, num_list[i - 1]))
            start = num_list[i]
    ranges.append((start, num_list[-1]))
    parts: List[str] = []
    for s_val, e_val in ranges:
        parts.append(str(s_val) if s_val == e_val else f"{s_val}-{e_val}")
    return ",".join(parts)


def _pick_relative(
    cpu_list: List[int], rel_range: Tuple[int, int], label: str, owner: str
) -> List[int]:
    """Pick sublist from CPU list using relative start/end indices."""
    start, end = rel_range
    if end >= len(cpu_list):
        raise ValueError(
            f"{owner}: cpu list length {len(cpu_list)} is less than required index {end} for {label}"
        )
    return cpu_list[start : end + 1]


def _select_scheduler_base(
    available_cpus: List[int],
    device_to_cpu_map: Dict[int, List[int]],
    base_mode: str,
    scheduler_range: Tuple[int, int],
    device_ids: List[int],
) -> List[int]:
    """Select base CPU list for scheduler (free/global/device0 strategy)."""
    if base_mode == "device0":
        if not device_ids:
            raise RuntimeError("No device ids available for scheduler-base=device0.")
        base_list = device_to_cpu_map.get(device_ids[0], [])
        if not base_list:
            raise RuntimeError("device0 CPU list is empty for scheduler-base=device0.")
        return base_list
    if base_mode == "global":
        return available_cpus

    used: set[int] = set()
    for cpu_list in device_to_cpu_map.values():
        used.update(cpu_list)
    free = [cpu for cpu in available_cpus if cpu not in used]
    if len(free) >= (scheduler_range[1] + 1):
        _log("Scheduler CPU base: using free CPU pool.")
        return free
    _log("Scheduler CPU base: free CPU pool insufficient, using global available CPUs.")
    return available_cpus


def _build_cpu_to_numa(numa_to_cpu_map: Dict[int, List[int]]) -> Dict[int, int]:
    """Build reverse map from CPU ID to NUMA node ID."""
    cpu_to_numa: Dict[int, int] = {}
    for numa_id, cpu_list in numa_to_cpu_map.items():
        for cpu in cpu_list:
            cpu_to_numa.setdefault(cpu, numa_id)
    return cpu_to_numa


def _infer_single_numa(cpu_list: List[int], cpu_to_numa: Dict[int, int]) -> Optional[int]:
    """Infer single NUMA node for a CPU list (if all CPUs belong to one NUMA node)."""
    numas = {cpu_to_numa[cpu] for cpu in cpu_list if cpu in cpu_to_numa and cpu_to_numa[cpu] != -1}
    if len(numas) == 1:
        return next(iter(numas))
    return None


def _parse_device_ids(args: argparse.Namespace) -> Tuple[Optional[Dict[int, DeviceInfo]], List[int]]:
    """Parse device IDs from command args/env or auto-detect via npu-smi."""
    if args.device_ids:
        ids = [int(x) for x in args.device_ids.split(",") if x.strip()]
        ids = sorted(set(ids))
        if not ids:
            raise ValueError("--device-ids provided but empty.")
        return None, ids
    if args.device_count is not None:
        if args.device_count < 1:
            raise ValueError("--device-count must be >= 1")
        return None, list(range(args.device_count))

    try:
        device_map_info, device_ids = _get_device_map_info()
        return device_map_info, device_ids
    except Exception as exc:
        env_visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").strip()
        if env_visible:
            ids = [int(x) for x in env_visible.split(",") if x.strip()]
            ids = sorted(set(ids))
            if ids:
                _log("npu-smi unavailable; using ASCEND_RT_VISIBLE_DEVICES for device ids.")
                return None, ids
        raise RuntimeError("Failed to detect devices; use --device-ids or --device-count.") from exc


def _generate_device_to_numa_map(
    device_ids: List[int],
    available_numas: List[int],
    device_map_info: Optional[Dict[int, DeviceInfo]],
) -> Tuple[Dict[int, int], Dict[int, List[int]], bool]:
    """Generate device-to-NUMA map (affinity first, fallback to equal distribution)."""
    device_to_numa: Dict[int, int] = {}
    numa_to_cpu_map: Dict[int, List[int]] = {}
    affinity_ok = False

    if device_map_info is None:
        return device_to_numa, numa_to_cpu_map, False

    try:
        device_to_pcie = _get_pcie_info(device_map_info, device_ids)
        temp_device_to_numa, _ = _get_numa_info(device_to_pcie)
        if not temp_device_to_numa:
            return device_to_numa, numa_to_cpu_map, False
        affinity_numas = set(temp_device_to_numa.values())
        non_affinity_pool = [numa for numa in available_numas if numa not in affinity_numas]
        pool_iter = iter(non_affinity_pool)
        used_numas: set[int] = set()
        for device_id in device_ids:
            numa = temp_device_to_numa.get(device_id)
            if numa is None:
                numa = next(pool_iter, None)
            if numa in used_numas:
                fallback = next(pool_iter, None)
                if fallback is not None:
                    numa = fallback
            if numa is None:
                return device_to_numa, numa_to_cpu_map, False
            used_numas.add(numa)
            device_to_numa[device_id] = numa
        affinity_ok = True
    except Exception as exc:
        _log(f"Affinity detection failed, fallback to equal distribution. Reason: {exc}")
        affinity_ok = False

    return device_to_numa, numa_to_cpu_map, affinity_ok


def _build_config(
    device_ids: List[int],
    available_cpus: List[int],
    available_numas: List[int],
    device_map_info: Optional[Dict[int, DeviceInfo]],
    runtime_range: Tuple[int, int],
    minddata_range: Tuple[int, int],
    main_range: Tuple[int, int],
    scheduler_range: Tuple[int, int],
    pynative_range: Optional[Tuple[int, int]],
    scheduler_base: str,
) -> Dict[str, object]:
    """Main logic to build full bind config JSON structure."""
    device_to_numa_map, numa_to_cpu_map, affinity_ok = _generate_device_to_numa_map(
        device_ids, available_numas, device_map_info
    )

    if affinity_ok:
        numa_ids = sorted(set(device_to_numa_map.values()))
        numa_to_cpu_map = _get_cpu_info(numa_ids, available_cpus)
        if not numa_to_cpu_map:
            affinity_ok = False
            device_to_numa_map = {}

    if not affinity_ok:
        device_to_numa_map = _equal_distribute_numa_strategy(device_ids, available_numas)
        if not device_to_numa_map:
            raise RuntimeError("Failed to assign NUMA nodes equally; not enough NUMA nodes.")
        numa_to_cpu_map = _get_cpu_info(available_numas, available_cpus)

    device_to_cpu_map = _assemble_env_info(
        device_ids,
        available_cpus,
        affinity_ok,
        numa_to_cpu_map,
        device_to_numa_map,
    )
    if not device_to_cpu_map:
        device_to_cpu_map = _equal_distribute_cpu_strategy(device_ids, available_cpus)
    if not device_to_cpu_map:
        raise RuntimeError("Failed to assign CPUs to devices; check available CPU resources.")

    bind_cpu: Dict[str, Dict[str, str]] = {}
    bind_memory: Dict[str, int] = {}

    for device_id in device_ids:
        cpu_list = device_to_cpu_map.get(device_id, [])
        if not cpu_list:
            raise RuntimeError(f"Empty CPU list for device{device_id}.")
        runtime_cpus = _pick_relative(cpu_list, runtime_range, "runtime", f"device{device_id}")
        minddata_cpus = _pick_relative(cpu_list, minddata_range, "minddata", f"device{device_id}")
        main_cpus = _pick_relative(cpu_list, main_range, "main", f"device{device_id}")
        device_entry = {
            "runtime": _int_list_to_range_str(runtime_cpus),
            "minddata": _int_list_to_range_str(minddata_cpus),
            "main": _int_list_to_range_str(main_cpus),
        }
        if pynative_range is not None:
            pynative_cpus = _pick_relative(cpu_list, pynative_range, "pynative", f"device{device_id}")
            device_entry["pynative"] = _int_list_to_range_str(pynative_cpus)
        bind_cpu[f"device{device_id}"] = device_entry
        bind_memory[f"device{device_id}"] = device_to_numa_map[device_id]

    scheduler_cpu_base = _select_scheduler_base(
        available_cpus, device_to_cpu_map, scheduler_base, scheduler_range, device_ids
    )
    scheduler_cpus = _pick_relative(scheduler_cpu_base, scheduler_range, "main", "scheduler")
    bind_cpu["scheduler"] = {"main": _int_list_to_range_str(scheduler_cpus)}

    # Scheduler does not bind memory.

    return {
        "bind_config": {
            "bind_cpu_mode": "cpu",
            "bind_memory_mode": "numa",
            "actor_thread_fix_bind": True,
        },
        "bind_cpu": bind_cpu,
        "bind_memory": bind_memory,
    }


def _parse_range(value: str, name: str) -> Tuple[int, int]:
    """Parse string range (e.g. "4-8") to integer tuple (start, end)."""
    match = re.fullmatch(r"(\d+)-(\d+)", value.strip())
    if not match:
        raise ValueError(f"{name} must be in 'start-end' format, got '{value}'.")
    start_val = int(match.group(1))
    end_val = int(match.group(2))
    if start_val > end_val:
        raise ValueError(f"{name} start must be <= end, got '{value}'.")
    return start_val, end_val


def _parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(description="Generate unified bind JSON file.")
    parser.add_argument("-o", "--output", default="bind_config.json", help="Output JSON file path.")
    parser.add_argument("--device-ids", default="", help="Comma-separated device ids to use.")
    parser.add_argument("--device-count", type=int, default=None, help="Number of devices if auto-detect fails.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation spaces.")
    parser.add_argument("--runtime-range", default=f"{DEFAULT_RUNTIME[0]}-{DEFAULT_RUNTIME[1]}",
                        help="Relative CPU range for runtime threads, e.g. 4-8.")
    parser.add_argument("--minddata-range", default=f"{DEFAULT_MINDDATA[0]}-{DEFAULT_MINDDATA[1]}",
                        help="Relative CPU range for minddata threads, e.g. 9-12.")
    parser.add_argument("--main-range", default=f"{DEFAULT_MAIN[0]}-{DEFAULT_MAIN[1]}",
                        help="Relative CPU range for main thread, e.g. 13-19.")
    parser.add_argument("--scheduler-range", default=f"{DEFAULT_SCHEDULER[0]}-{DEFAULT_SCHEDULER[1]}",
                        help="Relative CPU range for scheduler main, e.g. 20-23.")
    parser.add_argument("--pynative-range", default="",
                        help="Relative CPU range for pynative threads (optional), e.g. 10-14.")
    parser.add_argument(
        "--scheduler-base",
        choices=["free", "global", "device0"],
        default="free",
        help=(
            "CPU base list used by --scheduler-range. "
            "'free': use CPUs not assigned to any device; if insufficient, fallback to global list. "
            "'global': use the full available CPU list. "
            "'device0': use device0's CPU segment as the base."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point - orchestrate config generation and write to file."""
    args = _parse_args()
    device_map_info, device_ids = _parse_device_ids(args)

    available_cpus = _get_available_cpus()
    if not available_cpus:
        raise RuntimeError("No available CPUs detected from cpuset.")

    available_numas = _get_available_numas()
    if not available_numas:
        raise RuntimeError("No available NUMA nodes detected.")

    runtime_range = _parse_range(args.runtime_range, "runtime-range")
    minddata_range = _parse_range(args.minddata_range, "minddata-range")
    main_range = _parse_range(args.main_range, "main-range")
    scheduler_range = _parse_range(args.scheduler_range, "scheduler-range")
    pynative_range = _parse_range(args.pynative_range, "pynative-range") if args.pynative_range else None

    config = _build_config(
        device_ids,
        available_cpus,
        available_numas,
        device_map_info,
        runtime_range,
        minddata_range,
        main_range,
        scheduler_range,
        pynative_range,
        args.scheduler_base,
    )

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=args.indent, ensure_ascii=False)
    _log(f"Bind JSON generated: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        _log(f"ERROR: {exc}")
        raise SystemExit(1) from exc
