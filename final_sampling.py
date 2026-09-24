#!/usr/bin/env python3
"""Generate and freeze 560 ML daylight design cases.

This script is intentionally independent from the existing CNN datasets.
It creates a small, reproducible pilot set for checking the workflow:

* 56 unique discrete design strata, with 10 cases in each stratum;
* Latin Hypercube Sampling (LHS) for each continuous variable within a stratum;
* a fixed seed, so every model ID always describes the same design condition;
* CSV and JSON outputs that can be read by a later geometry/simulation script.

The script uses Python's standard library only. It will not overwrite the
fixed outputs unless --overwrite is supplied deliberately.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable


# Reproducibility settings. Do not change these after simulations begin.
SAMPLES_PER_STRATUM = 10
RANDOM_SEED = 20260924

# Fixed study geometry and the Section 4.2 parameter bounds.
ROOM_WIDTH_M = 6.0
ROOM_DEPTH_M = 6.0
ROOM_HEIGHT_M = 3.0
WINDOW_WIDTH_RANGE_M = (0.60, 2.40)
CONVENTIONAL_WINDOW_HEIGHT_RANGE_M = (0.60, 2.10)
OVERHANG_DEPTH_RANGE_M = (0.30, 1.50)

# The fourteen unique façade patterns avoid duplicated geometry. For example,
# north + south and south + north describe the same opposite-façade condition.
FACADE_PATTERNS = (
    {"pattern_id": "single_north", "facade_configuration": "single", "reference_orientation": "north", "active_facades": ("north",)},
    {"pattern_id": "single_east", "facade_configuration": "single", "reference_orientation": "east", "active_facades": ("east",)},
    {"pattern_id": "single_south", "facade_configuration": "single", "reference_orientation": "south", "active_facades": ("south",)},
    {"pattern_id": "single_west", "facade_configuration": "single", "reference_orientation": "west", "active_facades": ("west",)},
    {"pattern_id": "adjacent_north_east", "facade_configuration": "two_adjacent", "reference_orientation": "north", "active_facades": ("north", "east")},
    {"pattern_id": "adjacent_east_south", "facade_configuration": "two_adjacent", "reference_orientation": "east", "active_facades": ("east", "south")},
    {"pattern_id": "adjacent_south_west", "facade_configuration": "two_adjacent", "reference_orientation": "south", "active_facades": ("south", "west")},
    {"pattern_id": "adjacent_west_north", "facade_configuration": "two_adjacent", "reference_orientation": "west", "active_facades": ("west", "north")},
    {"pattern_id": "opposite_north_south", "facade_configuration": "two_opposite", "reference_orientation": "north", "active_facades": ("north", "south")},
    {"pattern_id": "opposite_east_west", "facade_configuration": "two_opposite", "reference_orientation": "east", "active_facades": ("east", "west")},
    {"pattern_id": "three_north_east_west", "facade_configuration": "three_facades", "reference_orientation": "north", "active_facades": ("north", "east", "west")},
    {"pattern_id": "three_north_east_south", "facade_configuration": "three_facades", "reference_orientation": "east", "active_facades": ("north", "east", "south")},
    {"pattern_id": "three_east_south_west", "facade_configuration": "three_facades", "reference_orientation": "south", "active_facades": ("east", "south", "west")},
    {"pattern_id": "three_north_south_west", "facade_configuration": "three_facades", "reference_orientation": "west", "active_facades": ("north", "south", "west")},
)
WINDOW_TYPES = ("conventional", "floor_to_ceiling")
SHADING_STATES = (False, True)

N_STRATA = len(FACADE_PATTERNS) * len(WINDOW_TYPES) * len(SHADING_STATES)
N_SAMPLES = N_STRATA * SAMPLES_PER_STRATUM

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "fixed_design_cases_560.csv"
JSON_PATH = SCRIPT_DIR / "fixed_design_cases_560.json"


def lhs_uniform(low: float, high: float, count: int, rng: random.Random) -> list[float]:
    """Return `count` values from one-dimensional uniform Latin Hypercube Sampling."""
    if not low < high:
        raise ValueError("LHS bounds must satisfy low < high.")
    positions = [(index + rng.random()) / count for index in range(count)]
    rng.shuffle(positions)
    return [low + position * (high - low) for position in positions]


def round_m(value: float) -> float:
    """Keep saved geometry readable with centimetre-level precision."""
    return round(value, 2)


def make_cases() -> list[dict[str, Any]]:
    """Build 560 fixed cases: 10 LHS samples in each discrete design stratum."""
    rng = random.Random(RANDOM_SEED)
    strata: list[tuple[str, list[dict[str, Any]]]] = []

    for pattern in FACADE_PATTERNS:
        for window_type in WINDOW_TYPES:
            for has_shading in SHADING_STATES:
                stratum_id = "__".join(
                    (pattern["pattern_id"], window_type, "overhang" if has_shading else "no_overhang")
                )
                widths = lhs_uniform(*WINDOW_WIDTH_RANGE_M, SAMPLES_PER_STRATUM, rng)
                heights = (
                    lhs_uniform(*CONVENTIONAL_WINDOW_HEIGHT_RANGE_M, SAMPLES_PER_STRATUM, rng)
                    if window_type == "conventional"
                    else [ROOM_HEIGHT_M] * SAMPLES_PER_STRATUM
                )
                depths = (
                    lhs_uniform(*OVERHANG_DEPTH_RANGE_M, SAMPLES_PER_STRATUM, rng)
                    if has_shading
                    else [0.0] * SAMPLES_PER_STRATUM
                )
                stratum_cases: list[dict[str, Any]] = []

                for local_index in range(SAMPLES_PER_STRATUM):
                    width = round_m(widths[local_index])
                    height = round_m(heights[local_index])
                    sill_height = 0.0 if window_type == "floor_to_ceiling" else 0.90
                    z0 = sill_height
                    z1 = ROOM_HEIGHT_M if window_type == "floor_to_ceiling" else sill_height + height
                    x0 = (ROOM_WIDTH_M - width) / 2
                    x1 = x0 + width
                    facades = list(pattern["active_facades"])
                    overhang_depth = round_m(depths[local_index])
                    windows = [
                        {"face": face, "x0_m": round_m(x0), "x1_m": round_m(x1), "z0_m": round_m(z0), "z1_m": round_m(z1)}
                        for face in facades
                    ]
                    shading = [
                        {"type": "overhang", "face": face, "x0_m": round_m(x0), "x1_m": round_m(x1), "z_m": round_m(z1), "depth_m": overhang_depth}
                        for face in facades
                    ] if has_shading else []
                    stratum_cases.append(
                        {
                            "stratum_id": stratum_id,
                            "sample_within_stratum": local_index + 1,
                            "facade_configuration": pattern["facade_configuration"],
                            "reference_orientation": pattern["reference_orientation"],
                            "active_facades": facades,
                            "window_type": window_type,
                            "sill_height_m": round_m(sill_height),
                            "window_width_m": width,
                            "window_height_m": height,
                            "overhang_present": has_shading,
                            "overhang_depth_m": overhang_depth,
                            "overhang_length_m": width,
                            "windows": windows,
                            "shading": shading,
                        }
                    )
                strata.append((stratum_id, stratum_cases))

    # Interleave strata: every first group of 56 models contains one case from
    # every discrete condition, which makes partial visual checks more varied.
    cases: list[dict[str, Any]] = []
    for local_index in range(SAMPLES_PER_STRATUM):
        round_strata = strata.copy()
        rng.shuffle(round_strata)
        cases.extend(stratum_cases[local_index] for _, stratum_cases in round_strata)

    for index, case in enumerate(cases, start=1):
        case["model_id"] = f"model_{index:03d}"
        case["sample_index"] = index
    return cases


def write_outputs(cases: list[dict[str, Any]], overwrite: bool) -> None:
    outputs = (CSV_PATH, JSON_PATH)
    existing = [path.name for path in outputs if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(existing)
        raise FileExistsError(
            f"Fixed output already exists: {joined}. "
            "The cases are intentionally frozen. Use --overwrite only if you "
            "have not started simulations and deliberately want to regenerate them."
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        fields = [
            "model_id",
            "sample_index",
            "stratum_id",
            "sample_within_stratum",
            "facade_configuration",
            "reference_orientation",
            "active_facades",
            "window_type",
            "sill_height_m",
            "window_width_m",
            "window_height_m",
            "overhang_present",
            "overhang_depth_m",
            "overhang_length_m",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = {field: case[field] for field in fields if field != "active_facades"}
            row["active_facades"] = ";".join(case["active_facades"])
            writer.writerow(row)

    payload = {
        "metadata": {
            "purpose": "Fixed 560-case ML sampling set",
            "random_seed": RANDOM_SEED,
            "n_cases": N_SAMPLES,
            "n_unique_facade_patterns": len(FACADE_PATTERNS),
            "n_discrete_strata": N_STRATA,
            "samples_per_discrete_stratum": SAMPLES_PER_STRATUM,
            "sampling_method": (
                "Ten one-dimensional Latin Hypercube samples for the continuous "
                "variables within each discrete design stratum"
            ),
            "room_dimensions_m": {
                "width": ROOM_WIDTH_M,
                "depth": ROOM_DEPTH_M,
                "height": ROOM_HEIGHT_M,
            },
            "continuous_ranges_m": {
                "window_width": list(WINDOW_WIDTH_RANGE_M),
                "conventional_window_height": list(CONVENTIONAL_WINDOW_HEIGHT_RANGE_M),
                "overhang_depth_when_present": list(OVERHANG_DEPTH_RANGE_M),
            },
            "important_note": (
                "Every defined discrete design stratum is represented by ten cases."
            ),
        },
        "cases": cases,
    }
    with JSON_PATH.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)
        json_file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the frozen outputs. Use only before any simulations have started.",
    )
    args = parser.parse_args()
    cases = make_cases()
    write_outputs(cases, overwrite=args.overwrite)
    print(f"Created {len(cases)} fixed cases in {N_STRATA} strata with seed {RANDOM_SEED}.")
    print(f"CSV:  {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
