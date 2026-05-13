from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float)
    window = max(1, int(window))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values.astype(float), kernel, mode="same")


def _fill_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = mask.astype(bool).copy()
    runs = _runs_from_mask(~mask)
    for start, end in runs:
        if start == 0 or end == len(mask):
            continue
        if end - start <= max_gap:
            mask[start:end] = True
    return mask


def _runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _best_split_point(run: tuple[int, int], profile: np.ndarray) -> int:
    start, end = run
    width = end - start
    if width <= 2:
        return start + max(1, width // 2)
    inset = max(1, int(width * 0.20))
    left = start + inset
    right = end - inset
    if left >= right:
        return start + width // 2
    local = profile[left:right]
    if local.size == 0:
        return start + width // 2
    return left + int(np.argmin(local))


def _split_widest_until_count(
    runs: list[tuple[int, int]],
    profile: np.ndarray,
    target_count: int,
    min_width: int,
) -> list[tuple[int, int]]:
    runs = sorted(runs)
    while len(runs) < target_count and runs:
        candidates = [(end - start, index, (start, end)) for index, (start, end) in enumerate(runs)]
        candidates.sort(reverse=True)
        _width, index, run = candidates[0]
        start, end = run
        split = _best_split_point(run, profile)
        if split - start < min_width or end - split < min_width:
            split = start + (end - start) // 2
        if split <= start or split >= end:
            break
        runs[index:index + 1] = [(start, split), (split, end)]
        runs = sorted(runs)
    return runs


def _merge_until_count(runs: list[tuple[int, int]], target_count: int) -> list[tuple[int, int]]:
    runs = sorted(runs)
    while len(runs) > target_count:
        best_index = 0
        best_gap = None
        for index in range(len(runs) - 1):
            gap = runs[index + 1][0] - runs[index][1]
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_index = index
        merged = (runs[best_index][0], runs[best_index + 1][1])
        runs[best_index:best_index + 2] = [merged]
    return runs


def _normalize_runs_to_nine(
    runs: list[tuple[int, int]],
    profile: np.ndarray,
    width: int,
) -> tuple[list[tuple[int, int]], str]:
    notes: list[str] = []
    min_width = max(8, int(width * 0.025))
    runs = [(max(0, int(start)), min(width, int(end))) for start, end in runs if end - start >= min_width]
    runs = sorted(runs)

    if not runs:
        notes.append("projection found no stable wood bands; split the full image width into 9 approximate bands")
        return _equal_runs(0, width, 9), "; ".join(notes)

    if len(runs) < 9:
        notes.append(f"projection found {len(runs)} bands; split wider bands until 9 hints")
        runs = _split_widest_until_count(runs, profile, 9, min_width)
    if len(runs) > 9:
        notes.append(f"projection found {len(runs)} bands; merged closest bands down to 9 hints")
        runs = _merge_until_count(runs, 9)

    if len(runs) < 9:
        left = min(start for start, _end in runs)
        right = max(end for _start, end in runs)
        notes.append("band splitting could not reach 9; used equal fallback over detected pallet span")
        runs = _equal_runs(left, right, 9)

    return sorted(runs), "; ".join(notes)


def _equal_runs(left: int, right: int, count: int) -> list[tuple[int, int]]:
    if right <= left:
        right = left + count
    edges = np.linspace(left, right, count + 1)
    runs = []
    for index in range(count):
        start = int(round(edges[index]))
        end = int(round(edges[index + 1]))
        if end <= start:
            end = start + 1
        runs.append((start, end))
    return runs


def compute_board_x_range_hints(image_path: Path) -> dict[str, Any]:
    """Return approximate vertical board x-ranges for prompting and debugging."""
    path = Path(image_path)
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            width, height = rgb.size
            arr = np.asarray(rgb).astype(np.float32)
    except Exception as exc:
        return {
            "image_width": 0,
            "image_height": 0,
            "candidate_x_ranges_pixels": [],
            "candidate_x_ranges_normalized": [],
            "notes": f"Board hints unavailable: {exc}",
        }

    row_profiles: list[np.ndarray] = []
    row_runs: list[tuple[int, int]] = []
    fractions = (0.25, 0.40, 0.60, 0.75)
    half_window = max(4, int(height * 0.0125))
    smooth_window = max(5, int(width * 0.012))
    max_gap = max(3, int(width * 0.010))
    min_width = max(8, int(width * 0.025))

    for fraction in fractions:
        y = int(round(height * fraction))
        y0 = max(0, y - half_window)
        y1 = min(height, y + half_window + 1)
        crop = arr[y0:y1, :, :]
        if crop.size == 0:
            continue
        mean_rgb = crop.mean(axis=0)
        gray = 0.299 * mean_rgb[:, 0] + 0.587 * mean_rgb[:, 1] + 0.114 * mean_rgb[:, 2]
        profile = _smooth(gray / 255.0, smooth_window)
        threshold = max(0.16, float(np.percentile(profile, 38)))
        mask = profile >= threshold
        mask = _fill_small_gaps(mask, max_gap)
        runs = [(start, end) for start, end in _runs_from_mask(mask) if end - start >= min_width]
        row_profiles.append(profile)
        row_runs.extend(runs)

    if row_profiles:
        combined_profile = _smooth(np.mean(np.vstack(row_profiles), axis=0), smooth_window)
    else:
        combined_profile = np.ones(width, dtype=float)

    if row_runs:
        coverage = np.zeros(width, dtype=float)
        for start, end in row_runs:
            coverage[max(0, start):min(width, end)] += 1.0
        coverage = _smooth(coverage / max(1.0, float(len(fractions))), smooth_window)
        mask = coverage >= 0.35
        mask = _fill_small_gaps(mask, max_gap)
        runs = [(start, end) for start, end in _runs_from_mask(mask) if end - start >= min_width]
    else:
        threshold = max(0.16, float(np.percentile(combined_profile, 38)))
        mask = _fill_small_gaps(combined_profile >= threshold, max_gap)
        runs = [(start, end) for start, end in _runs_from_mask(mask) if end - start >= min_width]

    runs, normalization_note = _normalize_runs_to_nine(runs, combined_profile, width)
    ranges_pixels = [[int(start), int(end)] for start, end in runs]
    ranges_normalized = [
        [round(max(0.0, min(1.0, start / width)), 3), round(max(0.0, min(1.0, end / width)), 3)]
        for start, end in runs
    ]
    note_parts = [
        "Hints are approximate and used only for prompting/validation.",
        "Computed from brightness projections across y=25%,40%,60%,75%.",
    ]
    if normalization_note:
        note_parts.append(normalization_note)
    return {
        "image_width": width,
        "image_height": height,
        "candidate_x_ranges_pixels": ranges_pixels,
        "candidate_x_ranges_normalized": ranges_normalized,
        "scan_rows_y_fraction": list(fractions),
        "notes": " ".join(note_parts),
    }


def compact_board_x_range_hints(hints: dict[str, Any]) -> str:
    payload = {
        "image_width": hints.get("image_width"),
        "image_height": hints.get("image_height"),
        "candidate_x_ranges_pixels": hints.get("candidate_x_ranges_pixels", []),
        "candidate_x_ranges_normalized": hints.get("candidate_x_ranges_normalized", []),
        "notes": hints.get("notes", ""),
    }
    return json.dumps(payload, separators=(",", ":"))


def create_board_hint_overlay(image_path: Path, hints: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        canvas = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    font = ImageFont.load_default()
    height = canvas.height
    colors = [(50, 190, 255, 70), (255, 210, 60, 70)]
    for index, item in enumerate(hints.get("candidate_x_ranges_pixels", []), start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        x1, x2 = int(item[0]), int(item[1])
        color = colors[index % 2]
        draw.rectangle([x1, 0, x2, height], fill=color, outline=(255, 255, 255, 180), width=2)
        draw.text((x1 + 3, 6), str(index), fill=(0, 0, 0, 255), font=font)
    canvas.save(output_path, quality=95)
    return output_path
