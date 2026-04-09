#!/usr/bin/env python3
"""Bucket temporal-3 (Animation Property Extraction) samples by motion type.

Reads benchmarks/temporal/AnimationPropertyExtraction/samples.csv, normalizes each
component's motion_type, and writes per-motion folders with video + JSON.

Usage::
    python scripts/bucket_animation_videos.py --dataset-root data/lica-benchmarks-dataset
    python scripts/bucket_animation_videos.py --dataset-root data/lica-benchmarks-dataset --copy

Also writes a slim ``outputs/animation_final/<motion>/scene_context_<id>.json`` tree (see ``--final-dir``).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Keep in sync with design_benchmarks.tasks.temporal (stdlib-only; avoids importing the package).
LICA_MOTION_TYPES: Tuple[str, ...] = (
    "ascend", "baseline", "block", "blur", "bounce", "breathe", "burst",
    "clarify", "drift", "fade", "flicker", "merge", "neon", "pan",
    "photoFlow", "photoRise", "pop", "pulse", "rise", "roll", "rotate",
    "scrapbook", "shift", "skate", "stomp", "succession", "tectonic",
    "tumble", "typewriter", "wiggle", "wipe", "none",
)


def normalize_motion_type(raw: str) -> str:
    text = raw.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    canon_map = {t.lower().replace("_", ""): t for t in LICA_MOTION_TYPES}
    if text in canon_map:
        return canon_map[text]
    for canon_key, canon_val in canon_map.items():
        if text in canon_key or canon_key in text:
            return canon_val
    return raw.strip().lower()

DEFAULT_MOTIONS = ("rise", "fade", "pop", "pan")
CSV_REL = Path("benchmarks/temporal/AnimationPropertyExtraction/samples.csv")


def _parse_expected(raw: str) -> List[Dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def layout_json_path(dataset_root: Path, image_rel: str) -> Optional[Path]:
    """Map ``lica-data/images/<template>/<id>.mp4`` → layout JSON path."""
    rel = Path(image_rel)
    parts = rel.parts
    if len(parts) >= 4 and parts[0] == "lica-data" and parts[1] == "images":
        template_id, file_name = parts[2], parts[3]
        layout_id = Path(file_name).stem
        cand = dataset_root / "lica-data" / "layouts" / template_id / f"{layout_id}.json"
        return cand if cand.is_file() else None
    return None


def layout_component_by_id(
    layout_file: Optional[Path], component_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Return the ``components[]`` dict whose ``id`` matches ``component_id``."""
    if not layout_file or not layout_file.is_file() or not component_id:
        return None
    try:
        data = json.loads(layout_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    comps = data.get("components")
    if not isinstance(comps, list):
        return None
    cid = str(component_id).strip()
    for c in comps:
        if isinstance(c, dict) and str(c.get("id", "")).strip() == cid:
            return c
    return None


def image_src_for_component(layout_file: Optional[Path], component_id: Optional[str]) -> Optional[str]:
    """Return ``components[].src`` for the component whose ``id`` matches ``component_id``."""
    c = layout_component_by_id(layout_file, component_id)
    if not c:
        return None
    src = c.get("src")
    return src if isinstance(src, str) else None


def layout_geometry_from_component(comp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``left``, ``top``, ``width``, ``height`` from a layout component (often px strings)."""
    if not comp:
        return {"left": None, "top": None, "width": None, "height": None}
    out: Dict[str, Any] = {}
    for key in ("left", "top", "width", "height"):
        val = comp.get(key)
        out[key] = val if val is not None else None
    return out


def _link_or_copy_video(src: Path, dest: Path, *, use_copy: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if use_copy:
        shutil.copy2(src, dest)
    else:
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)


def _minimal_scene_payload(
    *,
    sample_id: str,
    row_index: int,
    comp_index: int,
    comp_id: Optional[str],
    motion_type: str,
    image_src: Optional[str],
    video_rel: str,
    anim_obj: Dict[str, Any],
    layout_component: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Small JSON for animation_final (no prompt, no full expected_output)."""
    geom = layout_geometry_from_component(layout_component)
    payload: Dict[str, Any] = {
        "sample_id": sample_id,
        "bundle_id": f"{sample_id}_r{row_index}_c{comp_index}",
        "component_id": comp_id,
        "motion_type": motion_type,
        "image_src": image_src,
        "video_path": video_rel,
        "duration_seconds": anim_obj.get("duration_seconds"),
        "start_time_seconds": anim_obj.get("start_time_seconds"),
        "left": geom["left"],
        "top": geom["top"],
        "width": geom["width"],
        "height": geom["height"],
    }
    if layout_component and layout_component.get("type") is not None:
        payload["component_type"] = layout_component.get("type")
    return payload


def run(
    dataset_root: Path,
    out_dir: Path,
    target_motions: Set[str],
    *,
    use_copy: bool,
    dry_run: bool,
    final_dir: Optional[Path],
) -> Tuple[Counter, Counter, int, int]:
    """Returns (histogram_all, bucket_counts, n_written_dirs, n_written_final)."""
    csv_path = dataset_root / CSV_REL
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run: python scripts/download_data.py --out-dir data"
        )

    histogram_all: Counter = Counter()
    bucket_counts: Counter = Counter()
    n_written = 0
    n_final = 0
    canonical_targets = {normalize_motion_type(m) for m in target_motions}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            sample_id = row.get("sample_id", f"row{row_index}")
            image_rel = row.get("image_path", "")
            video_path = (dataset_root / image_rel).resolve()
            expected_raw = row.get("expected_output", "[]")
            items = _parse_expected(expected_raw)

            for comp_index, obj in enumerate(items):
                raw_mt = str(obj.get("motion_type", ""))
                mt = normalize_motion_type(raw_mt)
                histogram_all[mt] += 1

                if mt not in canonical_targets:
                    continue

                bucket_counts[mt] += 1
                if dry_run:
                    n_written += 1
                    continue

                if not video_path.is_file():
                    print(f"WARN row {row_index} sample_id={sample_id}: missing video {video_path}", file=sys.stderr)
                    continue

                subdir = out_dir / mt / f"{sample_id}_r{row_index}_c{comp_index}"
                subdir.mkdir(parents=True, exist_ok=True)

                anim_path = subdir / "animation_component.json"
                anim_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

                _link_or_copy_video(video_path, subdir / "video.mp4", use_copy=use_copy)

                layout = layout_json_path(dataset_root, image_rel)
                if layout is not None:
                    shutil.copy2(layout, subdir / "layout.json")
                else:
                    manifest = {
                        "layout_json": None,
                        "reason": "layout not found for image_path",
                        "image_path": image_rel,
                    }
                    (subdir / "layout_manifest.json").write_text(
                        json.dumps(manifest, indent=2), encoding="utf-8"
                    )

                comp_id = row.get("component_id")
                layout_comp = layout_component_by_id(layout, comp_id)
                img_src: Optional[str] = None
                if layout_comp:
                    s = layout_comp.get("src")
                    img_src = s if isinstance(s, str) else None
                ctx = {
                    "sample_id": sample_id,
                    "row_index": row_index,
                    "component_index": comp_index,
                    "component_id": comp_id,
                    "image_src": img_src,
                    "image_path": image_rel,
                    "expected_output_full": items,
                }
                (subdir / "scene_context.json").write_text(
                    json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                n_written += 1

                if final_dir is not None:
                    bundle_id = f"{sample_id}_r{row_index}_c{comp_index}"
                    final_sub = final_dir / mt
                    final_sub.mkdir(parents=True, exist_ok=True)
                    mini = _minimal_scene_payload(
                        sample_id=sample_id,
                        row_index=row_index,
                        comp_index=comp_index,
                        comp_id=comp_id,
                        motion_type=mt,
                        image_src=img_src,
                        video_rel=image_rel,
                        anim_obj=obj,
                        layout_component=layout_comp,
                    )
                    final_path = final_sub / f"scene_context_{bundle_id}.json"
                    final_path.write_text(
                        json.dumps(mini, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    n_final += 1

    return histogram_all, bucket_counts, n_written, n_final


def _print_histogram(title: str, hist: Counter, *, limit: int = 50) -> None:
    print(title)
    for mt, c in hist.most_common(limit):
        print(f"  {mt}: {c}")
    if len(hist) > limit:
        print(f"  ... ({len(hist) - limit} more types)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group AnimationPropertyExtraction components into motion-type folders.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/lica-benchmarks-dataset"),
        help="Root containing lica-data/ and benchmarks/ (default: data/lica-benchmarks-dataset)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/animation_buckets"),
        help="Output directory (default: outputs/animation_buckets)",
    )
    parser.add_argument(
        "--motions",
        nargs="*",
        default=list(DEFAULT_MOTIONS),
        help=f"Motion types to bucket (default: {list(DEFAULT_MOTIONS)})",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy MP4s instead of symlinking",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count; do not write files",
    )
    parser.add_argument(
        "--final-dir",
        type=Path,
        default=Path("outputs/animation_final"),
        help="Slim scene_context_<id>.json only per motion (default: outputs/animation_final)",
    )
    parser.add_argument(
        "--no-final",
        action="store_true",
        help="Do not write the animation_final JSON tree",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    out = args.out_dir.resolve()
    unknown = [m for m in args.motions if normalize_motion_type(m) not in LICA_MOTION_TYPES]
    if unknown:
        print(
            "WARN: these labels are not in LICA_MOTION_TYPES after normalization "
            f"(still used as filter targets): {unknown}",
            file=sys.stderr,
        )

    final_path: Optional[Path] = None if args.no_final else args.final_dir.resolve()

    hist, buckets, n_written, n_final = run(
        root,
        out,
        set(args.motions),
        use_copy=args.copy,
        dry_run=args.dry_run,
        final_dir=final_path,
    )

    _print_histogram("All component motion_type counts (normalized):", hist)
    print()
    _print_histogram("Bucketed counts (target motions only):", buckets, limit=len(args.motions) + 10)
    print()
    print(f"Written bucket entries: {n_written} → {out}")
    if final_path is not None and not args.dry_run:
        print(f"Written animation_final JSON files: {n_final} → {final_path}")
    for m in sorted(set(normalize_motion_type(x) for x in args.motions)):
        if buckets.get(m, 0) == 0:
            print(f"WARN: no samples bucketed for motion '{m}'", file=sys.stderr)


if __name__ == "__main__":
    main()
