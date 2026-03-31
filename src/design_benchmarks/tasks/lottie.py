"""Lottie animation benchmarks (lottie-1, lottie-2).

Both tasks are implemented.

Data contract: each task reads ``{task-id}.json`` from the ``--data`` directory.
The JSON is an array of objects with ``question``, ``image``, and ``answer`` keys.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from design_benchmarks.utils.data_helpers import load_task_json
from design_benchmarks.utils.text_helpers import strip_thinking

# -- Lottie JSON helpers ----------------------------------------------------


def _parse_lottie_json(text: str) -> Optional[dict]:
    text = strip_thinking(text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


_LOTTIE_REQUIRED_KEYS = ("v", "fr", "ip", "op", "w", "h", "layers")


def _is_valid_lottie(data: Optional[dict]) -> bool:
    if not data:
        return False
    return all(k in data for k in _LOTTIE_REQUIRED_KEYS)


def _lottie_structural_similarity(pred: dict, gt: dict) -> float:
    scores: list = []
    pl = len(pred.get("layers", []))
    gl = len(gt.get("layers", []))
    scores.append(1.0 - abs(pl - gl) / max(pl, gl, 1))
    ptypes = {lay.get("ty") for lay in pred.get("layers", [])}
    gtypes = {lay.get("ty") for lay in gt.get("layers", [])}
    if ptypes or gtypes:
        scores.append(len(ptypes & gtypes) / len(ptypes | gtypes))
    pw, ph = pred.get("w", 1), pred.get("h", 1)
    gw, gh = gt.get("w", 1), gt.get("h", 1)
    scores.append(1.0 - abs(pw * ph - gw * gh) / max(pw * ph, gw * gh, 1))
    pfr, gfr = pred.get("fr", 30), gt.get("fr", 30)
    scores.append(1.0 - abs(pfr - gfr) / max(pfr, gfr, 1))
    return sum(scores) / len(scores) if scores else 0.0


def _evaluate_lottie(
    predictions: List[str], ground_truth: List[Dict[str, str]],
) -> Dict[str, float]:
    n = max(len(predictions), 1)
    val_s = struct_s = cl_s = mse_s = ssim_s = clip_s = 0.0
    for pred_text, gt_dict in zip(predictions, ground_truth):
        gt_text = gt_dict.get("lottie_json", "")
        cl_s += len(pred_text.encode("utf-8"))
        pred_data = _parse_lottie_json(pred_text)
        gt_data = _parse_lottie_json(gt_text)
        valid = _is_valid_lottie(pred_data)
        val_s += 1.0 if valid else 0.0
        if valid and gt_data:
            struct_s += _lottie_structural_similarity(pred_data, gt_data)
        mse_s += 1.0
        ssim_s += 0.0
    return {
        "frame_mse": mse_s / n,
        "frame_ssim": ssim_s / n,
        "clip_score": clip_s / n,
        "lottie_validity": val_s / n,
        "structural_similarity": struct_s / n,
        "code_length": cl_s / n,
    }


# ===========================================================================
# Implemented tasks
# ===========================================================================


@benchmark
class TextToLottieGeneration(BaseBenchmark):
    """lottie-1 — Generate Lottie animation JSON from a text description."""

    pipeline_implemented = True

    PROMPT = (
        "You are a Lottie animation generator. "
        "Given a description of an animation, output ONLY valid Lottie JSON "
        "(the standard Bodymovin format with keys: v, fr, ip, op, w, h, layers). "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="lottie-1",
        name="Text-to-Lottie Generation",
        task_type=TaskType.GENERATION,
        domain="lottie",
        description="Generate Lottie animation JSON from a text description",
        input_spec="Natural-language description of animation",
        output_spec="Lottie JSON (Bodymovin format)",
        metrics=["lottie_validity", "structural_similarity", "frame_mse", "frame_ssim", "code_length"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"lottie_{i:03d}",
                "ground_truth": {
                    "lottie_json": item["answer"],
                    "description": item["question"][0] if item.get("question") else "",
                },
                "description": item["question"][0] if item.get("question") else "",
                "image_path": (
                    str(data_root / item["image"])
                    if item.get("image") and not Path(item["image"]).is_absolute()
                    else item.get("image", "")
                ),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        return ModelInput(
            text=f"{self.PROMPT}\n\nDescription: {sample['description']}",
            images=[],
        )

    def parse_model_output(self, output):
        return strip_thinking(output.text.strip())

    def evaluate(self, predictions, ground_truth):
        return _evaluate_lottie(predictions, ground_truth)


@benchmark
class ImageTextToLottieGeneration(BaseBenchmark):
    """lottie-2 — Generate Lottie animation JSON from a keyframe image and description."""

    pipeline_implemented = True

    PROMPT = (
        "You are a Lottie animation generator. "
        "Given an animation keyframe image and its description, output ONLY valid "
        "Lottie JSON (the standard Bodymovin format with keys: v, fr, ip, op, w, h, layers). "
        "Do not include any explanation, markdown fences, or extra text."
    )

    meta = BenchmarkMeta(
        id="lottie-2",
        name="Image-Text-to-Lottie Generation",
        task_type=TaskType.GENERATION,
        domain="lottie",
        description="Generate Lottie animation JSON from a keyframe image and description",
        input_spec="Animation keyframe image + natural-language description",
        output_spec="Lottie JSON (Bodymovin format)",
        metrics=["lottie_validity", "structural_similarity", "frame_mse", "frame_ssim", "code_length"],
    )

    def load_data(self, data_dir, *, n=None, dataset_root: Union[str, Path]):
        data, data_root = load_task_json(data_dir, self.meta.id)
        samples = [
            {
                "sample_id": f"lottie_{i:03d}",
                "ground_truth": {
                    "lottie_json": item["answer"],
                    "description": item["question"][0] if item.get("question") else "",
                },
                "description": item["question"][0] if item.get("question") else "",
                "image_path": (
                    str(data_root / item["image"])
                    if item.get("image") and not Path(item["image"]).is_absolute()
                    else item.get("image", "")
                ),
            }
            for i, item in enumerate(data)
        ]
        return samples[:n] if n is not None else samples

    def build_model_input(self, sample, *, modality=None):
        from design_benchmarks.models.base import ModelInput

        images: list = []
        ip = sample.get("image_path", "")
        if ip and Path(ip).exists():
            images.append(ip)
        return ModelInput(
            text=f"{self.PROMPT}\n\nDescription: {sample['description']}",
            images=images,
        )

    def parse_model_output(self, output):
        return strip_thinking(output.text.strip())

    def evaluate(self, predictions, ground_truth):
        return _evaluate_lottie(predictions, ground_truth)
