"""Image benchmarks: image-1 and image-2 (layer-aware object insertion)."""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from design_benchmarks.base import BaseBenchmark, BenchmarkMeta, TaskType, benchmark
from design_benchmarks.metrics.core import fid as fid_metric

logger = logging.getLogger(__name__)


class _ImageUtils:
    """Utility stubs referenced by LayerAwareObjectInsertion.

    The static methods below are placeholders for image processing helpers
    (``_to_rgb_array``, ``_resize_to_match``, ``_inception_feature``,
    ``_to_gray_mask``, ``_read_image_size``) that need to be implemented
    when the image evaluation pipeline is fully wired up.
    """

    @staticmethod
    def _to_rgb_array(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        try:
            from PIL import Image

            if isinstance(value, (str, Path)):
                img = Image.open(str(value)).convert("RGB")
                return np.array(img)
            if hasattr(value, "convert"):
                return np.array(value.convert("RGB"))
        except Exception:
            pass
        return None

    @staticmethod
    def _resize_to_match(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        h, w = target_hw
        if image.shape[0] == h and image.shape[1] == w:
            return image
        try:
            from PIL import Image

            pil = Image.fromarray(image).resize((w, h), Image.LANCZOS)
            return np.array(pil)
        except Exception:
            return image

    @staticmethod
    def _inception_feature(image: np.ndarray) -> Optional[np.ndarray]:
        return None

    @staticmethod
    def _to_gray_mask(value: Any, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        arr = _ImageUtils._to_rgb_array(value)
        if arr is None:
            return None
        gray = np.mean(arr, axis=2).astype(np.uint8)
        if gray.shape[:2] != target_hw:
            gray_arr = _ImageUtils._resize_to_match(
                np.stack([gray] * 3, axis=2), target_hw
            )
            gray = gray_arr[:, :, 0]
        return gray

    @staticmethod
    def _read_image_size(path: Any) -> Tuple[int, int]:
        try:
            from PIL import Image

            img = Image.open(str(path))
            return img.size
        except Exception:
            return (0, 0)


TextRemoval = _ImageUtils


@benchmark
class LayerAwareObjectInsertion(BaseBenchmark):
    """image-1 — Layer-aware object insertion and asset synthesis (G15-style)."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="image-1",
        name="Layer-Aware Object Insertion & Asset Synthesis",
        task_type=TaskType.GENERATION,
        domain="image",
        description="Insert a reference asset into a masked layout region with seamless style blending",
        input_spec="Masked layout + insertion mask + reference asset (+ optional prompt/context)",
        output_spec="Composited layout with identity-preserving object insertion",
        metrics=[
            "clip_identity",
            "dino_identity",
            "dreamsim_distance",
            "fid",
            "lpips",
            "imagereward",
            "hpsv3",
        ],
    )

    DEFAULT_PROMPT = (
        "Insert the reference asset into the masked region and blend it naturally "
        "with the surrounding layout."
    )

    _clip_img_bundle: Any = None
    _dino_bundle: Any = None
    _dreamsim_bundle: Any = None
    _lpips_bundle: Any = None
    _sample_component_pattern = re.compile(r"^(?P<layout_id>.+)_component_(?P<index>\d+)$")
    MANIFEST_JSON_FILENAMES = (
        "g15_object_insertion_manifest.json",
        "layer_aware_insertion_manifest.json",
        "object_insertion_manifest.json",
        "manifest.json",
    )
    MANIFEST_CSV_FILENAMES = (
        "g15_object_insertion_manifest.csv",
        "layer_aware_insertion_manifest.csv",
        "object_insertion_manifest.csv",
        "manifest.csv",
    )

    def _resolve(self, base_dir: Path, value: str) -> str:
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((base_dir / p).resolve())

    def _should_include_reference_asset(self, sample: Dict[str, Any]) -> bool:
        return bool(sample.get("reference_asset"))

    def _should_include_asset_description(self, sample: Dict[str, Any]) -> bool:
        return False

    @staticmethod
    def _normalize_reference_alt(raw: Any, *, max_chars: int = 500) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _normalize_context(raw: Any, *, max_chars: int = 1400) -> str:
        text = str(raw or "").strip()
        if not text:
            return ""
        compact = " ".join(text.split())
        if len(compact) <= max_chars:
            return compact
        return compact[: max_chars - 3].rstrip() + "..."

    def _resolve_reference_asset_alt(
        self,
        *,
        base_dir: Path,
        row: Dict[str, Any],
        sample_id: str,
    ) -> str:
        direct = self._normalize_reference_alt(
            row.get("reference_asset_alt") or row.get("asset_alt") or row.get("alt")
        )
        if direct:
            return direct

        parsed = self._parse_sample_component(row=row, sample_id=sample_id)
        if parsed is None:
            return ""

        layout_id, component_index = parsed
        return self._lookup_alt_from_layout(
            base_dir=base_dir,
            layout_id=layout_id,
            component_index=component_index,
        )

    @classmethod
    def _parse_sample_component(
        cls,
        *,
        row: Dict[str, Any],
        sample_id: str,
    ) -> Optional[Tuple[str, int]]:
        layout_id = str(row.get("layout_id") or "").strip()
        index_raw = row.get("removed_component_index")
        if layout_id and index_raw is not None:
            try:
                index_val = int(index_raw)
            except Exception:
                index_val = -1
            if index_val >= 0:
                return layout_id, index_val

        match = cls._sample_component_pattern.match(sample_id)
        if not match:
            return None
        return match.group("layout_id"), int(match.group("index"))

    def _lookup_alt_from_layout(
        self,
        *,
        base_dir: Path,
        layout_id: str,
        component_index: int,
    ) -> str:
        if component_index < 0:
            return ""

        layout_path = base_dir / "layouts" / f"{layout_id}.json"
        if not layout_path.exists():
            return ""
        try:
            with open(layout_path, "r", encoding="utf-8") as f:
                layout_row = json.load(f)
        except Exception:
            return ""

        components = (layout_row.get("layout_config") or {}).get("components") or []
        if (
            not isinstance(components, list)
            or component_index >= len(components)
            or component_index < 0
        ):
            return ""
        cfg = components[component_index]
        if not isinstance(cfg, dict):
            return ""

        direct = self._normalize_reference_alt(cfg.get("alt"))
        if direct:
            return direct

        elem = cfg.get("element")
        if isinstance(elem, dict):
            nested = self._normalize_reference_alt(
                elem.get("alt") or elem.get("description")
            )
            if nested:
                return nested

        return ""

    def load_data(
        self,
        data_dir: Union[str, Path],
        *,
        n: Optional[int] = None,
        dataset_root: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        path = Path(data_dir).resolve()
        if path.is_dir():
            candidates = self.MANIFEST_JSON_FILENAMES + self.MANIFEST_CSV_FILENAMES
            matched = [path / name for name in candidates if (path / name).is_file()]
            if not matched:
                raise FileNotFoundError(
                    f"No G15 insertion manifest found under {path}. "
                    f"Tried: {', '.join(candidates)}"
                )
            path = matched[0]

        if not path.exists():
            raise FileNotFoundError(f"G15 insertion manifest not found: {path}")

        rows = self._read_insertion_manifest_rows(path)

        base_dir = path.parent
        samples: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            masked_layout = (
                row.get("masked_layout")
                or row.get("input_image")
                or row.get("masked_image")
                or row.get("layout_masked_image")
                or row.get("image")
            )
            mask = row.get("mask") or row.get("insert_mask") or row.get("component_mask")
            reference_asset = (
                row.get("reference_asset")
                or row.get("reference_image")
                or row.get("asset_image")
                or row.get("target_asset")
            )
            gt_image = (
                row.get("ground_truth_image")
                or row.get("target_image")
                or row.get("ground_truth")
            )

            if not masked_layout or not mask or not reference_asset or not gt_image:
                logger.warning("Incomplete G15 sample at index %d, skipping", i)
                continue

            sid = str(row.get("sample_id") or f"g15_insert_{i:04d}")
            reference_asset_alt = self._resolve_reference_asset_alt(
                base_dir=base_dir,
                row=row,
                sample_id=sid,
            )

            prompt = str(row.get("prompt") or row.get("instruction") or self.DEFAULT_PROMPT)
            context = row.get("contextual_cues") or row.get("context") or row.get("surrounding_layers")
            if isinstance(context, (dict, list)):
                context = json.dumps(context, ensure_ascii=False)
            context = str(context or "").strip()

            samples.append(
                {
                    "sample_id": sid,
                    "input_image": self._resolve(base_dir, str(masked_layout)),
                    "mask": self._resolve(base_dir, str(mask)),
                    "reference_asset": self._resolve(base_dir, str(reference_asset)),
                    "reference_asset_alt": reference_asset_alt,
                    "prompt": prompt,
                    "context": context,
                    "ground_truth": {
                        "image": self._resolve(base_dir, str(gt_image)),
                        "mask": self._resolve(base_dir, str(mask)),
                        "reference_asset": self._resolve(base_dir, str(reference_asset)),
                        "reference_asset_alt": reference_asset_alt,
                        "prompt": prompt,
                    },
                }
            )

        if n is not None:
            samples = samples[:n]
        return samples

    @classmethod
    def _read_insertion_manifest_rows(cls, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                with open(path, "r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    if not reader.fieldnames:
                        raise ValueError(f"CSV manifest has no header row: {path}")
                    return [
                        cls._normalize_insertion_manifest_csv_row(row)
                        for row in reader
                        if isinstance(row, dict)
                    ]
            except Exception as exc:
                raise ValueError(f"Failed to parse CSV manifest {path}: {exc}") from exc

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("samples") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Manifest must be a list or {'samples': [...]} format.")
        return [row for row in rows if isinstance(row, dict)]

    @classmethod
    def _normalize_insertion_manifest_csv_row(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(row)
        for key in (
            "sample_id",
            "masked_layout",
            "input_image",
            "masked_image",
            "layout_masked_image",
            "image",
            "mask",
            "insert_mask",
            "component_mask",
            "reference_asset",
            "reference_image",
            "asset_image",
            "target_asset",
            "ground_truth_image",
            "target_image",
            "ground_truth",
            "prompt",
            "instruction",
            "contextual_cues",
            "context",
            "surrounding_layers",
            "reference_asset_alt",
            "asset_alt",
            "alt",
        ):
            value = row.get(key)
            if isinstance(value, str):
                out[key] = value.replace("\\r\\n", "\n").replace("\\n", "\n").strip()

        out["removed_component_index"] = cls._safe_int(row.get("removed_component_index"), -1)
        out["mask_area_ratio"] = cls._safe_float(row.get("mask_area_ratio"), float("nan"))
        return out

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = float("nan")) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def build_model_input(self, sample: Dict[str, Any], *, modality: Any = None) -> Any:
        from design_benchmarks.models.base import ModelInput

        prompt = self._compose_prompt(sample)
        images = [sample["input_image"]]
        if self._should_include_reference_asset(sample) and sample.get("reference_asset"):
            images.append(sample["reference_asset"])

        metadata: Dict[str, Any] = {
            "mask": sample["mask"],
            "reference_asset": sample.get("reference_asset", ""),
            "reference_asset_alt": sample.get("reference_asset_alt", ""),
            "task": "g15_layer_aware_object_insertion",
            "benchmark_id": self.meta.id,
            "sample_id": str(sample.get("sample_id") or ""),
        }
        width, height = TextRemoval._read_image_size(sample.get("input_image"))
        if width > 0 and height > 0:
            metadata["target_width"] = width
            metadata["target_height"] = height

        return ModelInput(
            text=prompt,
            images=images,
            metadata=metadata,
        )

    def _compose_prompt(self, sample: Dict[str, Any]) -> str:
        user_intent = str(sample.get("prompt") or self.DEFAULT_PROMPT).strip()
        context = self._normalize_context(sample.get("context", ""))
        alt = self._normalize_reference_alt(sample.get("reference_asset_alt", ""))
        has_reference = self._should_include_reference_asset(sample)

        lines = [
            "You are an expert graphic design retoucher specialized in layer-aware object insertion.",
            "Task: insert exactly one target object into the editable masked region while preserving the rest of the layout.",
            "",
            "Objective:",
            f"- User intent: {user_intent}",
            "- Return one final composited image only (no text explanation).",
            "",
            "Input semantics:",
            "- Image #1 is the layout canvas with the target region removed/masked.",
            "- The mask defines editable pixels only (white=editable, black=preserve).",
        ]
        if has_reference:
            lines.extend(
                [
                    "- A reference asset image is provided as an additional input image.",
                    "- Preserve the reference asset's visual identity while matching local scene style.",
                ]
            )
        else:
            lines.extend(
                [
                    "- No reference asset image is provided.",
                    "- Reconstruct the target object from textual description and context.",
                ]
            )

        if self._should_include_asset_description(sample):
            if alt:
                lines.append(f"- Target object description: {alt}")
            else:
                lines.append("- Target object description: unavailable; infer from intent/context.")

        if context:
            lines.extend(["", "Contextual cues:", f"- {context}"])

        identity_requirement = (
            "- Identity: keep key shape/material/details consistent with the reference asset."
            if has_reference
            else "- Identity: generate an object that closely matches the target description."
        )

        lines.extend(
            [
                "",
                "Hard constraints (must satisfy all):",
                "- Edit only masked pixels; keep unmasked regions unchanged.",
                "- Keep the inserted object fully inside the editable mask.",
                "- Do not erase, warp, or occlude nearby text/logo/important elements.",
                "- Match perspective, lighting, shadow, and color grading to neighbors.",
                "- Insert exactly one coherent object (no duplicates/fragments).",
                "",
                "Quality checklist:",
                identity_requirement,
                "- Boundary blending: edges should look natural without obvious cutout artifacts.",
                "- Semantic fit: the inserted object should support the user intent and design context.",
                "",
                "Output: a single composited image.",
            ]
        )
        return "\n".join(lines)

    def parse_model_output(self, output: Any) -> Any:
        if output is None:
            return None

        images = getattr(output, "images", None)
        if images:
            return images[0]

        text = getattr(output, "text", "")
        if isinstance(text, str):
            cleaned = text.strip()
            cleaned = re.sub(r"^```(?:txt|text)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
            if cleaned.startswith(("http://", "https://")):
                return cleaned
            as_path = Path(cleaned)
            if as_path.exists():
                return str(as_path)
        return None

    def evaluate(self, predictions: List[Any], ground_truth: List[Any]) -> Dict[str, float]:
        clip_scores: List[float] = []
        dino_scores: List[float] = []
        dreamsim_scores: List[float] = []
        lpips_scores: List[float] = []
        imagereward_scores: List[float] = []
        hps_scores: List[float] = []
        fid_real_features: List[np.ndarray] = []
        fid_gen_features: List[np.ndarray] = []

        evaluated = 0
        identity_pairs = 0

        from design_benchmarks.tasks.layout import IntentToLayoutGeneration

        for pred_raw, gt_raw in zip(predictions, ground_truth):
            gt = self._normalize_gt_bundle(gt_raw)
            pred_img = TextRemoval._to_rgb_array(self._extract_image_like(pred_raw))
            gt_img = TextRemoval._to_rgb_array(self._extract_image_like(gt["image"]))
            ref_img = TextRemoval._to_rgb_array(self._extract_image_like(gt["reference_asset"]))

            if pred_img is None or gt_img is None:
                continue

            pred_img = TextRemoval._resize_to_match(pred_img, gt_img.shape[:2])
            evaluated += 1

            real_feat = TextRemoval._inception_feature(gt_img)
            gen_feat = TextRemoval._inception_feature(pred_img)
            if real_feat is not None and gen_feat is not None:
                fid_real_features.append(real_feat)
                fid_gen_features.append(gen_feat)

            lpips = self._lpips_distance(pred_img, gt_img)
            self._append_if_finite(lpips_scores, lpips)

            prompt = str(gt.get("prompt", ""))
            clip_fallback = IntentToLayoutGeneration._clip_score(prompt, pred_img)
            imagereward = IntentToLayoutGeneration._imagereward_score(prompt, pred_img)
            hps = IntentToLayoutGeneration._hpsv3_score(prompt, pred_img, clip_fallback=clip_fallback)
            self._append_if_finite(imagereward_scores, imagereward)
            self._append_if_finite(hps_scores, hps)

            if ref_img is None:
                continue

            mask = None
            if gt.get("mask"):
                mask = TextRemoval._to_gray_mask(gt["mask"], gt_img.shape[:2])

            pred_obj = self._extract_object_region(pred_img, mask)
            ref_obj = self._extract_object_region(ref_img, None)

            clip_identity = self._clip_image_similarity(pred_obj, ref_obj)
            dino_identity = self._dino_similarity(pred_obj, ref_obj)
            dreamsim_distance = self._dreamsim_distance(pred_obj, ref_obj)

            self._append_if_finite(clip_scores, clip_identity)
            self._append_if_finite(dino_scores, dino_identity)
            self._append_if_finite(dreamsim_scores, dreamsim_distance)
            identity_pairs += 1

        fid_score = float("nan")
        if len(fid_real_features) >= 2 and len(fid_gen_features) >= 2:
            try:
                fid_score = float(fid_metric(np.stack(fid_real_features), np.stack(fid_gen_features)))
                if math.isfinite(fid_score):
                    fid_score = max(0.0, fid_score)
            except Exception:
                fid_score = float("nan")

        denom = max(evaluated, 1)
        return {
            "clip_identity": self._mean_or_nan(clip_scores),
            "dino_identity": self._mean_or_nan(dino_scores),
            "dreamsim_distance": self._mean_or_nan(dreamsim_scores),
            "fid": fid_score,
            "lpips": self._mean_or_nan(lpips_scores),
            "imagereward": self._mean_or_nan(imagereward_scores),
            "hpsv3": self._mean_or_nan(hps_scores),
            "evaluated_samples": float(evaluated),
            "identity_pair_count": float(identity_pairs),
            "identity_coverage": identity_pairs / denom,
        }

    @staticmethod
    def _normalize_gt_bundle(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return {
                "image": raw.get("image", raw.get("ground_truth_image", raw)),
                "reference_asset": (
                    raw.get("reference_asset")
                    or raw.get("reference_image")
                    or raw.get("asset_image")
                ),
                "mask": raw.get("mask") or raw.get("insert_mask") or raw.get("component_mask"),
                "prompt": str(raw.get("prompt", "")),
            }
        return {"image": raw, "reference_asset": None, "mask": None, "prompt": ""}

    @staticmethod
    def _extract_image_like(value: Any) -> Any:
        if isinstance(value, dict):
            for key in ("image", "output_image", "predicted_image", "path"):
                if key in value:
                    return value[key]

        images = getattr(value, "images", None)
        if images:
            return images[0]
        return value

    @staticmethod
    def _extract_object_region(image: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return image

        ys, xs = np.where(mask > 127)
        if ys.size == 0 or xs.size == 0:
            return image

        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        region = image[y1:y2, x1:x2].copy()
        local_mask = (mask[y1:y2, x1:x2] > 127)

        isolated = np.full_like(region, 255)
        isolated[local_mask] = region[local_mask]
        return isolated

    @classmethod
    def _clip_image_similarity(cls, img_a: np.ndarray, img_b: np.ndarray) -> float:
        if cls._clip_img_bundle is None:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
                cls._clip_img_bundle = (model, processor, torch, device)
            except Exception as exc:
                logger.info("CLIP image-image metric unavailable: %s", exc)
                cls._clip_img_bundle = False

        if not cls._clip_img_bundle:
            return float("nan")

        model, processor, torch, device = cls._clip_img_bundle
        try:
            from PIL import Image

            inputs = processor(images=[Image.fromarray(img_a), Image.fromarray(img_b)], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                feats = model.get_image_features(pixel_values=pixel_values)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return float((feats[0] @ feats[1]).item())
        except Exception:
            return float("nan")

    @classmethod
    def _dino_similarity(cls, img_a: np.ndarray, img_b: np.ndarray) -> float:
        if cls._dino_bundle is None:
            try:
                import torch
                from transformers import AutoImageProcessor, AutoModel

                device = "cuda" if torch.cuda.is_available() else "cpu"
                processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                model = AutoModel.from_pretrained("facebook/dinov2-base").to(device).eval()
                cls._dino_bundle = (model, processor, torch, device)
            except Exception as exc:
                logger.info("DINO metric unavailable: %s", exc)
                cls._dino_bundle = False

        if not cls._dino_bundle:
            return float("nan")

        model, processor, torch, device = cls._dino_bundle
        try:
            from PIL import Image

            inputs = processor(images=[Image.fromarray(img_a), Image.fromarray(img_b)], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                feats = outputs.last_hidden_state[:, 0, :]
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return float((feats[0] @ feats[1]).item())
        except Exception:
            return float("nan")

    @classmethod
    def _dreamsim_distance(cls, img_a: np.ndarray, img_b: np.ndarray) -> float:
        if cls._dreamsim_bundle is None:
            try:
                import torch
                from dreamsim import (
                    dreamsim as dreamsim_factory,  # type: ignore[reportMissingImports]
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, preprocess = dreamsim_factory(pretrained=True, device=device)
                model.eval()
                cls._dreamsim_bundle = (model, preprocess, torch, device)
            except Exception as exc:
                logger.info("DreamSim unavailable, falling back to LPIPS proxy: %s", exc)
                cls._dreamsim_bundle = False

        if not cls._dreamsim_bundle:
            return cls._lpips_distance(img_a, img_b)

        model, preprocess, torch, device = cls._dreamsim_bundle
        try:
            from PIL import Image

            ta = preprocess(Image.fromarray(img_a))
            tb = preprocess(Image.fromarray(img_b))
            # dreamsim preprocess currently returns batched tensors (1, C, H, W),
            # but we also support older paths that return (C, H, W).
            if hasattr(ta, "ndim") and ta.ndim == 3:
                ta = ta.unsqueeze(0)
            if hasattr(tb, "ndim") and tb.ndim == 3:
                tb = tb.unsqueeze(0)
            ta = ta.to(device)
            tb = tb.to(device)
            with torch.no_grad():
                score = model(ta, tb)
            if hasattr(score, "item"):
                score = score.item()
            return float(score)
        except Exception:
            return cls._lpips_distance(img_a, img_b)

    @classmethod
    def _lpips_distance(cls, img_a: np.ndarray, img_b: np.ndarray) -> float:
        if img_a.shape[:2] != img_b.shape[:2]:
            img_b = TextRemoval._resize_to_match(img_b, img_a.shape[:2])

        if cls._lpips_bundle is None:
            try:
                import lpips
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = lpips.LPIPS(net="alex").to(device).eval()
                cls._lpips_bundle = (model, torch, device)
            except Exception as exc:
                logger.info("LPIPS unavailable, using MSE proxy: %s", exc)
                cls._lpips_bundle = False

        if not cls._lpips_bundle:
            mse = float(np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2))
            return mse / (255.0 ** 2)

        model, torch, device = cls._lpips_bundle
        try:
            ta = cls._to_lpips_tensor(img_a, torch).to(device)
            tb = cls._to_lpips_tensor(img_b, torch).to(device)
            with torch.no_grad():
                score = model(ta, tb)
            return float(score.item())
        except Exception:
            mse = float(np.mean((img_a.astype(np.float32) - img_b.astype(np.float32)) ** 2))
            return mse / (255.0 ** 2)

    @staticmethod
    def _to_lpips_tensor(image: np.ndarray, torch: Any) -> Any:
        x = image.astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        return x

    @staticmethod
    def _append_if_finite(bucket: List[float], value: float) -> None:
        if isinstance(value, float) and math.isfinite(value):
            bucket.append(value)

    @staticmethod
    def _mean_or_nan(values: List[float]) -> float:
        if not values:
            return float("nan")
        return float(sum(values) / len(values))


@benchmark
class LayerAwareObjectInsertionDescriptionGuided(LayerAwareObjectInsertion):
    """image-2 — Layer-aware insertion guided by textual asset description."""

    pipeline_implemented = True

    meta = BenchmarkMeta(
        id="image-2",
        name="Layer-Aware Object Insertion (Description-Guided)",
        task_type=TaskType.GENERATION,
        domain="image",
        description="Insert an object into a masked layout using textual asset description guidance",
        input_spec="Masked layout + insertion mask + asset description text (+ optional prompt/context)",
        output_spec="Composited layout with description-guided object insertion",
        metrics=[
            "clip_identity",
            "dino_identity",
            "dreamsim_distance",
            "fid",
            "lpips",
            "imagereward",
            "hpsv3",
        ],
    )

    DEFAULT_PROMPT = (
        "Insert an object matching the provided description into the masked region "
        "and blend it naturally with the surrounding layout."
    )

    def _should_include_reference_asset(self, sample: Dict[str, Any]) -> bool:
        return False

    def _should_include_asset_description(self, sample: Dict[str, Any]) -> bool:
        return True
