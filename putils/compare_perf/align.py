from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


_TOKEN_SPLIT_PATTERN = re.compile(r"[^a-z0-9]+")
_PATH_SPLIT_PATTERN = re.compile(r"[./:_-]+")
_LAYER_PATTERN = re.compile(r"(?:layer|layers|block|blocks)[._-]?(\d+)", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\d+")
_TOKEN_SYNONYMS = {
    "attn": "attention",
    "enc": "encoder",
    "ffn": "feedforward",
    "mlp": "feedforward",
    "block": "layer",
    "blocks": "layers",
    "out": "output",
}
_MODULE_TYPE_HINTS = {
    "attention": {"attention", "selfattention", "crossattention"},
    "feedforward": {"feedforward", "ffn", "mlp"},
    "normalization": {"norm", "layernorm", "rmsnorm"},
    "embedding": {"embedding", "embed"},
    "projection": {"projection", "proj", "linear"},
    "loss": {"loss"},
    "optimizer": {"optimizer"},
    "backward": {"backward"},
}
_SCORE_WEIGHTS = {
    "token": 0.35,
    "path": 0.25,
    "type": 0.20,
    "layer": 0.12,
    "order": 0.08,
}


class AlignmentMappingCache:
    def __init__(self, cache_file: str | Path):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, left_modules: list[str], right_modules: list[str]) -> str:
        hasher = hashlib.sha1()
        hasher.update("\n".join(left_modules).encode("utf-8"))
        hasher.update(b"\n---\n")
        hasher.update("\n".join(right_modules).encode("utf-8"))
        return hasher.hexdigest()

    def _read_all(self) -> dict[str, Any]:
        if not self.cache_file.exists():
            return {"version": 1, "entries": {}}
        with self.cache_file.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, dict):
            return {"version": 1, "entries": {}}
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            payload["entries"] = {}
        return payload

    def _write_all(self, payload: Mapping[str, Any]) -> None:
        with self.cache_file.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, sort_keys=True, indent=2)

    def load_confirmed_mappings(
        self,
        *,
        left_modules: list[str],
        right_modules: list[str],
    ) -> dict[str, str]:
        payload = self._read_all()
        key = self._cache_key(left_modules=left_modules, right_modules=right_modules)
        value = payload.get("entries", {}).get(key, {})
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v) for k, v in value.items()}

    def save_confirmed_mappings(
        self,
        *,
        left_modules: list[str],
        right_modules: list[str],
        confirmed_mappings: Mapping[str, str],
    ) -> None:
        payload = self._read_all()
        key = self._cache_key(left_modules=left_modules, right_modules=right_modules)
        entries = dict(payload.get("entries", {}))
        entries[key] = {str(k): str(v) for k, v in sorted(confirmed_mappings.items())}
        next_payload = {
            "version": 1,
            "entries": entries,
        }
        self._write_all(next_payload)


def _extract_layer_index(module_name: str) -> int | None:
    match = _LAYER_PATTERN.search(module_name)
    if match:
        return int(match.group(1))

    number_match = _NUMBER_PATTERN.search(module_name)
    if number_match:
        return int(number_match.group(0))
    return None


def _tokenize(module_name: str) -> set[str]:
    lowered = module_name.lower()
    raw_tokens = [token for token in _TOKEN_SPLIT_PATTERN.split(lowered) if token]
    normalized_tokens = []
    for token in raw_tokens:
        normalized_tokens.append(_TOKEN_SYNONYMS.get(token, token))
    return set(normalized_tokens)


def _path_fragments(module_name: str) -> list[str]:
    lowered = module_name.lower()
    fragments = [item for item in _PATH_SPLIT_PATTERN.split(lowered) if item]
    return [_TOKEN_SYNONYMS.get(item, item) for item in fragments]


def _detect_module_type(module_name: str) -> str | None:
    tokens = _tokenize(module_name)
    for module_type, candidates in _MODULE_TYPE_HINTS.items():
        if tokens.intersection(candidates):
            return module_type
    return None


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return 0.0
    return intersection / union


def _score_candidate(
    *,
    left_name: str,
    right_name: str,
    left_order: int,
    right_order: int,
    total_orders: int,
) -> dict[str, Any]:
    left_tokens = _tokenize(left_name)
    right_tokens = _tokenize(right_name)
    token_score = _jaccard_similarity(left_tokens, right_tokens)

    left_fragments = _path_fragments(left_name)
    right_fragments = _path_fragments(right_name)
    left_parent = ".".join(left_fragments[:-1]) if len(left_fragments) > 1 else ""
    right_parent = ".".join(right_fragments[:-1]) if len(right_fragments) > 1 else ""
    parent_score = _jaccard_similarity(_tokenize(left_parent), _tokenize(right_parent))
    left_leaf = left_fragments[-1] if left_fragments else ""
    right_leaf = right_fragments[-1] if right_fragments else ""
    leaf_score = _jaccard_similarity(_tokenize(left_leaf), _tokenize(right_leaf))
    max_depth = max(len(left_fragments), len(right_fragments), 1)
    depth_score = max(0.0, 1.0 - (abs(len(left_fragments) - len(right_fragments)) / max_depth))
    path_score = (parent_score * 0.45) + (leaf_score * 0.45) + (depth_score * 0.10)

    left_type = _detect_module_type(left_name)
    right_type = _detect_module_type(right_name)
    if left_type is None and right_type is None:
        type_score = 0.50
    elif left_type is None or right_type is None:
        type_score = 0.35
    elif left_type == right_type:
        type_score = 1.0
    else:
        type_score = 0.0

    left_layer = _extract_layer_index(left_name)
    right_layer = _extract_layer_index(right_name)
    layer_score = 0.0
    if left_layer is not None and right_layer is not None:
        layer_gap = abs(left_layer - right_layer)
        layer_score = max(0.0, 1.0 - (layer_gap / 8.0))

    distance = abs(left_order - right_order)
    order_score = max(0.0, 1.0 - (distance / max(total_orders, 1)))

    raw_score = (
        (token_score * _SCORE_WEIGHTS["token"])
        + (path_score * _SCORE_WEIGHTS["path"])
        + (type_score * _SCORE_WEIGHTS["type"])
        + (layer_score * _SCORE_WEIGHTS["layer"])
        + (order_score * _SCORE_WEIGHTS["order"])
    )
    score = round(max(0.0, min(1.0, raw_score)), 6)

    return {
        "confidence": score,
        "score_components": {
            "token": round(token_score, 6),
            "path": round(path_score, 6),
            "type": round(type_score, 6),
            "layer": round(layer_score, 6),
            "order": round(order_score, 6),
            "weights": dict(_SCORE_WEIGHTS),
            "left_module_type": left_type,
            "right_module_type": right_type,
        },
    }


def align_modules(
    left_modules: list[str],
    right_modules: list[str],
    *,
    confirmed_mappings: Mapping[str, str] | None = None,
    cache: AlignmentMappingCache | None = None,
    candidate_threshold: float = 0.30,
    auto_match_threshold: float = 0.85,
    ambiguous_margin: float = 0.05,
) -> dict[str, Any]:
    left_normalized = [str(item) for item in left_modules]
    right_normalized = [str(item) for item in right_modules]

    cache_mappings: dict[str, str] = {}
    if cache is not None:
        cache_mappings = cache.load_confirmed_mappings(
            left_modules=left_normalized,
            right_modules=right_normalized,
        )

    manual_mappings = {str(k): str(v) for k, v in (confirmed_mappings or {}).items()}
    effective_mappings = dict(cache_mappings)
    effective_mappings.update(manual_mappings)

    total_orders = max(len(left_normalized), len(right_normalized), 1)
    used_right: set[str] = set()
    matched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unmatched_left: list[str] = []

    for left_order, left_name in enumerate(left_normalized):
        candidates: list[dict[str, Any]] = []
        for right_order, right_name in enumerate(right_normalized):
            score_payload = _score_candidate(
                left_name=left_name,
                right_name=right_name,
                left_order=left_order,
                right_order=right_order,
                total_orders=total_orders,
            )
            confidence = float(score_payload["confidence"])
            if confidence < candidate_threshold:
                continue
            candidates.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "confidence": confidence,
                    "score_components": dict(score_payload["score_components"]),
                    "status": "candidate",
                }
            )

        candidates.sort(key=lambda item: (-item["confidence"], item["right"]))

        confirmed_target = effective_mappings.get(left_name)
        if confirmed_target is not None and confirmed_target in right_normalized:
            if confirmed_target in used_right:
                ambiguous.append(
                    {
                        "left": left_name,
                        "status": "ambiguous",
                        "reason": "confirmed_target_conflict",
                        "candidates": candidates,
                    }
                )
                continue

            confirmed_candidate = next((item for item in candidates if item["right"] == confirmed_target), None)
            if confirmed_candidate is None:
                score_payload = _score_candidate(
                    left_name=left_name,
                    right_name=confirmed_target,
                    left_order=left_order,
                    right_order=right_normalized.index(confirmed_target),
                    total_orders=total_orders,
                )
                confidence = float(score_payload["confidence"])
                score_components = dict(score_payload["score_components"])
            else:
                confidence = float(confirmed_candidate["confidence"])
                score_components = dict(confirmed_candidate.get("score_components", {}))
            source = "cache" if left_name in cache_mappings and left_name not in manual_mappings else "manual"
            matched.append(
                {
                    "left": left_name,
                    "right": confirmed_target,
                    "confidence": confidence,
                    "score_components": score_components,
                    "status": "confirmed",
                    "source": source,
                }
            )
            used_right.add(confirmed_target)
            continue

        open_candidates = [item for item in candidates if item["right"] not in used_right]
        if not open_candidates:
            unmatched_left.append(left_name)
            continue

        best = open_candidates[0]
        second = open_candidates[1] if len(open_candidates) > 1 else None
        if best["confidence"] >= auto_match_threshold:
            if second is None or (best["confidence"] - second["confidence"]) >= ambiguous_margin:
                matched.append(
                    {
                        "left": left_name,
                        "right": best["right"],
                        "confidence": best["confidence"],
                        "score_components": dict(best.get("score_components", {})),
                        "status": "auto_matched",
                        "source": "rule",
                    }
                )
                used_right.add(best["right"])
                continue

        ambiguous.append(
            {
                "left": left_name,
                "status": "ambiguous",
                "reason": "low_confidence_or_multiple_candidates",
                "candidates": open_candidates,
            }
        )

    unmatched_right = [name for name in right_normalized if name not in used_right]

    if cache is not None and manual_mappings:
        merged_for_save = dict(cache_mappings)
        for left_name, right_name in manual_mappings.items():
            if left_name in left_normalized and right_name in right_normalized:
                merged_for_save[left_name] = right_name
        cache.save_confirmed_mappings(
            left_modules=left_normalized,
            right_modules=right_normalized,
            confirmed_mappings=merged_for_save,
        )

    return {
        "matched": matched,
        "ambiguous": ambiguous,
        "unmatched": {
            "left": unmatched_left,
            "right": unmatched_right,
        },
        "counts": {
            "matched": len(matched),
            "ambiguous": len(ambiguous),
            "unmatched": len(unmatched_left) + len(unmatched_right),
        },
    }
