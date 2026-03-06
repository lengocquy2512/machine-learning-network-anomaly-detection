from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Tuple

import joblib


@dataclass
class ModelMeta:
    best_model: str
    threshold: float
    score_type: str
    features: list[str]


def _infer_best_model_name(model: Any) -> str:
    name = type(model).__name__.lower()
    if "randomforest" in name:
        return "RF"
    if "xgb" in name or "xgboost" in name:
        return "XGB"
    if "svc" in name:
        return "SVC"
    if "mlp" in name:
        return "MLP"
    if "kneighbors" in name or "knn" in name:
        return "KNN"
    return type(model).__name__


def _infer_score_type(model: Any) -> str:
    if hasattr(model, "predict_proba"):
        return "proba"
    if hasattr(model, "decision_function"):
        return "decision"
    return "proba"


def load_best_model(models_dir: Path) -> Tuple[Any, ModelMeta]:
    path = Path(models_dir) / "best_model.joblib"
    obj = joblib.load(path)

    # hỗ trợ cả kiểu dump object trực tiếp hoặc dict
    if isinstance(obj, dict):
        model = obj.get("model", obj)
        thr = float(obj.get("thr", 0.5))
        features = list(obj.get("feature_names", obj.get("features", [])))
        best_model = str(obj.get("best_model") or _infer_best_model_name(model))
        score_type = str(obj.get("score_type") or _infer_score_type(model))
    else:
        model = obj
        thr = 0.5
        features = []
        best_model = _infer_best_model_name(model)
        score_type = _infer_score_type(model)

    if not features:
        raise ValueError("Không tìm thấy feature_names/features trong best_model.joblib")

    meta = ModelMeta(
        best_model=best_model,
        threshold=thr,
        score_type=score_type,
        features=features,
    )
    return model, meta
