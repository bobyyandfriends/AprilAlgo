"""Machine learning utilities (v0.3+)."""

from aprilalgo.ml.cv import PurgedKFold, learning_matrix
from aprilalgo.ml.evaluator import purged_cv_evaluate
from aprilalgo.ml.explain import shap_importance_table, shap_values_table
from aprilalgo.ml.features import (
    DEFAULT_EXCLUDED_FROM_FEATURES,
    align_features_and_labels,
    build_feature_matrix,
    extract_feature_matrix,
    feature_column_names,
)
from aprilalgo.ml.importance import (
    permutation_importance_table,
    xgb_importance_table,
)
from aprilalgo.ml.oof import compute_primary_oof
from aprilalgo.ml.trainer import (
    ModelBundle,
    load_model_bundle,
    proba_positive_takeprofit,
    save_model_bundle,
    train_xgb_classifier,
)

__all__ = [
    "DEFAULT_EXCLUDED_FROM_FEATURES",
    "ModelBundle",
    "PurgedKFold",
    "align_features_and_labels",
    "build_feature_matrix",
    "extract_feature_matrix",
    "feature_column_names",
    "learning_matrix",
    "compute_primary_oof",
    "load_model_bundle",
    "permutation_importance_table",
    "shap_importance_table",
    "shap_values_table",
    "proba_positive_takeprofit",
    "purged_cv_evaluate",
    "save_model_bundle",
    "train_xgb_classifier",
    "xgb_importance_table",
]
