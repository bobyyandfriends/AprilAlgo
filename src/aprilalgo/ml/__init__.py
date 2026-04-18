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
from aprilalgo.ml.meta_bundle import (
    MetaLogitBundle,
    load_meta_logit_bundle,
    save_meta_logit_bundle,
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
    "MetaLogitBundle",
    "ModelBundle",
    "PurgedKFold",
    "align_features_and_labels",
    "build_feature_matrix",
    "compute_primary_oof",
    "extract_feature_matrix",
    "feature_column_names",
    "learning_matrix",
    "load_meta_logit_bundle",
    "load_model_bundle",
    "permutation_importance_table",
    "shap_importance_table",
    "shap_values_table",
    "proba_positive_takeprofit",
    "purged_cv_evaluate",
    "save_meta_logit_bundle",
    "save_model_bundle",
    "train_xgb_classifier",
    "xgb_importance_table",
]
