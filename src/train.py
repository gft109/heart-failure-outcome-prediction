"""
Model definition and hyperparameter tuning (step 3).

Every model is a Pipeline (median imputation -> standardization -> classifier)
so that all data-driven preprocessing is refitted inside each training fold.
"""

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import RANDOM_STATE

N_SPLITS = 10


def _pipeline(classifier):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", classifier),
    ])


def get_models():
    """Candidate models and their hyperparameter grids.

    Returns {name: (kind, pipeline, grid)}, kind being "linear" or "non-linear"."""
    return {
        "Regressão logística (Ridge)": (
            "linear",
            _pipeline(LogisticRegression(max_iter=5000)),
            {"model__C": np.logspace(-3, 2, 11)},
        ),
        "LDA": (
            "linear",
            _pipeline(LinearDiscriminantAnalysis(solver="lsqr")),
            {"model__shrinkage": [None, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]},
        ),
        "SVM (kernel RBF)": (
            "non-linear",
            _pipeline(SVC(kernel="rbf")),
            {"model__C": [0.01, 0.1, 1, 10], "model__gamma": [0.001, 0.003, 0.01, 0.03, 0.1]},
        ),
        "Random Forest": (
            "non-linear",
            _pipeline(RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=RANDOM_STATE)),
            {"model__max_features": [2, 4, 8], "model__min_samples_leaf": [5, 15, 30]},
        ),
        "Gradient Boosting": (
            "non-linear",
            _pipeline(GradientBoostingClassifier(subsample=0.8, random_state=RANDOM_STATE)),
            {"model__n_estimators": [100, 300], "model__learning_rate": [0.01, 0.05],
             "model__max_depth": [1, 2, 3]},
        ),
        "KNN": (
            "non-linear",
            _pipeline(KNeighborsClassifier()),
            {"model__n_neighbors": [5, 15, 25, 51, 75, 101, 151], "model__weights": ["uniform", "distance"]},
        ),
    }


def tune_models(X_train, y_train, models=None):
    """Tune every model by stratified 10-fold cross-validation (AUC) on the
    training set and refit the best configuration on the whole training set.

    Returns (summary table, {name: fitted GridSearchCV})."""
    models = models or get_models()
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows, searches = [], {}
    for name, (kind, pipeline, grid) in models.items():
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=folds, n_jobs=-1, return_train_score=True)
        search.fit(X_train, y_train)
        best = search.best_index_
        results = search.cv_results_
        rows.append({
            "model": name,
            "kind": kind,
            "cv_auc_mean": results["mean_test_score"][best],
            "cv_auc_se": results["std_test_score"][best] / np.sqrt(N_SPLITS),
            "train_auc_mean": results["mean_train_score"][best],
            "n_configurations": len(results["params"]),
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        })
        searches[name] = search

    summary = pd.DataFrame(rows).set_index("model")
    summary[["cv_auc_mean", "cv_auc_se", "train_auc_mean"]] = summary[["cv_auc_mean", "cv_auc_se", "train_auc_mean"]].round(4)
    return summary, searches


def cv_fold_scores(searches):
    """AUC of the best configuration of each model in each validation fold
    (same folds for all models, so they can be compared pairwise)."""
    scores = {}
    for name, search in searches.items():
        best = search.best_index_
        scores[name] = [search.cv_results_[f"split{i}_test_score"][best] for i in range(N_SPLITS)]
    return pd.DataFrame(scores, index=[f"fold {i + 1}" for i in range(N_SPLITS)])
