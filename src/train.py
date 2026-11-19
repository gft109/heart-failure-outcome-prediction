"""
Step 3: the three models and their hyperparameter tuning.

Each model is a Pipeline, so the median imputation (and the scaling, for the
Lasso) is refitted on the training part of every fold. All three use
class_weight="balanced", which compensates the 41.5% / 58.5% class ratio.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.utils import RANDOM_STATE

N_SPLITS = 10


def get_models():
    """{name: (kind, pipeline, hyperparameter grid)}."""
    impute = ("impute", SimpleImputer(strategy="median"))
    return {
        "Árvore de decisão": (
            "não linear",
            Pipeline([impute, ("model", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))]),
            {"model__ccp_alpha": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.015, 0.02]},
        ),
        "Regressão logística Lasso": (
            "linear",
            Pipeline([impute, ("scale", StandardScaler()),
                      ("model", LogisticRegression(l1_ratio=1.0, solver="liblinear", class_weight="balanced",
                                                   random_state=RANDOM_STATE))]),
            {"model__C": list(np.logspace(-2.5, 0, 11))},
        ),
        "Random Forest": (
            "não linear",
            Pipeline([impute, ("model", RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                                               n_jobs=-1, random_state=RANDOM_STATE))]),
            {"model__max_features": ["sqrt", 0.05, 0.2], "model__min_samples_leaf": [5, 15, 30]},
        ),
    }


def tune_models(X_train, y_train):
    """Grid search with stratified 10-fold cross-validation (AUC) on the
    training set; the best configuration of each model is refitted on the
    whole training set.

    Returns (summary table, {name: fitted GridSearchCV})."""
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows, searches = [], {}
    for name, (kind, pipeline, grid) in get_models().items():
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=folds, n_jobs=-1,
                              return_train_score=True).fit(X_train, y_train)
        best = search.best_index_
        rows.append({
            "modelo": name,
            "tipo": kind,
            "auc_cv": search.cv_results_["mean_test_score"][best],
            "auc_cv_erro_padrao": search.cv_results_["std_test_score"][best] / np.sqrt(N_SPLITS),
            "auc_treino": search.cv_results_["mean_train_score"][best],
            "melhores_hiperparametros": {k.replace("model__", ""): round(v.item(), 4) if hasattr(v, "item") else v
                                         for k, v in search.best_params_.items()},
        })
        searches[name] = search
    return pd.DataFrame(rows).set_index("modelo").round(3), searches


def indicator_ablation(X_train, y_train, model_name="Regressão logística Lasso"):
    """Does the "exam not ordered" treatment help? Same model, grid and folds,
    with and without the not-ordered indicators (the measured values stay in
    both, imputed by the median)."""
    _, pipeline, grid = get_models()[model_name]
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    indicators = [c for c in X_train.columns if c.endswith("_not_ordered")]
    rows = {}
    for label, columns in [("com os indicadores", list(X_train.columns)),
                           ("sem os indicadores", [c for c in X_train.columns if c not in indicators])]:
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=folds, n_jobs=-1).fit(X_train[columns], y_train)
        rows[label] = {
            "n_atributos": len(columns),
            "auc_cv": search.best_score_,
            "auc_cv_erro_padrao": search.cv_results_["std_test_score"][search.best_index_] / np.sqrt(N_SPLITS),
        }
    return pd.DataFrame(rows).T.round(3)


def model_sizes(searches, n_features):
    """How big each fitted model is: leaves of the tree, features kept by the
    Lasso, trees in the forest."""
    tree = searches["Árvore de decisão"].best_estimator_.named_steps["model"]
    lasso = searches["Regressão logística Lasso"].best_estimator_.named_steps["model"]
    forest = searches["Random Forest"].best_estimator_.named_steps["model"]
    return pd.Series({
        "Árvore de decisão": f"{tree.get_n_leaves()} folhas, profundidade {tree.get_depth()}",
        "Regressão logística Lasso": f"{int((lasso.coef_ != 0).sum())} de {n_features} atributos mantidos",
        "Random Forest": f"{len(forest.estimators_)} árvores, todos os {n_features} atributos",
    }, name="tamanho do modelo")
