"""
Model assessment on the held-out test set (step 4).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.utils import RANDOM_STATE


def predict_scores(model, X):
    """Continuous score of the positive class (probability when available,
    otherwise the decision function, as in the SVM)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def bootstrap_auc_interval(y_true, scores, n_resamples=2000, level=0.95):
    """Percentile bootstrap confidence interval of the AUC (Aula 05): the test
    set is resampled with replacement and the AUC recomputed each time."""
    rng = np.random.default_rng(RANDOM_STATE)
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    n = len(y_true)
    aucs = []
    while len(aucs) < n_resamples:
        rows = rng.integers(0, n, n)
        if y_true[rows].min() == y_true[rows].max():
            continue  # resample with a single class: AUC undefined
        aucs.append(roc_auc_score(y_true[rows], scores[rows]))
    tail = (1 - level) / 2
    return float(np.quantile(aucs, tail)), float(np.quantile(aucs, 1 - tail))


def evaluate_on_test(searches, X_test, y_test):
    """Test-set AUC (with bootstrap confidence interval) of every tuned model,
    each already refitted on the whole training set."""
    rows = []
    for name, search in searches.items():
        scores = predict_scores(search.best_estimator_, X_test)
        low, high = bootstrap_auc_interval(y_test, scores)
        rows.append({"model": name, "test_auc": roc_auc_score(y_test, scores),
                     "auc_ci_low": low, "auc_ci_high": high})
    return pd.DataFrame(rows).set_index("model").round(3)


def choose_threshold_cv(model, X_train, y_train, n_splits=10):
    """Decision threshold chosen without the test set: out-of-fold predicted
    probabilities on the training set, then the cut-off that maximizes
    Youden's index (sensitivity + specificity - 1)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def threshold_metrics(model, X_test, y_test, thresholds):
    """Test-set classification metrics of one model at each decision threshold.

    `thresholds` maps a label to a probability cut-off."""
    scores = predict_scores(model, X_test)
    y_true = np.asarray(y_test)
    rows = {}
    for label, threshold in thresholds.items():
        predicted = scores >= threshold
        tp = int((predicted & (y_true == 1)).sum()); fn = int((~predicted & (y_true == 1)).sum())
        tn = int((~predicted & (y_true == 0)).sum()); fp = int((predicted & (y_true == 0)).sum())
        sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
        rows[label] = {
            "threshold": threshold,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2,
            "accuracy": (tp + tn) / len(y_true),
            "positive_predictive_value": tp / (tp + fp),
            "negative_predictive_value": tn / (tn + fn),
        }
    return pd.DataFrame(rows).T.round(3)


def confusion_table(model, X_test, y_test, threshold=0.5):
    """Confusion matrix of one model at a given threshold, as a labelled table."""
    predicted = (predict_scores(model, X_test) >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predicted)
    return pd.DataFrame(matrix, index=["Real: sem evento", "Real: evento"],
                        columns=["Previsto: sem evento", "Previsto: evento"])


def logistic_odds_ratios(model, feature_names):
    """Coefficients of a fitted logistic pipeline as odds ratios per standard
    deviation of each feature, sorted by absolute effect."""
    coef = pd.Series(model.named_steps["model"].coef_[0], index=feature_names, name="coefficient")
    table = coef.reindex(coef.abs().sort_values(ascending=False).index).to_frame()
    table["odds_ratio_per_sd"] = np.exp(table["coefficient"])
    return table.round(3)


def plot_roc_curves(searches, X_test, y_test, highlight):
    """ROC curves of all models on the test set; `highlight` is drawn in the
    series color and the others in grey."""
    import matplotlib.pyplot as plt
    from src.utils import SERIES_COLOR, NEUTRAL_COLOR, MUTED_COLOR

    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.plot([0, 1], [0, 1], color=MUTED_COLOR, linestyle=":", linewidth=0.8)
    for name, search in searches.items():
        if name == highlight:
            continue
        fpr, tpr, _ = roc_curve(y_test, predict_scores(search.best_estimator_, X_test))
        ax.plot(fpr, tpr, color=NEUTRAL_COLOR, linewidth=0.8)
    ax.plot([], [], color=NEUTRAL_COLOR, linewidth=0.8, label="Demais modelos")

    scores = predict_scores(searches[highlight].best_estimator_, X_test)
    fpr, tpr, _ = roc_curve(y_test, scores)
    ax.plot(fpr, tpr, color=SERIES_COLOR, linewidth=1.6,
            label=f"{highlight} (AUC = {roc_auc_score(y_test, scores):.2f})".replace(".", ","))

    ax.set_xlabel("1 − especificidade")
    ax.set_ylabel("Sensibilidade")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=6, frameon=False)
    return fig


def _select_tune_score(X_fit, y_fit, X_valid, y_valid, model_names):
    """One honest round: the whole feature selection and the hyperparameter
    tuning are done on (X_fit, y_fit) only; the AUC of each tuned model is then
    measured on (X_valid, y_valid), which took part in no decision."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from src import features, train

    selected, C = features.select_features(X_fit, y_fit)

    models = train.get_models()
    folds = StratifiedKFold(n_splits=train.N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    result = {"n_features": len(selected), "lasso_C": C, "features": selected, "cv_auc": {}, "auc": {}}
    for name in model_names:
        _, pipeline, grid = models[name]
        search = GridSearchCV(pipeline, grid, scoring="roc_auc", cv=folds).fit(X_fit[selected], y_fit)
        result["cv_auc"][name] = search.best_score_
        result["auc"][name] = roc_auc_score(y_valid, predict_scores(search.best_estimator_, X_valid[selected]))
    return result


def nested_cv_model_comparison(X_train, y_train, model_names=None):
    """Nested cross-validation on the training set, for an unbiased comparison
    of the models.

    Outer loop: the same stratified 10 folds used in step 3. In each outer
    fold, feature selection and tuning are redone with the other nine folds
    (inner loop) and every model is scored on the fold left out.

    Returns (AUC of each model in each outer fold, number of features selected
    in each outer fold)."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    from src import train

    model_names = model_names or list(train.get_models())
    outer = StratifiedKFold(n_splits=train.N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rounds = Parallel(n_jobs=-1)(
        delayed(_select_tune_score)(X_train.iloc[fit], y_train.iloc[fit], X_train.iloc[valid], y_train.iloc[valid], model_names)
        for fit, valid in outer.split(X_train, y_train))

    index = [f"fold {i + 1}" for i in range(len(rounds))]
    outer_scores = pd.DataFrame([r["auc"] for r in rounds], index=index)
    n_features = pd.Series([r["n_features"] for r in rounds], index=index, name="n_features")
    return outer_scores, n_features


def summarize_nested_cv(outer_scores, step3_cv_auc):
    """Mean nested AUC and its standard error per model, next to the AUC from
    the plain cross-validation of step 3 (features selected with the whole
    training set). Both use the same folds, so `optimism` (their difference)
    is the part of the step-3 AUC that came from selecting the features with
    the validation folds in sight."""
    n = len(outer_scores)
    table = pd.DataFrame({
        "nested_auc_mean": outer_scores.mean(),
        "nested_auc_se": outer_scores.std(ddof=1) / np.sqrt(n),
        "step3_cv_auc": step3_cv_auc,
    })
    table["optimism"] = table["step3_cv_auc"] - table["nested_auc_mean"]
    return table.sort_values("nested_auc_mean", ascending=False).round(4)


def paired_differences(outer_scores, reference):
    """Fold-by-fold AUC difference of each model to `reference` (same folds for
    all models): mean difference and its standard error. A difference smaller
    than about two standard errors is not distinguishable from noise."""
    n = len(outer_scores)
    diff = outer_scores.drop(columns=reference).sub(outer_scores[reference], axis=0)
    return pd.DataFrame({
        "auc_difference_mean": diff.mean(),
        "auc_difference_se": diff.std(ddof=1) / np.sqrt(n),
        "folds_better_than_reference": (diff > 0).sum(),
    }).round(4)


def _assess_one_split(X, y, seed, model_name):
    """One round: new 80/20 split, feature selection and tuning redone on the
    training part only, AUC measured on the test part."""
    from sklearn.model_selection import train_test_split

    X_fit, X_holdout, y_fit, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)
    r = _select_tune_score(X_fit, y_fit, X_holdout, y_holdout, [model_name])
    return {"seed": seed, "n_features": r["n_features"], "lasso_C": r["lasso_C"],
            "cv_auc": r["cv_auc"][model_name], "test_auc": r["auc"][model_name], "features": r["features"]}


def repeated_split_assessment(X, y, model_name, seeds):
    """Repeat the whole procedure (split -> feature selection -> tuning ->
    test) for several random splits, to estimate the expected performance of
    the method and how much it depends on the particular split.

    Returns (table with one row per split, selection share of each feature)."""
    from joblib import Parallel, delayed

    rounds = Parallel(n_jobs=-1)(delayed(_assess_one_split)(X, y, seed, model_name) for seed in seeds)

    counts = pd.Series([f for r in rounds for f in r["features"]]).value_counts()
    feature_share = (counts / len(rounds)).rename("share_of_splits").round(3)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in rounds]).set_index("seed")
    return table.round(4), feature_share


def plot_repeated_split_auc(test_aucs, official_auc):
    """Histogram of the test AUC across the repeated splits, with the mean and
    the official test set marked."""
    import matplotlib.pyplot as plt
    from src.utils import SERIES_COLOR, MUTED_COLOR, NEUTRAL_LIGHT_COLOR

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.hist(test_aucs, bins=12, color=NEUTRAL_LIGHT_COLOR, edgecolor=MUTED_COLOR, linewidth=0.5)
    ax.axvline(np.mean(test_aucs), color=MUTED_COLOR, linestyle="--", linewidth=1, label="média das divisões")
    ax.axvline(official_auc, color=SERIES_COLOR, linewidth=1.4, label="divisão usada no trabalho")
    ax.set_xlabel("AUC no conjunto de teste")
    ax.set_ylabel("Nº de divisões")
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    return fig
