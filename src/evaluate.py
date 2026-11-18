"""
Step 4: assessment on the held-out test set.

Every metric is compared with two chance references, so that a model that
only reproduces the class ratio is not mistaken for a useful one:
  - "sempre a classe majoritária": always predicts "no event";
  - "sorteio estratificado": random guesses in the class proportions.
AUC and balanced accuracy are 0.5 for both references, whatever the class
ratio; accuracy is not (the majority reference reaches 58.5%).
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from src.utils import RANDOM_STATE

N_BOOTSTRAP = 2000


def chance_references(X_train, y_train):
    """The two chance models, fitted on the training set."""
    return {
        "Acaso: sempre a classe majoritária": DummyClassifier(strategy="most_frequent").fit(X_train, y_train),
        "Acaso: sorteio estratificado": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE).fit(X_train, y_train),
    }


def _bootstrap_ci(y_true, scores, predicted, rng):
    """95% percentile bootstrap intervals of AUC, balanced accuracy and accuracy."""
    n = len(y_true)
    values = {"auc": [], "bacc": [], "acc": []}
    while len(values["auc"]) < N_BOOTSTRAP:
        rows = rng.integers(0, n, n)
        if y_true[rows].min() == y_true[rows].max():
            continue
        values["auc"].append(roc_auc_score(y_true[rows], scores[rows]))
        values["bacc"].append(balanced_accuracy_score(y_true[rows], predicted[rows]))
        values["acc"].append((y_true[rows] == predicted[rows]).mean())
    return {k: np.quantile(v, [0.025, 0.975]) for k, v in values.items()}


def test_metrics(models, X_test, y_test):
    """Test-set metrics of each model (decision threshold 0.5, i.e. the
    model's own prediction), with 95% bootstrap intervals for AUC and
    balanced accuracy and accuracy. `melhor_que_o_acaso` is True when the
    lower limits of the AUC and balanced-accuracy intervals are above 0.5;
    `acuracia_acima_da_majoritaria` when the lower limit of the accuracy
    interval is above the majority-class rate of the test set."""
    y_true = np.asarray(y_test)
    majority_rate = max(y_true.mean(), 1 - y_true.mean())
    rng = np.random.default_rng(RANDOM_STATE)
    rows = {}
    for name, model in models.items():
        scores = model.predict_proba(X_test)[:, 1]
        predicted = model.predict(X_test)
        ci = _bootstrap_ci(y_true, scores, predicted, rng)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        rows[name] = {
            "auc": roc_auc_score(y_true, scores),
            "auc_ic95": f"{ci['auc'][0]:.3f} – {ci['auc'][1]:.3f}",
            "acuracia_balanceada": balanced_accuracy_score(y_true, predicted),
            "acuracia_balanceada_ic95": f"{ci['bacc'][0]:.3f} – {ci['bacc'][1]:.3f}",
            "acuracia": (tp + tn) / len(y_true),
            "acuracia_ic95": f"{ci['acc'][0]:.3f} – {ci['acc'][1]:.3f}",
            "sensibilidade": tp / (tp + fn),
            "especificidade": tn / (tn + fp),
            "valor_preditivo_positivo": tp / (tp + fp) if tp + fp else np.nan,
            "melhor_que_o_acaso": bool(ci["auc"][0] > 0.5 and ci["bacc"][0] > 0.5),
            "acuracia_acima_da_majoritaria": bool(ci["acc"][0] > majority_rate),
        }
    return pd.DataFrame(rows).T.infer_objects().round(3)


def choose_model(cv_results, simplest_first):
    """Model with the best cross-validated AUC; among the models within one
    standard error of it, the simplest one (order given by `simplest_first`)."""
    best = cv_results.loc[cv_results["auc_cv"].idxmax()]
    tied = cv_results.index[cv_results["auc_cv"] >= best["auc_cv"] - best["auc_cv_erro_padrao"]]
    return next(name for name in simplest_first if name in tied)


def confusion_table(model, X_test, y_test):
    """Confusion matrix as a labelled table."""
    matrix = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
    return pd.DataFrame(matrix, index=["Real: sem evento", "Real: evento"],
                        columns=["Previsto: sem evento", "Previsto: evento"])


def plot_roc_curves(models, X_test, y_test):
    """ROC curves on the test set; the diagonal is the chance level."""
    import matplotlib.pyplot as plt
    from src.utils import MUTED_COLOR

    colors = ["#a3a29c", "#2a78d6", "#d6672a"]
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.plot([0, 1], [0, 1], color=MUTED_COLOR, linestyle=":", linewidth=0.8, label="Acaso (AUC = 0,50)")
    for (name, model), color in zip(models.items(), colors):
        scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, scores)
        label = f"{name} (AUC = {roc_auc_score(y_test, scores):.2f})".replace(".", ",")
        ax.plot(fpr, tpr, color=color, linewidth=1.3, label=label)
    ax.set_xlabel("1 − especificidade")
    ax.set_ylabel("Sensibilidade")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=5.5, frameon=False)
    return fig


def lasso_coefficients(lasso_pipeline, feature_names, top=15):
    """Non-zero Lasso coefficients (standardized scale) as odds ratios per
    standard deviation, largest effects first."""
    coef = pd.Series(lasso_pipeline.named_steps["model"].coef_[0], index=feature_names)
    coef = coef[coef != 0]
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(top)
    return pd.DataFrame({"coeficiente": coef, "razao_de_chances_por_dp": np.exp(coef)}).round(3)


def forest_importance(forest_pipeline, X_test, y_test, top=15):
    """Permutation importance of the Random Forest on the test set: how much
    the AUC drops when one feature is shuffled."""
    result = permutation_importance(forest_pipeline, X_test, y_test, scoring="roc_auc",
                                    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    table = pd.DataFrame({"queda_de_auc": result.importances_mean, "desvio_padrao": result.importances_std},
                         index=X_test.columns)
    return table.sort_values("queda_de_auc", ascending=False).head(top).round(4)


def plot_tree(tree_pipeline, feature_names):
    """Drawing of the pruned decision tree."""
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree as sk_plot_tree

    tree = tree_pipeline.named_steps["model"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    boxes = sk_plot_tree(tree, feature_names=list(feature_names), class_names=["sem evento", "evento"],
                         filled=True, impurity=False, proportion=True, rounded=True, fontsize=5, ax=ax)
    # scikit-learn writes the box labels in English
    labels = {"samples = ": "internações = ", "value = ": "proporção = ", "class = ": "classe = "}
    for box in boxes:
        text = box.get_text()
        for english, portuguese in labels.items():
            text = text.replace(english, portuguese)
        box.set_text(text)
    for text in ax.texts:
        if text.get_text().strip() in ("True", "False"):
            text.set_text(text.get_text().replace("True", "Sim").replace("False", "Não"))
    return fig


# ---------------------------------------------------------------------------
# Clinical use: the model would flag patients for closer follow-up, so a
# missed event (false negative) costs more than an unnecessary follow-up.
# ---------------------------------------------------------------------------

def clinical_threshold(model, X_train, y_train, target_sensitivity=0.90, n_splits=10):
    """Threshold that reaches `target_sensitivity` on the training set, using
    out-of-fold predicted probabilities (the test set is not used)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    return float(np.quantile(scores[np.asarray(y_train) == 1], 1 - target_sensitivity))


def threshold_comparison(model, X_test, y_test, thresholds):
    """Test-set consequences of each decision threshold, next to the
    'flag everyone' strategy. `thresholds` maps a label to a cut-off."""
    y_true = np.asarray(y_test)
    scores = model.predict_proba(X_test)[:, 1]
    rows = {}
    for label, threshold in {**thresholds, "Sinalizar todos": 0.0}.items():
        flagged = scores >= threshold
        tp = int((flagged & (y_true == 1)).sum()); fn = int((~flagged & (y_true == 1)).sum())
        tn = int((~flagged & (y_true == 0)).sum()); fp = int((flagged & (y_true == 0)).sum())
        rows[label] = {
            "limiar": threshold,
            "sensibilidade": tp / (tp + fn),
            "especificidade": tn / (tn + fp),
            "sinalizados": tp + fp,
            "eventos_perdidos": fn,
            "nao_sinalizados": tn + fn,
            "sem_evento_entre_nao_sinalizados": tn / (tn + fn) if tn + fn else np.nan,
        }
    return pd.DataFrame(rows).T.infer_objects().round(3)


def confusion_at(model, X_test, y_test, threshold):
    """Confusion matrix at a given probability threshold."""
    predicted = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    return pd.DataFrame(matrix, index=["Real: sem evento", "Real: evento"],
                        columns=["Previsto: sem evento", "Previsto: evento"])


def undo_class_weight(probabilities, train_prevalence):
    """Probabilities as if the model had been fitted without class weights.

    class_weight="balanced" multiplies the odds of the event by
    (1 - prevalence) / prevalence; this divides them back. Exact for the
    logistic regression, approximate for the trees."""
    ratio = train_prevalence / (1 - train_prevalence)
    return probabilities * ratio / (probabilities * ratio + 1 - probabilities)


def decision_curve(models, X_test, y_test, train_prevalence, thresholds=np.linspace(0.05, 0.60, 56)):
    """Net benefit (Vickers & Elkin, 2006) of each model and of flagging
    everyone, for each threshold probability p_t:

        net benefit = TP/n - FP/n * p_t / (1 - p_t)

    p_t expresses how much worse a missed event is than an unnecessary
    follow-up: p_t = 0.25 means a missed event counts as 3 unnecessary
    follow-ups. Flagging nobody has net benefit 0."""
    y_true = np.asarray(y_test)
    n, prevalence = len(y_true), y_true.mean()
    curves = {"Sinalizar todos": prevalence - (1 - prevalence) * thresholds / (1 - thresholds)}
    for name, model in models.items():
        p = undo_class_weight(model.predict_proba(X_test)[:, 1], train_prevalence)
        benefit = []
        for t in thresholds:
            flagged = p >= t
            tp = (flagged & (y_true == 1)).sum(); fp = (flagged & (y_true == 0)).sum()
            benefit.append(tp / n - fp / n * t / (1 - t))
        curves[name] = np.array(benefit)
    return pd.DataFrame(curves, index=pd.Index(thresholds.round(3), name="limiar_de_probabilidade"))


def plot_decision_curve(curves):
    """Net benefit against the threshold probability; the grey line at zero
    is 'flag nobody'."""
    import matplotlib.pyplot as plt
    from src.utils import MUTED_COLOR, NEUTRAL_COLOR

    colors = {"Sinalizar todos": MUTED_COLOR, "Árvore de decisão": NEUTRAL_COLOR,
              "Regressão logística Lasso": "#2a78d6", "Random Forest": "#d6672a"}
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", label="Não sinalizar ninguém")
    for name in curves.columns:
        style = ":" if name == "Sinalizar todos" else "-"
        ax.plot(curves.index, curves[name], style, color=colors.get(name), linewidth=1.2, label=name)
    ax.set_ylim(-0.02, curves.max().max() + 0.02)
    ax.set_xlabel("Limiar de probabilidade")
    ax.set_ylabel("Benefício líquido")
    ax.legend(fontsize=5.5, frameon=False, loc="upper right")
    return fig


def plot_confusion_matrices(model, X_test, y_test, thresholds):
    """Confusion matrices side by side, one per decision threshold
    (`thresholds` maps a panel label to a cut-off)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    scores = model.predict_proba(X_test)[:, 1]
    matrices = {label: confusion_matrix(y_test, (scores >= t).astype(int), labels=[0, 1])
                for label, t in thresholds.items()}
    vmax = max(m.max() for m in matrices.values())  # same color scale in every panel
    fig, axes = plt.subplots(1, len(thresholds), figsize=(3.5, 1.9), sharey=True)
    for ax, (label, matrix) in zip(np.atleast_1d(axes), matrices.items()):
        ConfusionMatrixDisplay(matrix, display_labels=["Sem evento", "Evento"]).plot(
            ax=ax, cmap="Blues", values_format="d", colorbar=False, text_kw={"fontsize": 7},
            im_kw={"vmin": 0, "vmax": vmax})
        ax.set_title(label, fontsize=7)
        ax.set_xlabel("Classe prevista", fontsize=7)
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Classe verdadeira", fontsize=7)
    fig.tight_layout()
    return fig
