"""
Feature engineering and feature selection (step 2).

- build_model_matrix: deterministic construction of the numeric predictor
  matrix (leakage removal, drug indicators, missing-data treatment, encoding).
- unsupervised_filter, lasso_*, forest_*, select_features: feature selection,
  always fitted on training data only.
"""

import pandas as pd

def remove_leakage(patients_df, variable_groups):
    """
    Remove as colunas que causam vazamento de informação (data leakage),
    como identificadores, desfechos e informações obtidas apenas após a alta.
    
    Parâmetros:
        patients_df (pd.DataFrame): DataFrame original de pacientes.
        variable_groups (dict): Dicionário com os grupos de variáveis gerado no preprocessing.
        
    Retorna:
        pd.DataFrame: DataFrame sem as colunas de vazamento.
    """
    leakage_columns = (
        variable_groups["Desfechos e seguimento"]
        + variable_groups["Informações da alta"]
        + variable_groups["Identificação"]
    )
    
    # Remove as colunas que existem no DataFrame para evitar erros
    cols_to_drop = [col for col in leakage_columns if col in patients_df.columns]
    
    features_df = patients_df.drop(columns=cols_to_drop)
    print(f"Removidas {len(cols_to_drop)} colunas de vazamento. Restam {features_df.shape[1]} atributos clínicos.")
    
    return features_df


def process_drugs(drugs_df):
    """
    Transforma a base de medicamentos do formato longo (long) para o formato largo (wide).
    Cada medicamento torna-se uma coluna binária (1 = usou, 0 = não usou).
    
    Parâmetros:
        drugs_df (pd.DataFrame): DataFrame de medicamentos (dat_md).
        
    Retorna:
        pd.DataFrame: DataFrame indexado por inpatient.number com variáveis binárias para cada fármaco.
    """
    # Cria uma tabela de contingência: linhas = internações, colunas = medicamentos
    drugs_wide = pd.crosstab(
        index=drugs_df["inpatient.number"], 
        columns=drugs_df["Drug_name"]
    )
    
    # Garante que os valores são estritamente binários (caso haja duplicados no original)
    drugs_wide = (drugs_wide > 0).astype(int)
    
    # Adiciona um prefixo para facilitar a identificação dos atributos
    drugs_wide.columns = [f"drug_{col}" for col in drugs_wide.columns]
    
    print(f"Base de medicamentos processada: {drugs_wide.shape[1]} fármacos extraídos.")
    return drugs_wide


# =====================================================================
# Step 2 (notebook 02): deterministic construction of the model matrix.
# Nothing here is learned from the target, so it is safe to run before
# the train/test split. Imputation and scaling are NOT done here: they
# are fitted on the training folds inside the sklearn Pipeline.
# =====================================================================

# Two-level text variables -> single 0/1 column named after the level coded as 1
BINARY_TEXT_MAP = {
    "gender": ("gender_Male", "Male"),
    "admission.way": ("admission.way_Emergency", "Emergency"),
    "type.II.respiratory.failure": ("type.II.respiratory.failure", "TypeII"),
    "oxygen.inhalation": ("oxygen.inhalation_OxygenTherapy", "OxygenTherapy"),
}

# Rare levels merged before one-hot encoding (level -> merged level)
RARE_LEVEL_MAP = {
    "admission.ward": {"ICU": "Others"},
    "occupation": {"worker": "Others", "Officer": "Others"},
    "ageCat": {"(21,29]": "(21,49]", "(29,39]": "(21,49]", "(39,49]": "(21,49]"},
}

# One-hot variables and the reference level left out (the most frequent one),
# which avoids the dummy-variable trap in unpenalized linear models
ONE_HOT_REFERENCE = {
    "admission.ward": "Cardiology",
    "occupation": "UrbanResident",
    "type.of.heart.failure": "Both",
    "NYHA.cardiac.function.classification": "III",
    "Killip.grade": "II",
    "ageCat": "(69,79]",
}

# Consciousness is the same information as GCS, and GCS is the exact sum
# of its three components (perfect collinearity): keep only GCS
REDUNDANT_WITH_GCS = ["consciousness", "eye.opening", "verbal.response", "movement"]

STRUCTURAL_MISSING_COLUMN = "respiratory.support."


def encode_structural_missing(df):
    """NA in respiratory.support. means "no mechanical ventilation" (not lost
    data). IMV (25 cases) and NIMV (17) are merged into one 0/1 column."""
    out = df.copy()
    out["respiratory.support"] = out[STRUCTURAL_MISSING_COLUMN].notna().astype(int)
    return out.drop(columns=[STRUCTURAL_MISSING_COLUMN])


def encode_categoricals(df):
    """Turn every text column into numbers: two-level variables become one
    0/1 column; variables with more levels are one-hot encoded (rare levels
    merged first, most frequent level dropped as reference).

    A missing level (only `occupation`, 1.3%) yields zeros in all dummies,
    i.e. it falls in the reference level, which is also the mode."""
    out = df.drop(columns=[c for c in REDUNDANT_WITH_GCS if c in df.columns])

    for col, (new_name, positive_level) in BINARY_TEXT_MAP.items():
        encoded = (out[col] == positive_level).astype(int)
        out = out.drop(columns=col)
        out[new_name] = encoded

    for col, reference in ONE_HOT_REFERENCE.items():
        levels = out[col].replace(RARE_LEVEL_MAP.get(col, {}))
        dummies = pd.get_dummies(levels, prefix=col, dtype=int)
        dummies = dummies.drop(columns=f"{col}_{reference}")
        position = out.columns.get_loc(col)
        out = pd.concat([out.iloc[:, :position], dummies, out.iloc[:, position + 1:]], axis=1)

    return out


def add_panel_indicators(df, threshold=0.20, min_agreement=0.98):
    """Add one "exam not ordered" 0/1 column per exam panel.

    Variables missing in more than `threshold` of the admissions are treated
    as "exam not ordered" (informative missingness). Variables whose missing
    masks agree in at least `min_agreement` of the rows belong to the same
    panel and share a single indicator (1 = no variable of the panel measured).
    The measured values are kept; they are imputed later, inside the Pipeline.

    Returns the new DataFrame and a table describing each panel."""
    out = df.copy()
    missing_rate = out.isna().mean()
    candidates = [c for c in out.columns if threshold < missing_rate[c] < 1.0]
    masks = out[candidates].isna()

    panels = []  # each panel is a list of columns
    for col in candidates:
        for panel in panels:
            if (masks[col] == masks[panel[0]]).mean() >= min_agreement:
                panel.append(col)
                break
        else:
            panels.append([col])

    rows = []
    for panel in panels:
        suffix = "_not_ordered" if len(panel) == 1 else "_panel_not_ordered"
        name = f"{panel[0]}{suffix}"
        out[name] = masks[panel].all(axis=1).astype(int)
        rows.append({
            "indicator": name,
            "n_variables": len(panel),
            "not_ordered_pct": round(out[name].mean() * 100, 1),
            "variables": ", ".join(panel),
        })

    return out, pd.DataFrame(rows)


def build_model_matrix(patients_df, drugs_df, variable_groups):
    """Build the numeric predictor matrix used from step 2 onwards.

    Steps (all deterministic, none uses the target):
      1. join the drug indicators and remove leakage columns;
      2. drop columns that are entirely empty;
      3. structural missingness -> category (respiratory support);
      4. informative missingness -> one indicator per exam panel;
      5. encode qualitative variables.
    Sporadic missing values (<= 20%) are left as NaN for the Pipeline imputer.

    Returns (X, panel_table), with X indexed by inpatient.number."""
    drugs_wide = process_drugs(drugs_df)
    X = patients_df.set_index("inpatient.number").join(drugs_wide, how="left")
    drug_cols = [c for c in X.columns if c.startswith("drug_")]
    X[drug_cols] = X[drug_cols].fillna(0).astype(int)

    X = remove_leakage(X, variable_groups)

    empty_cols = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=empty_cols)
    print(f"Removidas {len(empty_cols)} colunas totalmente vazias: {empty_cols}")

    X = encode_structural_missing(X)
    X, panel_table = add_panel_indicators(X)
    print(f"Adicionados {len(panel_table)} indicadores de exame não pedido.")
    X = encode_categoricals(X)

    return X, panel_table


# =====================================================================
# Step 2, feature selection. Everything below must receive TRAINING data
# only.
# =====================================================================

def unsupervised_filter(X_train, min_prevalence=0.01, max_correlation=0.90, min_pairs=100):
    """First selection stage: remove columns without looking at the target.

    1. Constant columns.
    2. Near-constant binary columns (minority value in less than
       `min_prevalence` of the rows).
    3. Redundant columns: for every pair with |Spearman| > `max_correlation`
       (strongest pairs first), drop the one with more missing values; ties
       drop the column that comes later in the matrix.

    Returns (kept_columns, report), where report has one row per dropped
    column with the reason."""
    report = []
    n_unique = X_train.nunique(dropna=True)

    constant = n_unique[n_unique <= 1].index.tolist()
    report += [{"dropped": c, "reason": "constante", "kept_instead": "", "abs_spearman": None} for c in constant]

    binary = [c for c in X_train.columns if n_unique[c] == 2]
    minority = X_train[binary].apply(lambda s: s.value_counts(normalize=True).min())
    rare = minority[minority < min_prevalence].index.tolist()
    report += [{"dropped": c, "reason": f"quase constante (<{min_prevalence:.0%})", "kept_instead": "", "abs_spearman": None} for c in rare]

    remaining = [c for c in X_train.columns if c not in constant and c not in rare]
    corr = X_train[remaining].corr(method="spearman", min_periods=min_pairs).abs()
    missing_rate = X_train[remaining].isna().mean()
    position = {c: i for i, c in enumerate(remaining)}

    pairs = [
        (corr.iat[i, j], remaining[i], remaining[j])
        for i in range(len(remaining)) for j in range(i + 1, len(remaining))
        if corr.iat[i, j] > max_correlation
    ]
    dropped = set()
    for rho, a, b in sorted(pairs, key=lambda p: -p[0]):
        if a in dropped or b in dropped:
            continue
        # keep the column with fewer missing values (tie: the earlier one)
        keep, drop = sorted([a, b], key=lambda c: (missing_rate[c], position[c]))
        dropped.add(drop)
        report.append({"dropped": drop, "reason": f"redundante (|rho|>{max_correlation})", "kept_instead": keep, "abs_spearman": round(float(rho), 3)})

    kept = [c for c in remaining if c not in dropped]

    # A column kept in one pair may be dropped later in another pair: follow
    # the chain so that `kept_instead` always names a column that survived
    replaced_by = {r["dropped"]: r["kept_instead"] for r in report if r["kept_instead"]}
    for r in report:
        while r["kept_instead"] in replaced_by:
            r["kept_instead"] = replaced_by[r["kept_instead"]]

    return kept, pd.DataFrame(report)


def make_lasso_pipeline(C=1.0):
    """Median imputation -> standardization -> L1-penalized logistic regression.

    Standardization is required because the L1 penalty acts on the size of the
    coefficients, which depends on the scale of each predictor (Aula 07)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src.utils import RANDOM_STATE

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lasso", LogisticRegression(l1_ratio=1.0, C=C, solver="liblinear", random_state=RANDOM_STATE)),
    ])


def lasso_path_cv(X_train, y_train, Cs, n_splits=10):
    """Cross-validated AUC of the Lasso logistic regression for each value of C
    (C = 1/lambda: smaller C means stronger penalty and fewer features).

    The whole pipeline is refitted in each training fold, so imputation,
    scaling and selection never see the validation fold.

    Returns a table with one row per C: mean AUC, its standard error across
    folds and the mean number of non-zero coefficients."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from src.utils import RANDOM_STATE

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for C in Cs:
        aucs, sizes = [], []
        for train_idx, valid_idx in folds.split(X_train, y_train):
            model = clone(make_lasso_pipeline(C)).fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            scores = model.predict_proba(X_train.iloc[valid_idx])[:, 1]
            aucs.append(roc_auc_score(y_train.iloc[valid_idx], scores))
            sizes.append(int((model.named_steps["lasso"].coef_ != 0).sum()))
        rows.append({
            "C": C,
            "auc_mean": np.mean(aucs),
            "auc_se": np.std(aucs, ddof=1) / np.sqrt(n_splits),
            "n_features_mean": np.mean(sizes),
        })
    return pd.DataFrame(rows)


def pick_one_standard_error(path):
    """One-standard-error rule (Aula 07): the simplest model (smallest C) whose
    AUC is within one standard error of the best AUC.

    Returns (C_best, C_one_se)."""
    best = path.loc[path["auc_mean"].idxmax()]
    threshold = best["auc_mean"] - best["auc_se"]
    C_one_se = path.loc[path["auc_mean"] >= threshold, "C"].min()
    return float(best["C"]), float(C_one_se)


def lasso_selected_features(X_train, y_train, C):
    """Fit the Lasso pipeline on the whole training set and return the features
    with non-zero coefficient (standardized scale), sorted by absolute size."""
    import numpy as np

    model = make_lasso_pipeline(C).fit(X_train, y_train)
    coef = pd.Series(model.named_steps["lasso"].coef_[0], index=X_train.columns, name="coefficient")
    coef = coef[coef != 0]
    table = coef.reindex(coef.abs().sort_values(ascending=False).index).to_frame()
    table["odds_ratio_per_sd"] = np.exp(table["coefficient"])
    return table.round(4)


def plot_lasso_path(path, C_best, C_one_se):
    """AUC (with +-1 standard error) and model size as a function of C."""
    import matplotlib.pyplot as plt
    from src.utils import SERIES_COLOR, MUTED_COLOR, NEUTRAL_COLOR

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.errorbar(path["C"], path["auc_mean"], yerr=path["auc_se"], color=SERIES_COLOR,
                marker="o", markersize=2.5, linewidth=1, elinewidth=0.6, capsize=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("C (inverso da penalidade L1)")
    ax.set_ylabel("AUC na validação cruzada", color=SERIES_COLOR)
    ax.axvline(C_best, color=NEUTRAL_COLOR, linestyle=":", linewidth=1, label="maior AUC")
    ax.axvline(C_one_se, color=MUTED_COLOR, linestyle="--", linewidth=1, label="regra de 1 erro padrão")
    ax.legend(loc="lower right", fontsize=6, frameon=False)

    size_ax = ax.twinx()
    size_ax.plot(path["C"], path["n_features_mean"], color=MUTED_COLOR, linewidth=1)
    size_ax.set_ylabel("Nº de atributos selecionados", color=MUTED_COLOR)
    size_ax.grid(False)
    return fig


def lasso_selection_frequency(X_train, y_train, C, n_resamples=200):
    """Stability of the Lasso selection: refit the pipeline on bootstrap
    resamples of the training set (Aula 05) and count how often each feature
    gets a non-zero coefficient.

    Returns a table indexed by feature with the selection frequency and the
    share of selections in which the coefficient was positive."""
    import numpy as np
    from src.utils import RANDOM_STATE

    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X_train)
    selected = np.zeros(X_train.shape[1])
    positive = np.zeros(X_train.shape[1])
    for _ in range(n_resamples):
        rows = rng.integers(0, n, n)
        model = make_lasso_pipeline(C).fit(X_train.iloc[rows], y_train.iloc[rows])
        coef = model.named_steps["lasso"].coef_[0]
        selected += coef != 0
        positive += coef > 0

    table = pd.DataFrame({
        "selection_frequency": selected / n_resamples,
        "positive_share": np.divide(positive, selected, out=np.full_like(positive, np.nan), where=selected > 0),
    }, index=X_train.columns)
    return table.sort_values("selection_frequency", ascending=False).round(3)


def make_forest_pipeline():
    """Median imputation -> Random Forest (no scaling needed for trees)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from src.utils import RANDOM_STATE

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE)),
    ])


def forest_permutation_importance(X_train, y_train, n_splits=5, n_repeats=5):
    """Random Forest importance measured by permutation on held-out folds:
    how much the validation AUC drops when one feature is shuffled.

    Unlike the impurity-based importance, it is not biased towards continuous
    features and is measured on data the forest did not see.

    Returns (importance table, cross-validated AUC of the forest per fold)."""
    import numpy as np
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from src.utils import RANDOM_STATE

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    drops, aucs = [], []
    for train_idx, valid_idx in folds.split(X_train, y_train):
        model = make_forest_pipeline().fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        X_valid, y_valid = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        aucs.append(roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]))
        result = permutation_importance(model, X_valid, y_valid, scoring="roc_auc",
                                        n_repeats=n_repeats, random_state=RANDOM_STATE)
        drops.append(result.importances_mean)

    drops = np.array(drops)
    table = pd.DataFrame({
        "auc_drop_mean": drops.mean(axis=0),
        "auc_drop_se": drops.std(axis=0, ddof=1) / np.sqrt(n_splits),
    }, index=X_train.columns)
    table = table.sort_values("auc_drop_mean", ascending=False)
    table["forest_rank"] = range(1, len(table) + 1)
    return table.round(4), aucs


def compare_feature_sets_cv(X_train, y_train, C, n_splits=10):
    """Compare feature sets by cross-validated AUC, for a Random Forest and an
    unpenalized logistic regression:

      - "todas": all columns (no supervised selection);
      - "Lasso": the columns the Lasso selects (embedded method);
      - "filtro F": the same number of columns, taken from the top of the
        univariate ANOVA F-test ranking (filter method).

    Both selections are redone inside every training fold, so the comparison
    is not optimistic."""
    import numpy as np
    from sklearn.feature_selection import f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from src.utils import RANDOM_STATE

    def make_logistic():
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=np.inf, max_iter=5000)),
        ])

    models = [("Random Forest", make_forest_pipeline), ("Logística sem penalidade", make_logistic)]
    set_names = ["todas", "Lasso", "filtro F"]
    scores = {(model, name): [] for model, _ in models for name in set_names}
    sizes = []

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, valid_idx in folds.split(X_train, y_train):
        X_fit, y_fit = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_valid, y_valid = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

        lasso = make_lasso_pipeline(C).fit(X_fit, y_fit)
        lasso_cols = X_train.columns[lasso.named_steps["lasso"].coef_[0] != 0]
        sizes.append(len(lasso_cols))

        f_values, _ = f_classif(SimpleImputer(strategy="median").fit_transform(X_fit), y_fit)
        filter_cols = X_train.columns[np.argsort(-np.nan_to_num(f_values))[:len(lasso_cols)]]

        column_sets = {"todas": X_train.columns, "Lasso": lasso_cols, "filtro F": filter_cols}
        for model_name, make_model in models:
            for set_name, columns in column_sets.items():
                model = make_model().fit(X_fit[columns], y_fit)
                auc = roc_auc_score(y_valid, model.predict_proba(X_valid[columns])[:, 1])
                scores[(model_name, set_name)].append(auc)

    table = pd.DataFrame({
        "auc_mean": {k: np.mean(v) for k, v in scores.items()},
        "auc_se": {k: np.std(v, ddof=1) / np.sqrt(n_splits) for k, v in scores.items()},
    }).round(4)
    table.index = pd.MultiIndex.from_tuples(table.index, names=["model", "feature_set"])
    return table, float(np.mean(sizes))


def plot_selection_frequency(frequency, selected_features, min_frequency=0.40):
    """Horizontal bars with the bootstrap selection frequency of each feature
    (only those above `min_frequency`); features in the Lasso set are highlighted."""
    import matplotlib.pyplot as plt
    from src.utils import SERIES_COLOR, NEUTRAL_COLOR, MUTED_COLOR

    data = frequency.loc[frequency["selection_frequency"] >= min_frequency, "selection_frequency"].sort_values()
    colors = [SERIES_COLOR if name in selected_features else NEUTRAL_COLOR for name in data.index]

    fig, ax = plt.subplots(figsize=(3.5, 0.17 * len(data) + 0.6))
    ax.barh(data.index, data.values * 100, color=colors, height=0.7)
    ax.axvline(50, color=MUTED_COLOR, linestyle="--", linewidth=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frequência de seleção nas reamostragens (%)")
    ax.tick_params(axis="y", labelsize=5.5)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.01)
    return fig


def information_criteria(X_train, y_train, feature_sets):
    """AIC and BIC of an unpenalized logistic regression fitted on each
    candidate feature set (Aulas 06 and 07): both add to the training
    deviance a penalty that grows with the number of coefficients, so a larger
    model is only preferred if its fit improves enough. Lower is better.

    `feature_sets` maps a label to a list of columns."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = len(y_train)
    rows = {}
    for label, columns in feature_sets.items():
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=np.inf, max_iter=10000))
        model.fit(X_train[columns], y_train)
        deviance = 2 * log_loss(y_train, model.predict_proba(X_train[columns])[:, 1], normalize=False)
        k = len(columns) + 1  # coefficients + intercept
        rows[label] = {"n_features": len(columns), "deviance": deviance,
                       "AIC": deviance + 2 * k, "BIC": deviance + np.log(n) * k}
    return pd.DataFrame(rows).T.round(1)


SELECTED_FEATURES_FILE = "selected_features.csv"


def save_selected_features(feature_names):
    """Persist the final feature list (names only, no patient data)."""
    from src.utils import RESULTS_DIR
    pd.Series(feature_names, name="feature").to_csv(RESULTS_DIR / SELECTED_FEATURES_FILE, index=False)


def load_selected_features():
    """Final feature list chosen in notebook 02, for the following notebooks."""
    from src.utils import RESULTS_DIR
    return pd.read_csv(RESULTS_DIR / SELECTED_FEATURES_FILE)["feature"].tolist()


LASSO_C_GRID = tuple(float(c) for c in __import__("numpy").logspace(-2.5, 0, 26))
MIN_SELECTION_FREQUENCY = 0.50


def select_features(X_train, y_train):
    """The whole selection procedure of notebook 02 in one call, so that it can
    be repeated on any training set: unsupervised filter -> Lasso path in CV ->
    one-standard-error rule -> keep the features selected in at least 50% of
    the bootstrap resamples.

    Returns (final feature list, chosen C)."""
    kept, _ = unsupervised_filter(X_train)
    X_filtered = X_train[kept]
    path = lasso_path_cv(X_filtered, y_train, LASSO_C_GRID)
    _, C_one_se = pick_one_standard_error(path)
    selected = lasso_selected_features(X_filtered, y_train, C_one_se).index.tolist()
    frequency = lasso_selection_frequency(X_filtered, y_train, C_one_se)["selection_frequency"]
    final = [f for f in selected if frequency[f] >= MIN_SELECTION_FREQUENCY]
    return final, C_one_se
