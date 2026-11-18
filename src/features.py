"""
Step 2: data treatment.

Turns the raw tables into the numeric matrix used by the three models.
Every step is deterministic and none of them uses the target, so the matrix
can be built before the train/test split. What is learned from the data
(imputation of sporadic missing values, scaling) is fitted later, on the
training folds, inside each model's Pipeline (src/train.py).
"""

import pandas as pd

from src.preprocessing import create_target

# Two-level text variables -> one 0/1 column named after the level coded as 1
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

# One-hot variables and the reference level left out (the most frequent one)
ONE_HOT_REFERENCE = {
    "admission.ward": "Cardiology",
    "occupation": "UrbanResident",
    "type.of.heart.failure": "Both",
    "NYHA.cardiac.function.classification": "III",
    "Killip.grade": "II",
    "ageCat": "(69,79]",
}

# GCS is the exact sum of its three components and consciousness repeats the
# same information: keep only GCS
REDUNDANT_WITH_GCS = ["consciousness", "eye.opening", "verbal.response", "movement"]

STRUCTURAL_MISSING_COLUMN = "respiratory.support."

# A variable missing in more than this share of admissions is treated as
# "exam not ordered"
NOT_ORDERED_THRESHOLD = 0.20


def leakage_columns(variable_groups):
    """Columns known only after discharge (outcomes, discharge data) plus the
    identifier. They never enter the matrix; the outcomes are used only to
    build the target."""
    return (variable_groups["Desfechos e seguimento"]
            + variable_groups["Informações da alta"]
            + variable_groups["Identificação"])


def drug_indicators(drugs_df):
    """Drug table from long to wide format: one 0/1 column per drug."""
    wide = (pd.crosstab(drugs_df["inpatient.number"], drugs_df["Drug_name"]) > 0).astype(int)
    wide.columns = [f"drug_{name}" for name in wide.columns]
    return wide


def add_panel_indicators(df, threshold=NOT_ORDERED_THRESHOLD, min_agreement=0.98):
    """One "exam not ordered" 0/1 column per exam panel.

    Variables missing in more than `threshold` of the admissions whose missing
    patterns agree in at least `min_agreement` of the rows belong to the same
    panel and share one indicator. The measured values are kept.

    Returns the new DataFrame and a table describing each panel."""
    missing_rate = df.isna().mean()
    candidates = [c for c in df.columns if threshold < missing_rate[c] < 1.0]
    masks = df[candidates].isna()

    panels = []
    for col in candidates:
        for panel in panels:
            if (masks[col] == masks[panel[0]]).mean() >= min_agreement:
                panel.append(col)
                break
        else:
            panels.append([col])

    indicators, rows = {}, []
    for panel in panels:
        name = f"{panel[0]}_not_ordered" if len(panel) == 1 else f"{panel[0]}_panel_not_ordered"
        indicators[name] = masks[panel].all(axis=1).astype(int)
        rows.append({"indicador": name, "n_variaveis": len(panel),
                     "nao_pedido_pct": round(indicators[name].mean() * 100, 1),
                     "variaveis": ", ".join(panel)})
    return pd.concat([df, pd.DataFrame(indicators)], axis=1), pd.DataFrame(rows)


def encode_categoricals(df):
    """Two-level text variables become one 0/1 column; variables with more
    levels are one-hot encoded (rare levels merged, most frequent level left
    out as reference). Missing occupation (1.3%) falls in the reference level,
    which is also the mode."""
    out = df.copy()
    for col, (new_name, positive_level) in BINARY_TEXT_MAP.items():
        out[col] = (out[col] == positive_level).astype(int)
        out = out.rename(columns={col: new_name})

    for col, reference in ONE_HOT_REFERENCE.items():
        levels = out[col].replace(RARE_LEVEL_MAP.get(col, {}))
        dummies = pd.get_dummies(levels, prefix=col, dtype=int).drop(columns=f"{col}_{reference}")
        position = out.columns.get_loc(col)
        out = pd.concat([out.iloc[:, :position], dummies, out.iloc[:, position + 1:]], axis=1)
    return out


def build_dataset(patients_df, drugs_df, variable_groups):
    """Target and numeric predictor matrix, both indexed by inpatient.number.

    1. Target: death or readmission within 6 months.
    2. Leakage: outcomes, discharge data and identifier removed.
    3. Missing values:
       - structural (respiratory support: empty = no ventilation) -> 0/1 column;
       - exam not ordered (> 20% missing) -> one indicator per panel;
       - sporadic (<= 20%) -> left as NaN, imputed later inside the Pipeline.
    4. Qualitative variables encoded (0/1 and one-hot).
    5. Empty, constant and redundant columns removed.

    Returns (X, y, report); report holds the tables shown in notebook 02."""
    y = create_target(patients_df)
    y.index = patients_df["inpatient.number"].values

    clinical = patients_df.set_index("inpatient.number")
    drugs = drug_indicators(drugs_df).reindex(clinical.index, fill_value=0)  # no record = not used
    X = pd.concat([clinical, drugs], axis=1)

    removed =[{"coluna": c, "motivo": "vazamento"} for c in leakage_columns(variable_groups) if c in X.columns]
    X = X.drop(columns=[r["coluna"] for r in removed])

    X[STRUCTURAL_MISSING_COLUMN] = X[STRUCTURAL_MISSING_COLUMN].notna().astype(int)
    X = X.rename(columns={STRUCTURAL_MISSING_COLUMN: "respiratory.support"})

    empty = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=empty)
    removed += [{"coluna": c, "motivo": "vazia"} for c in empty]

    X, panels = add_panel_indicators(X)

    X = X.drop(columns=REDUNDANT_WITH_GCS)
    removed += [{"coluna": c, "motivo": "redundante com GCS"} for c in REDUNDANT_WITH_GCS]

    X = encode_categoricals(X)

    constant = X.columns[X.nunique(dropna=True) <= 1].tolist()
    X = X.drop(columns=constant)
    removed += [{"coluna": c, "motivo": "constante"} for c in constant]

    report = {"removed": pd.DataFrame(removed), "panels": panels}
    return X, y, report
