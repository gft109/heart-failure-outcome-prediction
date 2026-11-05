# """
# Módulo de Análise Exploratória de Dados (EDA).
# Responsável por gerar visualizações e resumos estatísticos.
# """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import Markdown, display
from src.utils import SERIES_COLOR, NEUTRAL_COLOR, NEUTRAL_LIGHT_COLOR, MUTED_COLOR, RESULTS_DIR, save_fig

def variable_kind(series):
    """Determina o tipo da variável clínica."""
    n_unique = series.nunique()
    if n_unique == 2:
        return "binária"
    if pd.api.types.is_numeric_dtype(series):
        return "numérica" if n_unique > 10 else "discreta"
    return "categórica"

def observed_values(series):
    """Formata os valores observados para o relatório."""
    values = series.dropna()
    if pd.api.types.is_numeric_dtype(series) and values.nunique() > 10:
        return f"{values.min():g} – {values.median():g} – {values.max():g}"
    return ", ".join(str(v) for v in sorted(values.unique(), key=str))

def generate_variable_overview(patients, data_dictionary, variable_groups):
    """
    Gera e guarda o resumo geral (Tabela 1) das variáveis, incluindo
    tipo, percentagem de dados em falta e valores observados.
    """
    column_group = {column: group for group, columns in variable_groups.items() for column in columns}
    descriptions = data_dictionary.set_index("Variables")["Description"]
    
    variable_overview = pd.DataFrame(
        {
            "grupo": [column_group.get(c, "Desconhecido") for c in patients.columns],
            "variável": patients.columns,
            "descrição": descriptions.reindex(patients.columns).values,
            "tipo": [variable_kind(patients[c]) for c in patients.columns],
            "faltantes (%)": patients.isna().mean().mul(100).round(1).values,
            "valores observados": [observed_values(patients[c]) for c in patients.columns],
        }
    )
    
    variable_overview["grupo"] = pd.Categorical(variable_overview["grupo"], categories=list(variable_groups), ordered=True)
    variable_overview = variable_overview.sort_values("grupo", kind="stable").reset_index(drop=True)
    variable_overview.to_csv(RESULTS_DIR / "variable_overview.csv", index=False)
    
    summary = variable_overview.groupby("grupo", observed=True).agg(
        n_variáveis=("variável", "size"),
        faltantes_médio=("faltantes (%)", "mean"),
        faltantes_máx=("faltantes (%)", "max"),
    ).round(1)
    
    print("Resumo por Grupo Clínico:")
    display(summary)
    return variable_overview

def display_variable_overview(variable_overview):
    """Exibe a tabela geral formatada por grupo."""
    with pd.option_context("display.max_colwidth", None):
        for group, table in variable_overview.groupby("grupo", observed=True):
            display(Markdown(f"#### {group}"))
            display(table.drop(columns="grupo").set_index("variável"))

def plot_missing_matrix(patients, variable_groups):
    """
    Gera a matriz visual de dados faltantes (Missing Matrix).
    """
    column_group = {column: group for group, columns in variable_groups.items() for column in columns}
    missing_rate = patients.isna().mean().mul(100)
    
    missing_columns = [c for c in patients.columns if missing_rate[c] > 0]
    missing_columns = sorted(missing_columns, key=lambda c: list(variable_groups).index(column_group[c]))
    
    missing_matrix = patients[missing_columns].isna()
    missing_matrix = missing_matrix.loc[missing_matrix.sum(axis=1).sort_values(kind="stable").index]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(missing_matrix.values, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([NEUTRAL_LIGHT_COLOR, SERIES_COLOR]))
    
    group_of_column = pd.Series([column_group[c] for c in missing_columns])
    boundaries = np.flatnonzero(group_of_column.ne(group_of_column.shift())).tolist() + [len(missing_columns)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start > 0:
            ax.axvline(start - 0.5, color="white", linewidth=1.5)
    ax.set_xticks([(start + end - 1) / 2 for start, end in zip(boundaries[:-1], boundaries[1:])])
    ax.set_xticklabels(group_of_column.iloc[np.asarray(boundaries[:-1], dtype=int)], rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_ylabel("Internações (mais completa no topo)")
    ax.grid(False)
    save_fig(fig, "missing_matrix")
    plt.show()
