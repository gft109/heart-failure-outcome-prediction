"""
Módulo de utilidades e configurações globais para o projeto.
Centraliza paths, configurações visuais e sementes aleatórias.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil

# ==========================================
# 1. Configurações Globais e Reprodutibilidade
# ==========================================
RANDOM_STATE = 42

# Define a raiz do projeto dinamicamente (um nível acima da pasta src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Mapeamento de diretórios
DATA_DIR = BASE_DIR / "DataBase"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Garante que as pastas de saída existam no momento da importação
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Identidade Visual (Cores)
# ==========================================
SERIES_COLOR = "#2a78d6"
NEUTRAL_COLOR = "#a3a29c"
NEUTRAL_LIGHT_COLOR = "#e8e7e3"
MUTED_COLOR = "#52514e"

# ==========================================
# 3. Funções Utilitárias
# ==========================================
def setup_plotting():
    """
    Configura os parâmetros globais de exibição do Pandas e 
    o tema visual do Seaborn/Matplotlib para o projeto.
    """
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)
    sns.set_theme(style="whitegrid", context="paper")

# def save_fig(fig, name):
#     """
#     Salva uma figura como PDF vetorial na pasta correta (results/figures/).
    
#     Parâmetros:
#         fig (matplotlib.figure.Figure): O objeto da figura a ser salva.
#         name (str): O nome do arquivo sem a extensão.
#     """
#     filepath = FIGURES_DIR / f"{name}.pdf"
#     fig.savefig(filepath, bbox_inches="tight")
#     print(f"\nFigura salva em: {filepath}")

from IPython.display import display, Markdown, FileLink

def save_fig(fig, name):
    """
    Salva uma figura como PDF vetorial na pasta correta (results/figures/).
    
    Parâmetros:
        fig (matplotlib.figure.Figure): O objeto da figura a ser salva.
        name (str): O nome do arquivo sem a extensão.
    """
    filepath = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(filepath, bbox_inches="tight")

    display(Markdown(
        f"Figura salva como PDF vetorial\n\n"
        f"- Arquivo: `{filepath}`\n"
        f"- Como abrir: clique no link abaixo ou acesse a pasta `{FIGURES_DIR}`."
    ))
    display(FileLink(filepath, result_html_prefix="📄 Abrir figura: "))
    
def clean_results():
    """
    Apaga todos os ficheiros e subpastas gerados no diretório de resultados,
    recriando a estrutura de pastas vazias logo a seguir.
    Útil para limpar saídas residuais de execuções anteriores.
    """
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
        
    # Recria as pastas para evitar erros de "FileNotFound" nas próximas execuções
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Limpeza concluída: O diretório '{RESULTS_DIR.name}/' foi esvaziado.")