# (Esta versão é o fork 1 da versão principal)

# Predição de óbito ou readmissão em pacientes com insuficiência cardíaca

Trabalho final da disciplina Inteligência Computacional Aplicada a Problemas de Engenharia (UFMG, 2026/2).

A partir das informações registradas durante a internação, o objetivo é prever se um paciente com insuficiência cardíaca terá óbito ou readmissão hospitalar em até 6 meses. O trabalho percorre análise exploratória, extração e seleção de atributos, treinamento de modelos lineares e não lineares, e avaliação e seleção de modelos.

> **Objetivo**: prever se um paciente com insuficiência cardíaca terá óbito ou readmissão hospitalar em até 6 meses.

## 1. Base de dados

Usamos a base Hospitalized patients with heart failure: integrating electronic healthcare records and external outcome data (versão 1.3), do PhysioNet:

https://physionet.org/content/heart-failure-zigong/1.3/

São 2008 internações de pacientes com insuficiência cardíaca no Zigong Fourth People's Hospital (China), entre 2016 e 2019, com dados clínicos, exames, medicamentos e desfechos de seguimento.

> Os dados não estão neste repositório. A base é de acesso restrito, e o acordo de uso do PhysioNet proíbe redistribuí-la. Cada pessoa precisa obter o acesso e baixar os arquivos por conta própria.

### 1.1. Como obter o acesso

A base tem acesso do tipo *restricted*: não exige credenciamento nem treinamento, só uma conta e a assinatura do acordo de uso.

1. Crie uma conta no [PhysioNet](https://physionet.org/register/) e faça login.
2. Abra a [página da base](https://physionet.org/content/heart-failure-zigong/1.3/) e, na seção *Files*, assine o **Data Use Agreement** (*PhysioNet Restricted Health Data Use Agreement 1.5.0*).

### 1.2. Como baixar

Depois de assinar o acordo, a seção *Files* da página da base libera o download. Use uma das opções:

- ZIP: clique em *Download the ZIP file* e extraia o conteúdo.
- Terminal (Linux):
  ```bash
  wget -r -N -c -np --user SEU_USUARIO --ask-password https://physionet.org/files/heart-failure-zigong/1.3/
  ```
  Os arquivos são salvos em `physionet.org/files/heart-failure-zigong/1.3/`.

### 1.3. Onde colocar os arquivos

Crie uma pasta **`DataBase/`** na raiz do repositório e coloque nela os arquivos baixados, sem subpastas e sem renomear:

```
<raiz do repositório>/
├── DataBase/                  <- criar e colocar os arquivos aqui
│   ├── dat.csv                <- obrigatório
│   ├── dat_md.csv             <- obrigatório
│   ├── dataDictionary.csv     <- obrigatório
│   ├── LICENSE.txt
│   └── SHA256SUMS.txt
├── notebooks/
│   └── main.ipynb             <- Painel executivo e orquestração
├── src/                       <- Módulos Python com a lógica do projeto
│   ├── __init__.py
│   ├── utils.py               <- Utilidades e configurações globais
│   ├── eda.py                 <- Análise exploratória de dados
│   ├── preprocessing.py       <- Limpeza e tratamento de dados ausentes
│   ├── features.py            <- Seleção e engenharia de atributos
│   ├── train.py               <- Definição, ajuste e otimização dos modelos
│   └── evaluate.py            <- Métricas, validação cruzada e avaliação
├── results/                   <- Tabelas e figuras geradas para o artigo
├── requirements.txt
└── README.md
```

O notebook lê os três CSVs obrigatórios a partir de `../DataBase/`, caminho relativo à pasta `notebooks/`. O nome da pasta deve ser exatamente `DataBase`, com "D" e "B" maiúsculos.

Para conferir se os arquivos estão íntegros (macOS ou Linux):

```bash
cd DataBase
sed 's/ /  /' SHA256SUMS.txt | shasum -a 256 -c -
```

As quatro linhas devem terminar em `OK`.

## 2. Configuração Inicial e Execução

Requer Python 3.13. Na primeira vez, rode na raiz do repositório:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
nbstripout --install
```

Depois, abra `notebooks/main.ipynb` no VS Code, selecione o interpretador `.venv` como kernel e rode todas as células. Fora do VS Code, use `jupyter lab notebooks/main.ipynb` com o ambiente ativo.

As figuras são salvas em `results/figures/` e as tabelas em `results/`.

> Por que o `nbstripout --install`? As saídas das células do notebook mostram dados de pacientes, que não podem ser publicados. Esse comando faz o git remover as saídas a cada commit; a cópia local do notebook continua com tudo. O git não permite que um repositório ative isso sozinho ao ser clonado, por isso o comando precisa ser rodado uma vez em cada clone.

## 3. Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `notebooks/main.ipynb` | Orquestração do fluxo, visualizações e relatórios para o artigo |
| `src/utils.py` | Centraliza paths, configurações visuais e sementes aleatórias. |
| `src/eda.py` | Responsável por gerar visualizações e resumos estatísticos. |
| `src/preprocessing.py` | Funções de limpeza, tratamento de missings e tipagem |
| `src/features.py` | Funções de engenharia de atributos e seleção de variáveis |
| `src/train.py` | Pipeline de treinamento e ajuste de hiperparâmetros dos modelos |
| `src/evaluate.py` | Cálculo de métricas de desempenho e validação cruzada |
| `results/` | Tabelas (CSV) e figuras (`figures/`) geradas pelo notebook |
| `requirements.txt` | Dependências com versões fixadas |

## 4. Referências da base

- Zhang, Z., Cao, L., Zhao, Y., Xu, Z., Chen, R., Lv, L., & Xu, P. (2022). *Hospitalized patients with heart failure: integrating electronic healthcare records and external outcome data* (version 1.3). PhysioNet. https://doi.org/10.13026/5m60-vs44
- Zhang, Z., Cao, L., Chen, R., Zhao, Y., Lv, L., Xu, Z., & Xu, P. (2021). Electronic healthcare records and external outcome data for hospitalized patients with heart failure. *Scientific Data*, 8(1), 46. https://doi.org/10.1038/s41597-021-00835-9
