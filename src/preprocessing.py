"""
Módulo de pré-processamento de dados.
Responsável pelo carregamento bruto, definição de grupos de variáveis 
e formulação da variável alvo (target).
"""

import pandas as pd
from src.utils import DATA_DIR

def load_data():
    """
    Carrega os conjuntos de dados principais a partir do diretório DataBase.
    
    Retorna:
        patients (pd.DataFrame): Dados clínicos e desfechos das internações.
        drugs (pd.DataFrame): Registo de medicamentos administrados.
        data_dictionary (pd.DataFrame): Dicionário com a descrição das variáveis.
    """
    patients = pd.read_csv(DATA_DIR / "dat.csv", index_col=0)
    drugs = pd.read_csv(DATA_DIR / "dat_md.csv", index_col=0)
    data_dictionary = pd.read_csv(DATA_DIR / "dataDictionary.csv", index_col=0)
    
    return patients, drugs, data_dictionary


def get_variable_groups(patients_df):
    """
    Agrupa as colunas do DataFrame de pacientes por domínio clínico,
    seguindo a ordem estabelecida no dataDictionary.
    
    Parâmetros:
        patients_df (pd.DataFrame): O DataFrame carregado de pacientes.
        
    Retorna:
        dict: Um dicionário onde as chaves são os nomes dos grupos e 
              os valores são listas com os nomes das colunas.
    """
    def column_span(first, last):
        """Função auxiliar para capturar um intervalo de colunas sequenciais."""
        return patients_df.loc[:, first:last].columns.tolist()

    groups = {
        "Identificação": ["inpatient.number"],
        "Demografia": ["gender", "ageCat", "occupation"],
        "Admissão e histórico": ["admission.ward", "admission.way", "visit.times"],
        "Sinais vitais e antropometria": column_span("body.temperature", "BMI"),
        "Classificação da IC": column_span("type.of.heart.failure", "Killip.grade"),
        "Comorbidades": column_span("myocardial.infarction", "CCI.score"),
        "Estado clínico na admissão": column_span("type.II.respiratory.failure", "acute.renal.failure") + ["GCS"],
        "Ecocardiograma": column_span("LVEF", "tricuspid.valve.return.pressure"),
        "Lab. — função renal": column_span("creatinine.enzymatic.method", "cystatin"),
        "Lab. — hemograma": column_span("white.blood.cell", "neutrophil.count"),
        "Lab. — coagulação": column_span("D.dimer", "fibrinogen"),
        "Lab. — marcadores cardíacos": ["high.sensitivity.troponin", "myoglobin"]
            + column_span("creatine.kinase.isoenzyme.to.creatine.kinase", "brain.natriuretic.peptide"),
        "Lab. — eletrólitos": column_span("carbon.dioxide.binding.capacity", "serum.magnesium"),
        "Lab. — bioquímica hepática e inflamação": column_span("high.sensitivity.protein", "erythrocyte.sedimentation.rate"),
        "Lab. — lipídios": column_span("cholesterol", "lipoprotein"),
        "Lab. — gasometria": column_span("pH", "total.hemoglobin"),
        "Desfechos e seguimento": column_span("outcome.during.hospitalization", "time.to.emergency.department.within.6.months"),
        "Informações da alta": ["DestinationDischarge", "discharge.department", "dischargeDay"],
    }
    
    return groups


def create_target(patients_df):
    """
    Cria a variável alvo binária: Óbito ou readmissão em até 6 meses.
    
    Parâmetros:
        patients_df (pd.DataFrame): O DataFrame de pacientes contendo as colunas de desfecho.
        
    Retorna:
        pd.Series: Uma série Pandas (0 ou 1) representando a variável alvo, 
                   com o mesmo índice do DataFrame original.
    """
    readmission_columns = [
        "re.admission.within.28.days", 
        "re.admission.within.3.months", 
        "re.admission.within.6.months"
    ]
    
    # Verifica se houve readmissão em qualquer uma das janelas
    readmitted = patients_df[readmission_columns].eq(1).any(axis=1)
    
    # Verifica óbito em até 6 meses (inclui óbitos durante a internação)
    died = patients_df["death.within.6.months"].eq(1)
    
    # O alvo é a união (OR) dos dois eventos
    target = (readmitted | died).astype(int).rename("target")
    
    return target

def get_variable_types(df):
    """
    Classifica as variáveis do DataFrame em categorias de tipologia
    para alimentar os pipelines de pré-processamento.
    
    Parâmetros:
        df (pd.DataFrame): O DataFrame a analisar.
        
    Retorna:
        dict: Dicionário com as listas de colunas 'binarias', 'categoricas', 
              'continuas' e 'discretas'.
    """
    binary = []
    categorical = []
    continuous = []
    discrete = []
    
    for col in df.columns:
        n_unique = df[col].dropna().nunique()
        if n_unique <= 2:
            binary.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if n_unique > 10:
                continuous.append(col)
            else:
                discrete.append(col)
        else:
            categorical.append(col)
            
    return {
        "binarias": binary,
        "categoricas": categorical,
        "continuas": continuous,
        "discretas": discrete
    }