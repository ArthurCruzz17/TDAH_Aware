import pandas as pd
import os

def limpar_dados():
    caminho_bruto = 'data/patient_info.csv'
    caminho_processado = 'data/processed_data.csv'

    if not os.path.exists(caminho_bruto):
        print(f"Erro: O arquivo {caminho_bruto} não foi encontrado!")
        return

    df = pd.read_csv(caminho_bruto, sep=';')

    colunas = ['ASRS', 'WURS', 'MADRS', 'CPT_II', 'ADHD']
    
    df_limpo = df[colunas].dropna()

    df_limpo.to_csv(caminho_processado, index=False)
    print(f"O arquivo '{caminho_processado}' agora inclui o CPT_II.")

if __name__ == "__main__":
    limpar_dados()