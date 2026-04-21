import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def treinar_modelo():
    caminho_dados = 'data/processed_data.csv'
    pasta_modelos = 'models'

    if not os.path.exists(pasta_modelos):
        os.makedirs(pasta_modelos)

    df = pd.read_csv(caminho_dados)
    
    X = df.drop('ADHD', axis=1)
    y = df['ADHD']

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    joblib.dump(modelo, 'models/tdah_model.pkl')
    print("O modelo foi treinado e salvo em 'models/tdah_model.pkl'.")

if __name__ == "__main__":
    treinar_modelo()