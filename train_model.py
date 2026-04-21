import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.impute import SimpleImputer
import os

try:
    df = pd.read_csv('patient_info.csv', sep=None, engine='python')
    print("Arquivo carregado com sucesso")
except Exception as e:
    print(f"Erro ao ler o arquivo: {e}")
    exit()

features = ['SEX', 'AGE', 'ASRS', 'WURS', 'MADRS', 'CPT_II', 'ANXIETY']

for col in features:
    if col not in df.columns:
        print(f"Erro: A coluna {col} não foi encontrada no CSV")
        exit()

X = df[features].copy()
y = df['ADHD']

for col in features:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X['ANXIETY'] = X['ANXIETY'].replace(9, 0)

imputer = SimpleImputer(strategy='median')
X_clean = imputer.fit_transform(X)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_clean, y)

if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(modelo, 'models/tdah_model.pkl')
joblib.dump(imputer, 'models/imputer.pkl')

print("Modelo treinado com sucesso")
print(f"Campos aprendidos: {features}")