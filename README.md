# TDAH Aware: Apoio Clínico Inteligente com IA

O **TDAH Aware** é uma plataforma de triagem que utiliza Inteligência Artificial para identificar padrões comportamentais associados ao TDAH.  

A solução foi desenvolvida tanto para **orientação ao público geral** quanto como **ferramenta de apoio à decisão para profissionais de saúde**.

---

## Demonstração Online

Acesse o Web App em tempo real:  
https://tdahaware.streamlit.app/

---

## Como Funciona

O sistema processa dados por meio de um modelo de Machine Learning (**Random Forest**), treinado com o dataset clínico *Hyperaktiv*.

### Entradas
- Perfil demográfico  
- Escalas clínicas:
  - ASRS  
  - WURS  
  - MADRS  
  - HADS  

### Inteligência
O modelo realiza um **diagnóstico diferencial**, avaliando se os sintomas de desatenção são:
- Primários (TDAH)  
- Ou secundários a quadros como ansiedade ou depressão  

### Resultado
Após responder as perguntas de todos os questionários, o usuário recebe um laudo estatístico imediato, que apresenta a probabilidade de sincronia com o perfil de TDAH em formato de porcentagem (0-100%). O resultado é acompanhado de uma interpretação textual que indica se os padrões informados possuem Alta ou Baixa Sincronia com os casos clínicos reais utilizados no treinamento da IA, servindo como um ponto de partida para a avaliação profissional.

---

## Tecnologias Utilizadas

- **Linguagem:** Python 3.11+  
- **IA/ML:** Scikit-Learn, Joblib  
- **Processamento de Dados:** Pandas, NumPy  
- **Interface:** Streamlit  

---

## Aviso importante ⚠️
Esta ferramenta é um protótipo de triagem estatística: ela gera uma probabilidade matemática e não substitui diagnóstico médico.
Embora os resultados iniciais sejam promissores, há limitações importantes: o modelo se baseia principalmente em escalas comportamentais (ASRS, WURS etc.) e não inclui o CPT-II laboratorial, removido para viabilizar o uso via web.
Portanto, o sistema deve ser utilizado apenas como apoio à triagem, sempre com interpretação de um profissional de saúde qualificado.

---

## Execução Local

```bash
git clone https://github.com/ArthurCruzz17/TDAH_Aware.git
cd TDAH_Aware

pip install -r requirements.txt
streamlit run app.py
