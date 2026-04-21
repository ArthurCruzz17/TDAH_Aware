import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="TDAH Aware", page_icon="🧠", layout="wide")

try:
    modelo = joblib.load('models/tdah_model.pkl')
    imputer = joblib.load('models/imputer.pkl')
except:
    st.error("Arquivos do modelo não encontrados na pasta 'models/'")
    st.stop()
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3079/3079013.png", width=80)
    st.title("TDAH Aware")
    st.info("""
    **Sua atenção em foco** > Este sistema analisa seu comportamento através de modelos de IA treinados com a base de dados **Hyperaktiv**, traduzindo padrões em informações úteis para o seu dia a dia.
    """)
    st.divider()
    st.markdown("[📚 Base de Dados Utilizada no TDAH Aware:](https://www.kaggle.com/datasets/arashnic/adhd-diagnosis-data)")
    st.divider()
    st.warning("⚠️ **Nota:** Esta é uma ferramenta de triagem estatística e não substitui um diagnóstico médico. Os resultados devem ser interpretados sempre em conjunto com a avaliação clínica completa do paciente.")

st.title("🧠 TDAH Aware: Apoio Clínico Inteligente")
st.caption("Preencha as informações abaixo para obter uma análise computacional complementar:")
st.markdown("---")

tab_cl, tab_res = st.tabs(["Escalas Clínicas", "Resultado Final"])

with tab_cl:
    st.header("Protocolo de Avaliação")
    
    with st.expander("👤 1. Perfil", expanded=True):
        c1, c2 = st.columns(2)
        sexo = c1.selectbox("Sexo Biológico", ["Feminino", "Masculino"])
        sexo_val = 1 if sexo == "Masculino" else 0
        idade_real = c2.number_input("Idade Real", 18, 80, 25)

    with st.expander("📝 2. Sintomas Atuais (ASRS-6)"):
        op_asrs = {"Nunca": 0, "Raramente": 1, "Às vezes": 2, "Frequentemente": 3, "Muito Frequentemente": 4}
        q_asrs = ["Dificuldade para finalizar detalhes?", "Dificuldade para organização?", "Problemas para lembrar compromissos?", "Evita ou adia o início de tarefas complexas?", "Costuma ficar mexendo as mãos ou batucando os pés quando precisa ficar sentado por muito tempo?", "Sente uma agitação interna constante, como se estivesse sempre 'ligado na tomada' ou sem conseguir relaxar?"]
        escore_asrs = sum([op_asrs[st.selectbox(q, op_asrs.keys(), key=f"asrs_{i}")] for i, q in enumerate(q_asrs)])

    with st.expander("👶 3. Histórico Infantil (WURS)"):
        op_wurs = {"Nada": 0, "Um pouco": 1, "Moderado": 2, "Muito": 3, "Severo": 4}
        q_wurs = ["Tinha facilidade em se distrair ou sentia dificuldade para manter o foco nas tarefas?", "Era muito agitado, daqueles que não paravam quietos ou precisavam estar sempre se mexendo?", "Costumava agir por impulso, fazendo as coisas sem pensar muito antes?", "Tinha problemas frequentes com regras, disciplina na escola ou com autoridades?", "Costumava se perder em pensamentos ou parecia 'desligado' do que acontecia ao redor?"]
        escore_wurs = sum([op_wurs[st.selectbox(q, op_wurs.keys(), key=f"wurs_{i}")] for i, q in enumerate(q_wurs)])

    with st.expander("😰 4. Escala de Ansiedade (HADS-A)"):
        op_hads = {"Não/Nunca": 0, "Às vezes": 1, "Muitas vezes": 2, "Quase sempre": 3}
        q_hads = ["Sinto-me tenso ou contraído?", "Sinto medo de algo ruim acontecer?", "Sinto desconforto ou sensações físicas no estômago quando estou sob pressão?", "Sou dominado por pensamentos que me causam apreensão ou medo?"]
        escore_hads = sum([op_hads[st.selectbox(q, op_hads.keys(), key=f"hads_{i}")] for i, q in enumerate(q_hads)])
        ansiedade_val = 1 if escore_hads >= 8 else 0

    with st.expander("😔 5. Bem estar emocional (MADRS)"):
        madrs_val = st.select_slider("Como você avalia seu nível de desânimo ou tristeza na última semana?", options=list(range(0, 61)), value=15)

with tab_res:
    st.header("📊 Laudo Estatístico")
    
    if st.button("🔍 GERAR RESULTADO FINAL", use_container_width=True):
        
        idade_para_modelo = idade_real 
        asrs_final = escore_asrs * 3  
        wurs_final = escore_wurs * 5  
        cpt_final = 1 

        input_data = pd.DataFrame([[
            sexo_val, idade_para_modelo, asrs_final, wurs_final, 
            madrs_val, cpt_final, ansiedade_val
        ]], columns=['SEX', 'AGE', 'ASRS', 'WURS', 'MADRS', 'CPT_II', 'ANXIETY'])
        
        input_clean = imputer.transform(input_data)
        prob = modelo.predict_proba(input_clean)[0][1]
        
        st.divider()
        st.subheader(f"Probabilidade de TDAH Detectada: {prob:.1%}")
        st.progress(prob)
        
        if prob > 0.5:
            st.error("### Alta Sincronia com Perfil de Desatenção/Hiperatividade")
            st.markdown("Os indicadores coletados apresentam uma correlação estatística significativa com os padrões observados em indivíduos diagnosticados com TDAH no dataset clínico de referência.")
        else:
            st.success("### Baixa Sincronia com Perfil de Desatenção/Hiperatividade")
            st.markdown("Seu perfil mostra um processamento de informações equilibrado e dentro dos padrões de foco convencionais.")

    if st.button("Reiniciar Sistema"):
        st.rerun()

st.markdown("---")
st.caption("Projeto destinado a fins de estudo e demonstração. Consulte um profissional de saúde para avaliação clínica completa.")
st.markdown("---")
st.caption("Desenvolvido por Arthur de Paula Cruz")
