import streamlit as st 

pgs = [
    st.Page("home.py", title="Home", icon="🏠"),
    st.Page("pagina_predictor.py", title="Predictor", icon="🧠"),
    st.Page("pagina_modelo.py", title="Crear Modelo", icon="🤖"),
    st.Page("pagina_nosotros.py",title="Nosotros", icon="👨‍👦‍👦"),
]

pg = st.navigation(pgs)

pg.run()
