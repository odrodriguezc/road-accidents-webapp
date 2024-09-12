import streamlit as st


# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Introduction", "Dataset", "Data Analysis", "Modeling", "Conclusions"])

# Page Routing


def introduction_render():
    # Title of the Project
      st.title("Accidents routiers en France ")

      # Image (make sure the image is placed in the 'images' folder within the project directory)
      st.image("images/accident.png", caption="Data Science Project Overview", use_column_width=True)

      # Introduction Text
      st.write("""
      ### Aperçu du projet
      Ce projet se concentre sur [problème ou sujet spécifique], avec pour objectif d'explorer et d'analyser un ensemble de données, et de construire des modèles prédictifs pour résoudre [problème ou objectif].
      
      ### Objectifs :
      1. **Comprendre le jeu de données** : Une analyse approfondie des caractéristiques du jeu de données, y compris ses sources, ses fonctionnalités et sa qualité.
      2. **Analyse des données** : Explorer et visualiser les données pour découvrir des modèles, tendances et insights clés.
      3. **Modélisation** : Construire des modèles d'apprentissage automatique pour résoudre le problème et évaluer leurs performances.
      4. **Conclusions** : Résumer les résultats, les performances des modèles et proposer des étapes ou recommandations à suivre.

      Cette application Streamlit vous guidera à travers ces sections, en commençant par les caractéristiques du jeu de données, pour ensuite passer à l'analyse et la construction de modèles.
      """)

      # Workflow Diagram or Additional Context (optional)
      st.write("""
      ### Déroulement du projet
      - **Étape 1** : Exploration et nettoyage des données.
      - **Étape 2** : Analyse exploratoire des données (EDA).
      - **Étape 3** : Construction et évaluation du modèle.
      - **Étape 4** : Conclusion et étapes suivantes.
      """)

def dataset_render():
    st.title("Dataset")

def analisys_render():
    # Title of the Project
    st.title("Analysis ")

    # Image (make sure the image is placed in the 'images' folder within the project directory)
    st.image("images/accident.png", caption="Data Science Project Overview", use_column_width=True)

    # Introduction Text
    st.write("""
    ### Aperçu du projet
    Ce projet se concentre sur [problème ou sujet spécifique], avec pour objectif d'explorer et d'analyser un ensemble de données, et de construire des modèles prédictifs pour résoudre [problème ou objectif].
    
    ### Objectifs :
    1. **Comprendre le jeu de données** : Une analyse approfondie des caractéristiques du jeu de données, y compris ses sources, ses fonctionnalités et sa qualité.
    2. **Analyse des données** : Explorer et visualiser les données pour découvrir des modèles, tendances et insights clés.
    3. **Modélisation** : Construire des modèles d'apprentissage automatique pour résoudre le problème et évaluer leurs performances.
    4. **Conclusions** : Résumer les résultats, les performances des modèles et proposer des étapes ou recommandations à suivre.

    Cette application Streamlit vous guidera à travers ces sections, en commençant par les caractéristiques du jeu de données, pour ensuite passer à l'analyse et la construction de modèles.
    """)

    # Workflow Diagram or Additional Context (optional)
    st.write("""
    ### Déroulement du projet
    - **Étape 1** : Exploration et nettoyage des données.
    - **Étape 2** : Analyse exploratoire des données (EDA).
    - **Étape 3** : Construction et évaluation du modèle.
    - **Étape 4** : Conclusion et étapes suivantes.
    """)


def modeling_render():
    st.title("Modelisation ")

def conclusion_render():
    st.title("Conclusion")


if options == "Introduction":
    introduction_render()
elif options == "Dataset":
    dataset_render()
elif options == "Data Analysis":
    analisys_render()
elif options == "Modeling":
    modeling_render()
elif options == "Conclusions":
    conclusion_render()