import json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx
import io

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Introduction", "Dataset", "Data Analysis", "Modeling", "Conclusions"])


def introduction_render():
    # Title of the Project
      st.title("Accidents routiers en France ")

      # Image (make sure the image is placed in the 'images' folder within the project directory)
      st.image("images/accident.png", caption="Source : « 20 Minutes », https://www.20minutes.fr/", use_column_width=True)


      # Introduction Text
      st.write("### Contexte")
      st.write("""
           - Un accident est un phénomène complexe, 
             en tant qu’usagers de la route et proches des services de transport, c'est plus qu'une curiosité pour nous.
           
           - Formation en data science : validation d'un projet
              
           Nous avons sollicité de travailler sur le projet « Accidents routiers en France »
           """)

  

      st.write("### Objectifs")
      st.write("""
           - « Bases de données annuelles accidents corporels de la circulation routière » géré par l'ONISR  
             « Fichier BAAC »,  de 2020 à 2022.
           
           - « Modèle de prédiction de la gravité des accidents routiers en France »,

           - Se familiariser avec divers outils de data science :  jupyter notebook , git et git hub, Visual Studio Code, Spider … 
           
           """)

      # Workflow Diagram or Additional Context (optional)
      st.write("""
      ### Déroulement du projet
      - **Étape 1** : Exploration et nettoyage des données.
      - **Étape 2** : Analyse exploratoire des données (EDA).
      - **Étape 3** : Construction et évaluation du modèle.
      - **Étape 4** : Conclusion et étapes suivantes.
      """)


accidents_df = pd.read_csv("data/accidents_clean_full_qualitat.csv")
# Nommons aa_df (analysing_accidents_dataframe) notre dataframe pour l'analyse 
aa_df = accidents_df.drop(accidents_df.columns[1], axis=1)
aa_df = aa_df.drop(aa_df.columns[0], axis=1)

def dataset_render():
    # Title of the Project
      st.title("L'aquisition des données")      
      st.write("""
           Consolidation des principaux fichiers BAAC :  
           
            - Le fichier « caracteristiques.csv », 
            - Le fichier « usagers.csv »,
            - Le fichier « lieux.csv »,
            - Le fichier « vehicules.csv ».


           Traitement des données manquantes
           
           Elimination des variables non pertinentes,

           Recodification des colonnes pour plus de lisibilités.
                      """)
      buffer = io.StringIO()
      aa_df.info(buf=buffer)
      s_aa_df = buffer.getvalue()
      
      st.text(s_aa_df)
      
      st.dataframe(aa_df.head()) 

def analisys_render():
    # Title of the Project
    st.title("Exploration des données")

    

    # Introduction Text
    st.write("""
             Notre projet est basé sur des données récoltées sur le site de l'ONISR et les effectifs avancés vont concerner trois périodes allant de 2020 à 2022.

            ### Gravité : la variable cible    
              - Pour la variable « gravité », plus de 40% indemnes et 40% avec des blessures légères.
              - Les accidents mortels sont de l’ordre de 2% et les hospitalisations dépassent les 10%. … 
    
             """)
    
    fig = plt.figure(figsize = (10,6))
    target = aa_df['gravite'].astype(int)
    dikt_ord = {'Indemne' : '1- Indemne',
            'Blessé léger' : '2- Blessé léger',
            'Blessé hospitalisé' : '3- Blessé hospitalisé',
            'Tué': '4 - Tué'}
    aa_df['gravite_cat'] = aa_df['gravite_cat'].replace(dikt_ord)
    ax = sns.countplot(data = aa_df.sort_values('gravite_cat', ascending = True), y = 'gravite_cat')
    total = len(aa_df['gravite_cat'])
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.04
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

    plt.title("Distribution des categories de gravités")
    st.pyplot(fig)   
    
    st.write("""
            ### Gravité / Sexe :
              - Pour la variable « gravité », plus de 40% indemnes et 40% avec des blessures légères.
              - Les accidents mortels sont de l’ordre de 2% et les hospitalisations dépassent les 10%. …     
            """)
    fig = plt.figure(figsize = (10,6))
    target = aa_df['gravite'].astype(int)
    sns.countplot(data = aa_df.sort_values('gravite_cat', ascending = True), x = 'gravite_cat', hue = 'sexe' )
    plt.title("Gravites / sexe ")
    st.pyplot(fig)
    
    st.write("""
            ### Périodicités des accidents :
              - 700 accidents / jours en moyenne.
              - Décrochage en mars 2020, à moins de 200, "un effet confinement" lié à la pandémie covid 2019 …     
            """)
# Image (make sure the image is placed in the 'images' folder within the project directory)
    st.image("images/06_Acc_par_jour.png", caption="Nombre d'accidents par jour", use_column_width=True)

    st.write("""          
              - Ces trois dernières années, + d'accidents en juillet et en septembre.
              - Un peu plus calme en avril (encore un impact confinnement?)
           """)

    fig = plt.figure(figsize = (10,6))
    sns.histplot(data=aa_df, x="mois", bins=12, palette = 'deep')
    st.pyplot(fig)


    st.write("""          
           En semaine, on a une fréquence plus élévé le vendredi et ça baisse significativement le dimanche.
           """)
    st.image("images/08_Acc_par_jour_semaine.png", caption="Les accidents en semaine", use_column_width=True)


    st.write("""
             Concernant les horaires, le graphique suivant montre qu’on a moins d’accidents généralement de 3 à 4 heures du matin mais on constate deux pics : 
              - à 8 heure du matin 35000 cas pour les 3 périodes.
              - à 17h de l’après-midi 60 000 cas de 2020 à 2022.
            """)
    st.image("images/09_Acc_par_heure.png", caption="Les accidents en semaine", use_column_width=True)

    st.write("""
             Géographiquement, pas de concentration de type d’accidents sur une localité particulière. 
             Il semble que la gravité des accidents est dispersée aléatoirement d’une manière générale. 
            """)
    st.image("images/10_Distri_Geo.png", caption="Les accidents en semaine", use_column_width=True)

def modeling_render():
    # Titre principal de la page
    st.title("Modélisation de Classification")

    # Section 1: Préprocessing
    st.header("1. Préprocessing")

    # Sous-section A: Split X, Y
    st.subheader("A. Split X, y")
    st.write("""
    Notre variable target (y) est la gravité de l'accident, représentée par la variable 'gravite', 
    que l'on sépare du reste des variables explicatives (X). 
    """)

    # Sous-section B: Sélection de Variables
    st.subheader("B. Sélection de Variables")
    st.write("Suppression des variables ")
    # Tableau des variables supprimées
    df_suppressed = pd.DataFrame({
        "Variable": ["latitude", "longitude", "commune"],
        "Raison": ["Hors problématique", "Hors problématique", "Trop de catégories"]
    })
    st.table(df_suppressed)

    # Liste des variables
    categorical_features = [
        'place', 
        'sexe', 
        'categorie_usager', 
        'type_de_trajet', 
        'element_de_securite_1', 
        'luminosite', 
        'agglomeration',
        'departement',
        'intersection', 
        'type_de_collision', 
        'condition_atmospheriques', 
        'categorie_de_route', 
        'regime_de_circulation', 
        'voie_reserve', 
        'profile_de_la_route', 
        'trace_en_plan', 
        'etat_de_la_surface', 
        'infrastructure', 
        'situation_de_l_accident', 
        'sens_de_circulation', 
        'categorie_vehicule', 
        'obstacle_fixe', 
        'obstacle_mobile', 
        'point_de_choque_initial', 
        'manoeuvre', 
        'type_de_motor', 
        'mois', 
    ]

    numerical_features = [
        'jour', 
        'hrmn', 
        'nombre_voies_de_circulation', 
        'vitesse_maximale', 
        'age'
    ]

    # Création d'un DataFrame combinant les deux types de variables
    df_variables = pd.DataFrame({
        "Variable": categorical_features + numerical_features,
        "Type": ["Catégorique"] * len(categorical_features) + ["Numérique"] * len(numerical_features)
    })

    st.subheader("B. Sélection de Variables")
    st.write("Liste des variables catégoriques et numériques utilisées pour l'analyse:")
    st.table(df_variables)

    # Graphe des types de variables
    st.write("Types de variables:")

    variable_counts = df_variables['Type'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(variable_counts, labels=variable_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  
    st.pyplot(fig)

    # Sous-section C: Étude de la Cardinalité
    st.subheader("C. Étude de la Cardinalité")
    st.image("images/cardinalite_variables.png", caption="Graphe de la Cardinalité des Variables")
    st.write("Méthodes d'encodage en fonction de la cardinalité des variables:")
    st.markdown("""
    - **Basse cardinalité** : One-hot encoding
    - **Cardinalité moyenne** : Frequency encoding
    - **Cardinalité élevée** : Target encoding
    """)

    # Sous-section D: Split et Standardisation
    st.subheader("D. Split et Standardisation")
    st.write("Découpage des données avec une proportion de 80% pour l'entraînement et 20% pour le test (random_state = 123).")

    # Création des proportions pour l'entraînement et le test
    labels = ['Entraînement (80%)', 'Test (20%)']
    sizes = [80, 20]

    # Création d'un graphique en camembert
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Assure que le graphique est bien circulaire
    # Ajout de la légende pour le random_state
    st.pyplot(fig)
    st.write("Le `random_state` est fixé à 123 pour assurer la reproductibilité.")
    
    st.write("Après encodage on 144 colonnes.")
    st.image("images/vars_after_encoding.png", caption="vars_after_encoding")


    # Sous-section E: Réduction de Dimension
    st.subheader("E. Réduction de Dimension")
    st.write("Analyse en Composantes Principales (PCA)...")
    # Graphe PCA
    st.image("images/pca.png", caption="PCA")
    # Graphe des 50 premiers features
    st.write("Graphe des correlations")
    st.image("images/plot_correlation.png", caption="PCA")

    st.write("50 prèmieres features")
    st.image("images/50_first_features.png", caption="50_first_features")

    # Sous-section F: Techniques de rééchantillonnage
    st.subheader("F. Rééchantillonnage - Techniques de balancing")
    # Graphe de distribution de la cible
    st.write("Distribution de la cible avant équilibrage")
    st.image("images/dist_target_multi.png", caption="dist_target_multi")

    # Liste des techniques
    st.write("Techniques utilisées:")
    st.write("  - Random over sampling\n  - Random under sampling\n  - SMOTE\n  - Tomek Links")
    # Résultats après rééchantillonnage
    df_balancing = pd.read_csv("images/balancing_methods_result.csv")
    st.table(df_balancing)
    # Graphe distribution après échantillonnage
    st.write("Graphe des distributions après échantillonnage")
    st.image("images/comparaison_distribution_balancing.png", caption="comparaison_distribution_balancing")

    # Section 2: Sélection de Modèles
    st.header("2. Sélection de Modèles (Classification)")

    # Sous-section A: Liste de Modèles
    st.subheader("A. Liste de Modèles")
    st.write("- SVC\n- KNN\n- Logistic Regression\n- Decision Tree\n- Random Forest\n- XGBoost\n- Perceptron Multi-layer")

    # Sous-section B: Sélection des métriques
    st.subheader("B. Sélection des Métriques")
    st.write("- f1-score, recall, accuracy, classification report")

    # Sous-section C: Problématique des ressources
    st.subheader("C. Problématique des ressources")
    # Graphe de monitoring des ressources
    st.image("images/monitoring_pc.png", caption="monitoring_pc")

    # Sous-section D: Résultats avec Bagging
    st.subheader("D. Résultats")

    # Dictionnaire des modèles et des fichiers associés
    models = {
        "Bagging": "bagging_multi",
        "Decision Tree": "decision_tree_multi",
        "KNN": "knn_multi",
        "Logistic Regression": "logistic_regression_multi",
        "Random Forest": "random_forest_multi",
        "SVM": "svm_multi",
        "XGBoost": "xgb_multi"
    }

    # Liste des métriques disponibles
    metrics = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Metrics per Class (CSV)"]

    # Fonction pour vérifier l'existence d'un fichier
    def file_exists(path):
        return os.path.exists(path)

    # Formulaire de sélection
    st.header("Résultats de la sélection de modèles")
    model_selected = st.selectbox("Sélectionnez un modèle", list(models.keys()))
    metric_selected = st.selectbox("Sélectionnez une métrique", metrics)

    # Chemins des fichiers selon le modèle et la métrique sélectionnés
    model_dir = f"images/{models[model_selected]}"

    if metric_selected == "Confusion Matrix":
        file_path = f"{model_dir}/confusion_matrix.png"
    elif metric_selected == "ROC Curve":
        file_path = f"{model_dir}/roc_curve_plot.png"
    elif metric_selected == "Precision-Recall Curve":
        file_path = f"{model_dir}/precision_recall_curve_plot.png"
    elif metric_selected == "Metrics per Class (CSV)":
        file_path = f"{model_dir}/per_class_metrics.csv"

    # Affichage des résultats selon la métrique choisie
    if file_exists(file_path):
        if metric_selected in ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]:
            # Affichage de l'image pour les graphiques
            st.image(file_path, caption=f"{metric_selected} pour le modèle {model_selected}")
        elif metric_selected == "Metrics per Class (CSV)":
            # Affichage du fichier CSV sous forme de tableau
            df = pd.read_csv(file_path)
            st.dataframe(df)
    else:
        st.write(f"Métrique {metric_selected} pour le modèle {model_selected} : Pas disponible.")


    # Dictionnaire des modèles et des répertoires associés
    models = {
    "Bagging": "bagging_multi",
    "Decision Tree": "decision_tree_multi",
    "KNN": "knn_multi",
    "Logistic Regression": "logistic_regression_multi",
    "Random Forest": "random_forest_multi",
    "SVM": "svm_multi",
    "XGBoost": "xgb_multi"
}

    # Liste des métriques à comparer
    metrics_to_compare = ['accuracy_score', 'recall_score', 'precision_score', 'f1_score']

    # Liste pour stocker les DataFrames des métriques par modèle
    df_list = []

    # Parcourir chaque modèle et charger les fichiers CSV
    for model_name, model_dir in models.items():
        csv_path = f"images/{model_dir}/per_class_metrics.csv"
        
        # Vérifier si le fichier CSV existe
        if os.path.exists(csv_path):
            # Lire le fichier CSV
            df = pd.read_csv(csv_path)
            
            # Ajouter une colonne pour identifier le modèle
            df["Model"] = model_name
            
            # Filtrer uniquement les colonnes des métriques à comparer et ajouter la moyenne pour chaque modèle
            df_filtered = df[metrics_to_compare].mean().to_frame().T
            df_filtered['Model'] = model_name
            
            # Ajouter les métriques moyennes au DataFrame filtré
            df_filtered['average'] = df[metrics_to_compare].mean(axis=0).mean()
            
            # Ajouter le DataFrame filtré à la liste
            df_list.append(df_filtered)

    # Combiner tous les DataFrames en un seul
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)

        # Graphique comparatif des métriques entre les modèles
        st.write("Comparaison des métriques entre les modèles (mean) :")
        
        # Création du plot comparatif
        fig, ax = plt.subplots(figsize=(10, 6))

        # Tracer les barres pour chaque métrique par modèle et afficher aussi la moyenne
        for metric in metrics_to_compare + ['average']:
            ax.plot(combined_df['Model'], combined_df[metric], marker='o', label=metric)
        
        # Configurations du graphique
        ax.set_title("Comparaison des performances par modèle")
        ax.set_xlabel("Modèle")
        ax.set_ylabel("Valeur des métriques")
        ax.legend(title="Métriques")
        ax.grid(True)

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
    else:
        st.write("Aucun fichier CSV de métriques n'a été trouvé pour les modèles.")


    # Section 3: Sélection de Modèles (Classification Binaire)
    st.header("3. Sélection de Modèles (Classification Binaire)")

    # Sous-section A: Groupement des Classes
    st.subheader("A. Groupement des Classes")
    # Tableau de groupement des classes
    df_grouping = pd.DataFrame({
        "Accident pas grave": ["Indemne", "Blessé non hospitalisé"],
        "Accident grave": ["Tué", "Hospitalisé"]
    })
    st.table(df_grouping)

    # Création du plot pour illustrer le regroupement des classes
    st.write("Visualisation du regroupement des classes :")

    # Création du graphe
    G = nx.DiGraph()

    # Ajout des nouvelles classes
    G.add_node("Accident grave", size=2)
    G.add_node("Accident pas grave", size=2)

    # Ajout des anciennes classes et des relations avec les nouvelles classes
    G.add_edge("Tué", "Accident grave")
    G.add_edge("Hospitalisé", "Accident grave")
    G.add_edge("Indemne", "Accident pas grave")
    G.add_edge("Blessé non hospitalisé", "Accident pas grave")

    # Dessin du graphe
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)

    # Dessin des nœuds et des arêtes
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=10, font_weight="bold", arrows=True)

    # Affichage du plot dans Streamlit
    st.pyplot(plt)

    # Sous-section B: Rééchantillonnage (SMOTE)
    st.subheader("B. Rééchantillonnage avec SMOTE")
    st.write("Distribution de la cible après équilibrage")
    st.image("images/dist_target_bin.png", caption="dist_target_binary")

    # Sous-section C: Test des Modèles
    st.subheader("C. Test des Modèles")
    st.write("Meilleurs résultats en termes de performance et de consommation des ressources:")
    # Modèles binary-class
    models_binary = {
        "Bagging": "bagging_binary",
        "Decision Tree": "decision_tree_binary",
        "Logistic Regression": "logistic_regression_binary",
        "Random Forest": "random_forest_binary",
        "SVM": "svm_binary",
        "XGBoost": "xgb_binary",
        "MLP": "mlp"
    }

    # Liste des métriques disponibles
    metrics_binary = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Classification Report (JSON)"]

    # Fonction pour vérifier l'existence d'un fichier
    def file_exists(path):
        return os.path.exists(path)

    # Formulaire de sélection pour les modèles binary-class
    st.header("Résultats de la sélection de modèles (Binary-Class)")
    model_selected_binary = st.selectbox("Sélectionnez un modèle", list(models_binary.keys()))
    metric_selected_binary = st.selectbox("Sélectionnez une métrique", metrics_binary)

    # Chemins des fichiers selon le modèle et la métrique sélectionnés
    model_dir_binary = f"images/{models_binary[model_selected_binary]}"

    if metric_selected_binary == "Confusion Matrix":
        file_path_binary = f"{model_dir_binary}/confusion_matrix.png"
    elif metric_selected_binary == "ROC Curve":
        file_path_binary = f"{model_dir_binary}/roc_curve_plot.png"
    elif metric_selected_binary == "Precision-Recall Curve":
        file_path_binary = f"{model_dir_binary}/precision_recall_curve_plot.png"
    elif metric_selected_binary == "Classification Report (JSON)":
        file_path_binary = f"{model_dir_binary}/classification_report.json"

    # Affichage des résultats selon la métrique choisie
    if file_exists(file_path_binary):
        if metric_selected_binary in ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]:
            # Affichage de l'image pour les graphiques
            st.image(file_path_binary, caption=f"{metric_selected_binary} pour le modèle {model_selected_binary}")
        elif metric_selected_binary == "Classification Report (JSON)":
            # Chargement du fichier JSON et affichage des métriques dans un tableau
            with open(file_path_binary, 'r') as file:
                classification_report = json.load(file)
            
            # Extraction des données et conversion en DataFrame pour un meilleur affichage
            if "macro avg" in classification_report:
                report_df = pd.DataFrame(classification_report).transpose()
                st.dataframe(report_df)
            else:
                st.write(f"Le fichier {file_path_binary} ne contient pas de rapport de classification valide.")
    else:
        st.write(f"Métrique {metric_selected_binary} pour le modèle {model_selected_binary} : Pas disponible.")

    def load_metrics_json(json_path, model_name):
        with open(json_path, 'r') as file:
            data = json.load(file)
            # On récupère les scores globaux (si disponibles) pour binary classification
            if "accuracy" in data:
                accuracy = data["accuracy"]
            else:
                accuracy = 0  # Valeur par défaut si non présente
            if "macro avg" in data:
                recall = data["macro avg"]["recall"]
                precision = data["macro avg"]["precision"]
                f1 = data["macro avg"]["f1-score"]
            else:
                recall = precision = f1 = 0  # Valeurs par défaut

            df_filtered = pd.DataFrame({
                'accuracy_score': [accuracy],
                'recall_score': [recall],
                'precision_score': [precision],
                'f1_score': [f1],
                'Model': [model_name],
            })
        return df_filtered

    df_list_binary = []

    # Parcourir chaque modèle binary-class et charger les fichiers JSON
    for model_name, model_dir in models_binary.items():
        json_path = f"images/{model_dir}/classification_report.json"
        
        # Vérifier si le fichier JSON existe
        if os.path.exists(json_path):
            df_filtered = load_metrics_json(json_path, model_name)
            df_list_binary.append(df_filtered)

    # Combiner tous les DataFrames pour les modèles binary-class
    if df_list_binary:
        combined_df_binary = pd.concat(df_list_binary, ignore_index=True)

        # Graphique comparatif des métriques entre les modèles binary-class
        st.write("Comparaison des métriques entre les modèles binary-class :")
        
        # Création du plot comparatif pour les modèles binary-class
        fig_binary, ax_binary = plt.subplots(figsize=(10, 6))

        # Tracer les barres pour chaque métrique par modèle et afficher aussi la moyenne
        for metric in metrics_to_compare:
            ax_binary.plot(combined_df_binary['Model'], combined_df_binary[metric], marker='o', label=metric)
        
        # Configurations du graphique
        ax_binary.set_title("Comparaison des performances par modèle (binary-class)")
        ax_binary.set_xlabel("Modèle")
        ax_binary.set_ylabel("Valeur des métriques")
        ax_binary.legend(title="Métriques")
        ax_binary.grid(True)

        # Afficher le graphique dans Streamlit
        st.pyplot(fig_binary)
    else:
        st.write("Aucun fichier JSON de métriques n'a été trouvé pour les modèles binary-class.")


    # Sous-section D: Grid Search avec Bagging
    st.subheader("D. Grid Search Bagging")
    # Tableau des paramètres de Grid Search
    df_params = pd.DataFrame({
        "Paramètre": ["n_estimators", "max_samples", "max_features", "bootstrap"],
        "Valeur": [[50, 100, 200], [0.5, 1.0], [0.5, 1.0], [True, False]]
    })
    st.write("Paramètres du Grid Search :")
    st.table(df_params)

    # Tableau des meilleurs paramètres
    df_best_params = pd.DataFrame({
        "Paramètre": ["bootstrap", "bootstrap_features", "estimator", "max_features", "max_samples", "n_estimators"],
        "Meilleure Valeur": [True, False, "None", 0.5, 1, 200]
    })
    st.write("Meilleurs paramètres après Grid Search :")
    # Modèles binary-class
    models_best = {
        "Bagging": "bagging_best",
    }
    model_dir_best = "images/bagging_best"
    # Liste des métriques disponibles
    metrics_best = ["Confusion Matrix best", "ROC Curve best", "Precision-Recall Curve best", "Classification Report (JSON) best"]

    # Fonction pour vérifier l'existence d'un fichier
    def file_exists(path):
        return os.path.exists(path)

    # Formulaire de sélection pour les modèles binary-class
    st.header("Résultats de la sélection de modèles (Binary-Class)")
    model_selected_best = st.selectbox("Sélectionnez un modèle", list(models_best.keys()))
    metric_selected_best = st.selectbox("Sélectionnez une métrique", metrics_best)

    # Chemins des fichiers selon le modèle et la métrique sélectionnés
    model_dir_binary = f"images/{models_binary[model_selected_best]}"

    if metric_selected_best == "Confusion Matrix best":
        file_path_best = f"{model_dir_best}/confusion_matrix.png"
    elif metric_selected_best == "ROC Curve best":
        file_path_best = f"{model_dir_best}/roc_curve_plot.png"
    elif metric_selected_best == "Precision-Recall Curve best":
        file_path_best = f"{model_dir_best}/precision_recall_curve_plot.png"
    elif metric_selected_best == "Classification Report (JSON) best":
        file_path_best = f"{model_dir_best}/classification_report.json"

    # Affichage des résultats selon la métrique choisie
    if file_exists(file_path_best):
        if metric_selected_best in ["Confusion Matrix best", "ROC Curve best", "Precision-Recall Curve best"]:
            # Affichage de l'image pour les graphiques
            st.image(file_path_best, caption=f"{metric_selected_best} pour le modèle {metric_selected_best}")
        elif metric_selected_best == "Classification Report (JSON) best":
            # Chargement du fichier JSON et affichage des métriques dans un tableau
            with open(file_path_best, 'r') as file:
                classification_report = json.load(file)
            
            # Extraction des données et conversion en DataFrame pour un meilleur affichage
            if "macro avg" in classification_report:
                report_df = pd.DataFrame(classification_report).transpose()
                st.dataframe(report_df)
            else:
                st.write(f"Le fichier {file_path_best} ne contient pas de rapport de classification valide.")
    else:
        st.write(f"Métrique {metric_selected_best} pour le modèle {metric_selected_best} : Pas disponible.")

    def load_metrics_json(json_path, model_name):
        with open(json_path, 'r') as file:
            data = json.load(file)
            # On récupère les scores globaux (si disponibles) pour binary classification
            if "accuracy" in data:
                accuracy = data["accuracy"]
            else:
                accuracy = 0  # Valeur par défaut si non présente
            if "macro avg" in data:
                recall = data["macro avg"]["recall"]
                precision = data["macro avg"]["precision"]
                f1 = data["macro avg"]["f1-score"]
            else:
                recall = precision = f1 = 0  # Valeurs par défaut

            df_filtered = pd.DataFrame({
                'accuracy_score': [accuracy],
                'recall_score': [recall],
                'precision_score': [precision],
                'f1_score': [f1],
                'Model': [model_name],
            })
        return df_filtered


    # Sous-section E: Comparaison Bagging vs XGBoost
    st.subheader("E. Bagging vs XGBoost")
    # Graphe de comparaison
    st.write("Comparaison des performances des deux modèles")
    st.image("images/compare_bagging_xgb.png", caption="compare_bagging_xgb")

    
    # Section 4: Interprétabilité
    st.header("4. Interprétabilité")
    st.write("Variables les plus importantes dans la prédiction:")
    # Graphe des variables les plus importantes
    st.image("images/shap_more_important.png", caption="")







def conclusion_render():
    st.title("Conclusion")

    st.write(""" Ce projet nous a permis de mettre en pratique plusieurs modèles d’intelligence artificiel. C'est une discipline qui nécessite de tester le maximum de modèles possible avec divers paramètres.
Concretement, les methodes d'ensemble (Bagging Classifier, XGBoost) nous ont données résultats encourageants. Techniquement les temps d'apprentissage de ces modèles prennent du temps si on veut affiner le résultats et nous avons fait face aux limites de nos matériels.
Concernant l'iterprétabilité des résultats, nous avons pu mettre en valeurs certaines variables déterminantes : 
 - l'utilisation de la ceinture de sécurité, 
 - la vitesse maximale autorisée, 
 - la catégorie du véhicule, 
 - la place occupée dans le véhicule. 

En expérimentant ces techniques familières aux Datascientists, et compte tenu de la particularité de notre dataset « Accidents routiers en France », 
nous avons obtenu un modèle robuste, mais il est bien sûr perfectible. Aussi, une meilleure connaissance métier, par un expert en assurance par exemple,
apporterait sans doute une orientation plus judicieuse dans les choix des modèles de même qu'une vision et qu'une interprétation réaliste.""")


# Page Routing
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