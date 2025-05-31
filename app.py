import numpy as np
import streamlit as st
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraudes Bancaires",
    page_icon='🏦',
    layout='wide', # Utilisez 'wide' pour une mise en page plus spacieuse
    initial_sidebar_state='expanded'
)

# Initialiser les variables globales
scaler = None
pca = None
model = None
type_frequency_map = None

# Fonction de chargement des modèles
@st.cache_resource # Cache les modèles pour ne les charger qu'une seule fois
@st.cache_resource
def load_artifacts():
    try:
        current_dir = os.path.dirname(__file__)
        model_path = os.path.join(current_dir, 'model_SVM.pkl')
        scaler_path = os.path.join(current_dir, 'scaler.pkl')
        pca_path = os.path.join(current_dir, 'pca.pkl')
        freq_map_path = os.path.join(current_dir, 'type_frequency_map.pkl')
        
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        with open(pca_path, 'rb') as file:
            pca = pickle.load(file)
        with open(freq_map_path, 'rb') as file:
            type_frequency_map = pickle.load(file)
            
        return model, scaler, pca,type_frequency_map
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None, None, None, None

model, scaler, pca,type_frequency_map = load_artifacts()


# Chargement des modèles et des préprocesseurs
# Utilisez st.cache_resource pour charger les modèles une seule fois et améliorer la performance
#@st.cache_resource
#def load_resources():
#    with open("model_SVM.pkl","rb") as file:
#        model = pickle.load(file)
#    with open("scaler.pkl", "rb") as file:
#        scaler = pickle.load(file)
#    with open("pca.pkl", "rb") as file:
#        pca = pickle.load(file)
#    return model, scaler, pca

#model, scaler, pca = load_resources()

# --- Style CSS personnalisé ---
st.markdown("""
    <style>
    /* Importation d'une police Google Fonts pour un look plus moderne */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Roboto', sans-serif;
    }

    /* Style général du corps de l'application */
    .main .block-container {
        padding-top: 3rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 3rem;
    }

    /* En-tête de la barre latérale */
    .sidebar .sidebar-content {
        background-color: #f0f2f6; /* Couleur de fond légèrement différente pour la sidebar */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-1lcbm56 { /* Ciblage de l'en-tête de la sidebar */
        font-size: 1.8rem;
        color: #1a1a2e; /* Couleur de texte plus foncée */
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* Boutons de la barre latérale */
    .stButton > button {
        width: 100%;
        padding: 12px 20px;
        margin-bottom: 10px;
        border-radius: 8px;
        border: none;
        background-color: #007bff; /* Bleu primaire */
        color: white;
        font-size: 1rem;
        font-weight: 500;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #0056b3; /* Bleu plus foncé au survol */
    }

    /* Titres de sections */
    h1 {
        color: #333333;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 10px;
    }
    h2 {
        color: #444444;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    h3 {
        color: #555555;
        font-weight: 500;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    /* Messages de succès/erreur/info */
    .stAlert {
        border-radius: 8px;
    }
    .st-emotion-cache-1ldf3d0.e1gfxi8q1 { /* succès */
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    .st-emotion-cache-1ldf3d0.e1gfxi8q2 { /* erreur */
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
    .st-emotion-cache-1ldf3d0.e1gfxi8q3 { /* info */
        background-color: #d1ecf1;
        color: #0c5460;
        border-color: #bee5eb;
    }

    /* Cadre de contact */
    .contact-box {
        background-color: #e9ecef; /* Couleur de fond douce */
        padding: 30px;
        border-radius: 12px;
        margin-top: 3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 5px solid #007bff; /* Bordure colorée pour l'accent */
    }
    .contact-box h3 {
        color: #007bff;
        text-align: center;
        margin-bottom: 20px;
    }
    .contact-box ul {
        list-style: none;
        padding: 0;
        text-align: center;
    }
    .contact-box li {
        margin-bottom: 10px;
        font-size: 1.1rem;
        color: #555555;
    }
    .contact-box strong {
        color: #333333;
    }

    /* Amélioration des inputs */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #ced4da;
        padding: 10px;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.075);
    }
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }

    /* Spinner */
    .stSpinner > div > div {
        color: #007bff;
    }

    /* Styles pour la jauge de précision sur la page d'accueil */
    .progress-container-accueil {
        width: 80%;
        background-color: #e0e0e0;
        border-radius: 25px;
        overflow: hidden;
        margin: 30px auto 10px auto; /* Marge ajustée */
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }
    .progress-bar-accueil {
        height: 35px; /* Hauteur légèrement augmentée */
        background-color: #28a745; /* Vert pour la précision */
        border-radius: 25px;
        text-align: center;
        color: white;
        line-height: 35px;
        font-weight: bold;
        font-size: 1.3em; /* Taille de police plus grande */
        transition: width 0.5s ease-in-out;
    }
    .accuracy-text {
        text-align:center;
        font-size:1.1em;
        color:#555;
        margin-top: 10px;
    }
    
        .welcome-container {
        background-color: #f8f8f8; /* Légère couleur de fond pour le cadre */
        border: 2px solid #e0e0e0; /* Bordure discrète */
        border-radius: 15px; /* Coins arrondis */
        padding: 30px; /* Espace interne */
        margin-bottom: 2rem; /* Marge en dessous pour séparer des autres éléments */
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Ombre légère pour un effet 3D */
        text-align: center; /* Centrer le contenu à l'intérieur */
    }

    /* Styles pour le titre H1 à l'intérieur du conteneur, en conservant votre couleur marron */
    .welcome-container h1 {
        color: brown; /* Votre couleur conservée */
        font-family: 'Times New Roman', serif; /* Votre police conservée */
        font-size: 2.5em; /* Taille de police légèrement ajustée pour le titre */
        margin-bottom: 0.5em; /* Marge en bas du titre */
        padding-bottom: 0; /* Supprimer le padding bottom si déjà dans le H1 */
        border-bottom: none; /* Supprimer la bordure du H1 si elle existe */
    }

    /* Style pour le sous-titre h4 */
    .welcome-container h4 {
        color: #555555; /* Couleur de texte plus douce pour le sous-titre */
        font-size: 1.2em;
        margin-top: 0;
    }


    </style>
    """, unsafe_allow_html=True)

# Barre latérale
st.sidebar.title("🏦Détection de Fraudes Bancaires")
st.sidebar.image("bank_logo.jpg", use_container_width=True)

# Navigation via des boutons radio pour un style plus propre et une meilleure UX
page = st.sidebar.radio(
    "Naviguez entre les sections :",
    ('Accueil 🤗', 'Visualisation 📊', 'Prédiction 🏦'),
    key='main_navigation'
)

# --- Contenu des pages ---

if page == 'Accueil 🤗':
    with st.container(border=False): # border=False si vous gerez la bordure via CSS
        st.markdown("""
            <div class="welcome-container">
                <h1>Bienvenue sur l'application de Détection de Fraudes Bancaires 💸</h1>
            </div>
        """, unsafe_allow_html=True)
    col1, col2= st.columns([1,3])
    with col1:
        st.image("Hello.gif",width=200,use_container_width=True)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        #st.image("bank-icon.jpg", width=250,use_container_width=True)
        # Utilisez un conteneur pour le texte d'accueil
    
        #st.markdown("<h1 style='color:brown;font-family:'Times New Roman';text-align:center;'>Bienvenue sur l'application de Détection De Fraudes Bancaires🤑</h1>",unsafe_allow_html=True)
        #st.title("Bienvenue sur l'application de Détection de Fraudes Bancaires")
        st.markdown("""
            <h4 style='text-align:center;'>
            Cette application vous offre des outils puissants pour analyser et prédire les transactions frauduleuses.🚨
            </h4>
        """, unsafe_allow_html=True)

    # Crée deux colonnes avec des largeurs relatives
    # Ici, la première colonne (col1) sera 3 fois plus large que la deuxième (col2)
   
    
    #st.image('bank-icon.jpg',width=200) # Utilisation de la largeur de la colonne

    st.markdown("---") # Séparateur visuel

    st.subheader("Ce que vous pouvez faire avec cette application :")
    st.markdown("""
    - **Charger vos données**🗒 : Importez votre jeu de données de transactions bancaires au format CSV pour une analyse approfondie.
    - **Visualiser les données**📊 : Explorez les caractéristiques de vos transactions grâce à des graphiques interactifs et des statistiques descriptives.
    - **Prédire la fraude**🚨 : Utilisez notre modèle pré-entraîné pour évaluer le risque de fraude sur de nouvelles transactions en entrant simplement les informations requises.
    """)

    st.info("""
        **Note :** Les sections de chargement de données et de prédiction fonctionnent indépendamment.
        Vous n'avez pas besoin de charger des données pour effectuer une prédiction individuelle.
    """)

    st.markdown("---")

    ## --- Affichage de la précision du modèle sur la page d'accueil ---
    st.subheader("Performance Générale du Modèle")
    st.markdown("""
        <p style='text-align:center; font-size:1.1em; color:#555;'>
            Notre modèle de détection de fraude a été rigoureusement testé et démontre une excellente capacité
            à identifier les transactions frauduleuses.
        </p>
    """, unsafe_allow_html=True)

    # VOUS POUVEZ CHANGER CETTE VALEUR DE PRÉCISION ICI :
    model_accuracy_accueil = 0.989  # <--- METTEZ LA VALEUR DE PRÉCISION QUE VOUS VOULEZ AFFICHER (entre 0 et 1)

    accuracy_percent_accueil = int(model_accuracy_accueil * 100)

    st.markdown(f"""
        <div class="progress-container-accueil">
            <div class="progress-bar-accueil" style="width: {accuracy_percent_accueil}%;">
                {accuracy_percent_accueil}%
            </div>
        </div>
        <p class="accuracy-text">
            Ce modèle atteint une précision remarquable de <strong>{model_accuracy_accueil:.1%}</strong>
            sur nos jeux de données de test.
        </p>
    """, unsafe_allow_html=True)
    ## --- Fin de l'affichage de la précision ---

    # Section Contact
    st.markdown("""
        <div class="contact-box">
            <h3>Contactez-nous</h3>
            <p>Si vous avez des questions ou souhaitez en savoir plus, n'hésitez pas à nous contacter :</p>
            <ul>
                <li><strong>Email :</strong> ccmlav@gmail.com</li>
                <li><strong>Téléphone :</strong> +237 000000000</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

elif page == 'Visualisation 📊':
    st.title('📈 Analyse Exploratoire et Visualisation')
    st.markdown("---")
    st.image('vis.jpg',width=500)
    st.markdown("Explorez les types de données, le nombre de valeurs non-nulles et la présence de données manquantes pour chaque colonne.")

    st.markdown("<h3 style='text-align:center;'>1. Chargement de votre fichier de données (CSV)</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Glissez-déposez un fichier CSV ici ou cliquez pour le parcourir", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df # Stocker le DataFrame dans session_state
            st.success("Fichier chargé avec succès ! ✅")

            st.subheader("Aperçu des Données")
            st.dataframe(df.head())

            st.subheader("📊 Informations Générales sur les Données")
            # ... (code de chargement du fichier CSV et aperçu du DF) ...

            #st.subheader("📊 Informations Détaillées sur les Données")

            # Créer un DataFrame pour afficher les informations clés
            info_df = pd.DataFrame({
                'Colonne': df.columns,
                'Type de Données': df.dtypes,
                'Valeurs Non-Nulles': df.count(),
                'Valeurs Manquantes': df.isnull().sum(),
                'Pourcentage Manquant (%)': (df.isnull().sum() / len(df) * 100).round(2)
            })

            # Appliquer des couleurs conditionnelles pour le pourcentage manquant
            def highlight_missing(s):
                return ['background-color: #ffe6e6' if s['Pourcentage Manquant (%)'] > 0 else '' for _ in s] # Rouge pâle pour les manquants

            st.dataframe(info_df.style.apply(highlight_missing, axis=1), use_container_width=True)

            st.info(f"Le jeu de données contient **{len(df)} lignes** et **{len(df.columns)} colonnes** au total.")

            # Optionnel : Afficher l'output brut de df.info() dans un expander pour les détails techniques
            with st.expander("Voir l'output technique complet de df.info()"):
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.code(buffer.getvalue(), language='text') # Utiliser st.code pour un meilleur formatage du texte brut


            st.subheader("Statistiques Descriptives")
            st.dataframe(df.describe())

            st.markdown("---")
            st.markdown("<h3 style='text-align:center;'>2. Analyse Exploratoire des Données (EDA)</h3>", unsafe_allow_html=True)
            st.markdown("Utilisez les options ci-dessous pour générer des graphiques et mieux comprendre vos données.")

            with st.expander("Générer des graphiques interactifs", expanded=True): # Ouvrir l'expander par défaut
                eda_col1, eda_col2 = st.columns(2)

                with eda_col1:
                    chart_type = st.selectbox(
                        "Choisissez le type de graphique :",
                        ["Histogramme", "Boîte à Moustaches", "Nuage de Points", "Comptage"],
                        key='chart_selector'
                    )
                    selected_column_x = st.selectbox(
                        "Choisissez la colonne pour l'axe X :",
                        df.columns.tolist(),
                        key='column_x_selector'
                    )

                with eda_col2:
                    selected_column_y = None
                    if chart_type == "Nuage de Points":
                        selected_column_y = st.selectbox(
                            "Choisissez la colonne pour l'axe Y :",
                            [col for col in df.columns if col != selected_column_x],
                            key='column_y_selector'
                        )
                    color_by = st.checkbox("Colorer par 'isFraud' (si disponible)", value=True)
                    if 'isFraud' not in df.columns:
                        color_by = False
                        st.warning("La colonne 'isFraud' n'est pas présente dans votre dataset pour la coloration.")

                if st.button("Générer le graphique 📈", key='generate_plot_button'):
                    with st.spinner("Génération du graphique en cours..."):
                        fig, ax = plt.subplots(figsize=(12, 7)) # Taille de figure améliorée

                        if chart_type == "Histogramme":
                            sns.histplot(df[selected_column_x], kde=True, ax=ax, color='#28a745') # Couleur plus esthétique
                            ax.set_title(f"Distribution de '{selected_column_x}'", fontsize=16)
                            ax.set_xlabel(selected_column_x, fontsize=12)
                            ax.set_ylabel("Fréquence", fontsize=12)
                        elif chart_type == "Boîte à Moustaches":
                            sns.boxplot(y=df[selected_column_x], ax=ax, color='#ffc107')
                            ax.set_title(f"Boîte à Moustaches de '{selected_column_x}'", fontsize=16)
                            ax.set_ylabel(selected_column_x, fontsize=12)
                        elif chart_type == "Nuage de Points" and selected_column_y:
                            if color_by and 'isFraud' in df.columns:
                                sns.scatterplot(x=df[selected_column_x], y=df[selected_column_y], hue=df['isFraud'], ax=ax, palette='viridis', alpha=0.7)
                            else:
                                sns.scatterplot(x=df[selected_column_x], y=df[selected_column_y], ax=ax, color='#17a2b8', alpha=0.7)
                            ax.set_title(f"Nuage de Points : '{selected_column_x}' vs '{selected_column_y}'", fontsize=16)
                            ax.set_xlabel(selected_column_x, fontsize=12)
                            ax.set_ylabel(selected_column_y, fontsize=12)
                        elif chart_type == "Comptage":
                            sns.countplot(x=df[selected_column_x], ax=ax, palette='coolwarm')
                            ax.set_title(f"Comptage des valeurs de '{selected_column_x}'", fontsize=16)
                            ax.set_xlabel(selected_column_x, fontsize=12)
                            ax.set_ylabel("Nombre d'occurrences", fontsize=12)
                            plt.xticks(rotation=45, ha='right') # Rotation pour les étiquettes longues

                        plt.tight_layout() # Ajustement automatique pour éviter les chevauchements
                        st.pyplot(fig)
                        st.balloons()
                        st.success("Graphique généré avec succès ! ✨")
        except Exception as e:
            st.error(f"Une erreur est survenue lors du chargement ou de l'analyse du fichier : {e}")
            st.info("Veuillez vous assurer que le fichier est un CSV valide et correctement formaté.")
    else:
        st.info("Veuillez charger un fichier CSV pour commencer l'exploration des données.")

elif page == 'Prédiction 🏦':
    st.image("th (1).jpg",width=200) # Assurez-vous que l'image est dans le même répertoire
    st.title('💸 Prédiction de Fraudes Bancaires')
    st.markdown("---")
    
    # La disposition des images et texte d'introduction
    col_intro1, col_intro2 = st.columns([3,1])
    with col_intro2:
        st.image('fraud2.jpg', width=500) # Assurez-vous que l'image est dans le même répertoire
    
    with col_intro1:
        st.markdown('Remplissez les informations ci-dessous pour déterminer si la transaction est frauduleuse.')
        
        # Afficher le statut de chargement du modèle
        if model is not None:
            st.success('Modèle prêt !')
        else:
            st.error('Modèle non chargé. Impossible de faire des prédictions.')
            st.stop() # Arrêter l'exécution si le modèle n'est pas disponible

    
    # --- Section de Détails de la Transaction pour la prédiction -----------------

    # Section de prédiction
    st.subheader("Détails de la Transaction")
    col_pred1, col_pred2 = st.columns(2)

    with col_pred1:
        step = st.number_input("Step (Heure de la transaction en heures)", min_value=0, value=1)
        amount = st.number_input("Amount (Montant de la transaction)", min_value=0.0, value=1000.0, format="%.2f")
        oldbalanceOrg = st.number_input("Ancien solde de l'initiateur", min_value=0.0, value=10000.0, format="%.2f")
        
    with col_pred2:
        newbalanceOrig = st.number_input("Nouveau solde de l'initiateur", min_value=0.0, value=9000.0, format="%.2f")
        oldbalanceDest = st.number_input("Ancien solde du destinataire", min_value=0.0, value=5000.0, format="%.2f")
        # Dans votre section de prédiction :
        if type_frequency_map is None:
            st.error("Le mapping des types de transactions n'a pas pu être chargé. Impossible de continuer.")
            st.stop()

        Type = st.selectbox(
            "Type de transaction",
            list(type_frequency_map.keys()) if type_frequency_map else [],
            key="pred_type"
        )

    type_encoded = type_frequency_map.get(Type, 0)

    if st.button('💸 Prédire la Fraude'):
        # 1. Créez le DataFrame d'entrée avec les mêmes colonnes qu'à l'entraînement
        input_df = pd.DataFrame([[
            step,
            amount,                
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,       
            type_encoded         
        ]], columns=[
            'step', 
            'amount',            
            'oldbalanceOrg', 
            'newbalanceOrig',
            'oldbalanceDest',    
            'type_encoded'       
        ])
        
        # Appliquez les transformations COMME PENDANT L'ENTRAÎNEMENT
        input_df['amount'] = np.log1p(input_df['amount'])  
        
        # 3. Sélectionnez les colonnes dans le bon ordre (comme pendant l'entraînement)
        features_to_scale = input_df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig','oldbalanceDest','type_encoded']]
        
        # Créez une liste des features dans L'ORDRE EXACT de l'entraînement
        feature_order = [
            'step',
            'amount',          
            'oldbalanceOrg',
            'newbalanceOrig',
            'oldbalanceDest',
            'type_encoded',
        ]
        features_for_prediction = input_df[feature_order]
        
        with st.spinner('Analyse de la transaction en cours...'):
            try:
                scaled_features = scaler.transform(features_to_scale)
            
                # 5. Application du PCA
                pca_features = pca.transform(scaled_features)
            
                # Prédiction directe
                prediction = model.predict(pca_features)[0]
                decision_score = model.decision_function(pca_features)[0]
                time.sleep(1)

                st.success('Prédiction terminée ! 🤗')

                if prediction == 1:
                    st.error("🚨 **Transaction frauduleuse détectée !** Veuillez enquêter.")
                    st.write(f"**Score de décision :** `{decision_score:.4f}`")
                    st.image("https://media.giphy.com/media/l0Hea3gqR4m3s8Q0/giphy.gif", width=200)
                else:
                    st.success("✅ **Transaction non frauduleuse.** Tout semble normal.")
                    st.write(f"**Score de décision :** `{decision_score:.4f}`")
                    st.balloons()
    
 
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.warning("Vérifiez les valeurs entrées.")
                st.write("Features input:", features_to_scale)  
        
st.markdown("---")
