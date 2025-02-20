import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Interface Streamlit
st.title("Prédiction de l'accès aux services bancaires")
st.write("Entrez les caractéristiques d'une personne pour prédire si elle a un compte bancaire.")
# Display the dataframe

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('Financial_inclusion_dataset.csv')
    st.dataframe(df)

    # Supprimer les doublons
    df.drop_duplicates(inplace=True)

    # Encoder les variables catégorielles

    encoding_mappings = {}
    df = df.drop(['uniqueid'], axis=1)


    # Iterate trough columns
    for col in df.columns:
        if df[col].dtype == 'object':  # Verify if our columns type is object
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            # Store the mapping in the dictionary
            encoding_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            
    #encoding_mappings

    # Replace NaN values with '__'
    corresponding = pd.DataFrame(encoding_mappings)
    corresponding.fillna('__', inplace=True)
    st.dataframe(corresponding)

    # Supprimer les valeurs aberrantes
    for col in df.select_dtypes(include=np.number):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        
        return df

df = load_data()

# Définir les variables cibles et explicatives
X = df.drop(columns=['year', 'bank_account'])
y = df['bank_account']

 # Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

    # Séparation des données en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Entraînement du modèle
tree = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
tree.fit(X_train, y_train)


    # Formulaire utilisateur
user_input = {}
for col in df.drop(columns=['year', 'bank_account']).columns:
    user_input[col] = st.number_input(f"{col}")

    # Bouton de prédiction
if st.button("Prédire"):
    input_df = pd.DataFrame([user_input])
    input_df = scaler.transform(input_df)  # Normalisation
    
    prediction = tree.predict(input_df)

    st.subheader("Résultat de la Prédiction")
    if prediction[0] == 1:
            st.success("✅ Cette personne a un compte bancaire.")
    else:
        st.error("❌ Cette personne n'a pas de compte bancaire.")

        # Affichage des performances du modèle
    st.subheader("Évaluation du Modèle")
    accuracy = accuracy_score(y_test, tree.predict(X_test))
    st.write(f"**Précision du modèle:** {accuracy:.4f}")

        # Matrice de confusion
    cm = confusion_matrix(y_test, tree.predict(X_test))
    fig = px.imshow(cm, text_auto=True, labels=dict(x="Prédictions", y="Réelles"), color_continuous_scale="Blues")
    st.plotly_chart(fig)

        # Rapport de classification
    st.text("Rapport de classification :")
    st.text(classification_report(y_test, tree.predict(X_test)))
