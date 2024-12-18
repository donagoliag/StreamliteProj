import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger la base de données Iris
from sklearn.datasets import load_iris
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# Titre de l'application
st.title("Analyse des données IRIS")


st.header("1. Aperçu et manipulation des données")
st.write(iris.head())


st.sidebar.header("Filtres")
selected_species = st.sidebar.multiselect(
    "Choisissez une ou plusieurs espèces :", 
    options=list(iris['species'].unique()),  # Convertir en liste
    default=list(iris['species'].unique())   # Convertir en liste
)


filtered_data = iris[iris['species'].isin(selected_species)]

st.subheader("Données filtrées")
st.write(filtered_data)

st.subheader("Statistiques descriptives")
st.write(filtered_data.describe())

st.header("2. Graphiques")

st.subheader("Histogramme")
hist_variable = st.selectbox("Choisissez une variable pour l'histogramme :", options=iris.columns[:-1])
fig, ax = plt.subplots()
sns.histplot(data=filtered_data, x=hist_variable, hue="species", kde=True, ax=ax, palette="pastel")
ax.set_title(f"Histogramme de {hist_variable}")
st.pyplot(fig)

st.subheader("Graphique de dispersion")
x_axis = st.selectbox("Choisissez la variable pour l'axe des X :", options=iris.columns[:-1])
y_axis = st.selectbox("Choisissez la variable pour l'axe des Y :", options=iris.columns[:-1])

fig, ax = plt.subplots()
sns.scatterplot(data=filtered_data, x=x_axis, y=y_axis, hue="species", palette="deep", ax=ax)
ax.set_title(f"Scatterplot : {x_axis} vs {y_axis}")
st.pyplot(fig)

st.subheader("Boxplot")
box_variable = st.selectbox("Choisissez une variable pour le boxplot :", options=iris.columns[:-1])
fig, ax = plt.subplots()
sns.boxplot(data=filtered_data, x="species", y=box_variable, palette="muted", ax=ax)
ax.set_title(f"Boxplot de {box_variable} par espèce")
st.pyplot(fig)


