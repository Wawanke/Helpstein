from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr

app = Flask(__name__)

# === Chargement des données depuis le fichier CSV ===
file_path = "Csv/data_villes_france.csv"
data = pd.read_csv(file_path)

# Renommer les colonnes pour correspondre aux attentes du code
column_mapping = {
    "Qualité de Vie (Score)": "Qualité de Vie",
    "Sécurité (Score)": "Sécurité",
    "Infrastructures (Score)": "Infrastructures"
}
data.rename(columns=column_mapping, inplace=True)

# === Modèle de Régression Linéaire ===
X = data[["Coût de la Vie (€)", "Sécurité", "Infrastructures"]]
y = data["Qualité de Vie"]
model = LinearRegression()
model.fit(X, y)

# === Moteur de Recherche ===
corpus = data["Description"].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus).toarray()


# === Routes Flask ===
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    statut = request.form.get("statut", "solo")
    enfants = request.form.get("enfants", "0")
    if enfants == "5":
        enfants = 5  # On considère "5 et +" comme 5 enfants
    else:
        enfants = int(enfants)

    if request.form.get("revenu", "1426,30").strip() == "" :
        return render_template('problem.html')
    revenu_str = request.form.get("revenu", "1426,30").strip()
    revenu = 1426,30  # Valeur par défaut
    if revenu_str.isdigit():
        revenu = max(int(revenu_str), 1426,30)

    # Ajustement du revenu en fonction du statut marital
    if statut == "couple":
        revenu *= 1.5  # Supposition qu'un couple a 50% de revenu en plus

    # Simulation de calcul prenant en compte le nombre d'enfants
    input_data = pd.DataFrame([[revenu - (enfants * 680), 7, 8]],
                              columns=["Coût de la Vie (€)", "Sécurité", "Infrastructures"])
    prediction = model.predict(input_data)[0]

    # Trouver les 10 villes les plus proches du score prédit
    data["Ecart"] = abs(data["Qualité de Vie"] - prediction)
    villes_selectionnees = data.nsmallest(10, "Ecart")[
        ["Ville", "Coût de la Vie (€)", "Qualité de Vie", "Sécurité", "Infrastructures", "Description"]]

    return render_template('result.html', villes=villes_selectionnees.to_dict(orient='records'))


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get("query", "").strip()
    if not query:
        return render_template('search_results.html', results=[])

    query_vector = vectorizer.transform([query]).toarray()[0]
    results = []

    for i, doc_vector in enumerate(tfidf_matrix):
        similarity, _ = pearsonr(query_vector, doc_vector)
        if similarity > 0.1:
            results.append({"ville": data["Ville"][i], "description": corpus[i], "score": round(similarity, 2)})

    return render_template('search_results.html', results=results)


# === Lancement de l'application ===
if __name__ == '__main__':
    app.run(debug=True)
