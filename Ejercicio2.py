import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Paso 1: Crear un dataset de frases y emojis
data = {
    "frase": [
        "Estoy muy feliz hoy",
        "Tengo hambre",
        "Me siento triste",
        "Es un día increíble",
        "Quiero comer pizza",
        "Estoy cansado",
        "El clima es hermoso",
        "Necesito dormir",
        "Esto es muy divertido",
        "Estoy molesto",
    ],
    "emoji": ["😊", "🍔", "😢", "😊", "🍕", "😴", "🌞", "😴", "😂", "😡"],
}

df = pd.DataFrame(data)

# Paso 2: Representar las frases como vectores utilizando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["frase"])  # Convertir frases en vectores TF-IDF
y = df["emoji"]  # Etiquetas (emojis)

# Paso 3: Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paso 4: Aplicar k-Nearest Neighbors (k-NN)
k = 3  # Número de vecinos
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Paso 5: Hacer predicciones
y_pred = model.predict(X_test)

# Paso 6: Evaluar el modelo
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred))

print("Precisión General:", accuracy_score(y_test, y_pred))

# Paso 7: Mostrar frases y predicciones incorrectas
test_phrases = vectorizer.inverse_transform(X_test)
results = pd.DataFrame({"Frase": [" ".join(phrase) for phrase in test_phrases], "Real": y_test, "Predicho": y_pred})
incorrect = results[results["Real"] != results["Predicho"]]

print("\nPredicciones Incorrectas:")
print(incorrect)

# Paso 8: Visualización (opcional)
print("\nFrases con sus emojis:")
for i, row in results.iterrows():
print(f"Frase: {row['Frase']} -> Real: {row['Real']} | Predicho: {row['Predicho']}")
