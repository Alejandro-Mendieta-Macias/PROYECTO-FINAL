import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset
file_path = r"C:\\Users\\alexm\\OneDrive\\Escritorio\\SIAFI WORK\\Proyecto Final\\vgchartz-2024.csv"  # Ruta al archivo CSV
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print("Asegúrate de tener el dataset descargado y la ruta correcta.")
    exit()

# 2. Inspeccionar los datos
print(data.head())
print(data.info())

# 3. Limpieza de datos
# Rellenar valores nulos en columnas de ventas con 0
sales_columns = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales']
data[sales_columns] = data[sales_columns].fillna(0)

# Crear una nueva columna 'calculated_total_sales' sumando las ventas regionales
data['calculated_total_sales'] = data[sales_columns].sum(axis=1)

# Definir función para clasificar la popularidad
def popularity_category(row):
    if row['calculated_total_sales'] > 10:
        return 'Muy populares'
    elif 5 <= row['calculated_total_sales'] <= 10:
        return 'Moderadamente populares'
    else:
        return 'Menos populares'

# Aplicar la función para clasificar los videojuegos
data['Popularity'] = data.apply(popularity_category, axis=1)

# Mapear categorías a valores numéricos para facilitar el análisis
data['Popularity'] = data['Popularity'].map({
    'Muy populares': 2,
    'Moderadamente populares': 1,
    'Menos populares': 0
})

# Seleccionar características relevantes
features = ['na_sales', 'jp_sales', 'pal_sales', 'other_sales', 'critic_score']  # Características importantes
X = data[features].fillna(0)  # Rellenar valores nulos si es necesario
y = data['Popularity']

# 4. Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# 6. Evaluación
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Importancia de características
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Importancia de las características")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.show()

# Visualización de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Matriz de confusión")
plt.show()
