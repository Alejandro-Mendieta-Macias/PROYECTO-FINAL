import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import chardet

# Ruta del archivo
dataset_path = r"C:\\Users\\alexm\\OneDrive\\Escritorio\\SIAFI WORK\\Proyecto Final\\spotify-2023.csv"

# Verificar si el archivo existe
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"El archivo no se encontró en la ruta especificada: {dataset_path}")
else:
    print("Archivo encontrado correctamente. Procediendo con la carga...")

# Detectar la codificación
with open(dataset_path, 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Codificación detectada: {encoding}")

# Leer el archivo con la codificación detectada
try:
    data = pd.read_csv(dataset_path, encoding=encoding)
    print("Datos cargados correctamente. Primeras filas:")
    print(data.head())
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    exit()

# Limpieza de datos
# Eliminar duplicados
data = data.drop_duplicates()

# Manejar valores faltantes
data = data.dropna()

# Convertir variables categóricas (si las hay)
data = pd.get_dummies(data, drop_first=True)

# Análisis exploratorio
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlación entre variables")
plt.show()

# Identificar las variables más influyentes
correlation_matrix = data.corr()
print("Correlaciones con la variable objetivo (streams):")
print(correlation_matrix['streams'].sort_values(ascending=False))

# Preparar los datos para el modelo
X = data.drop(columns=['streams'])  # Características
y = data['streams']  # Variable objetivo

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# Visualización de resultados
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.title("Valores Reales vs. Predichos")
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.show()

