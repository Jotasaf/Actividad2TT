import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Paso 1: Cargar la base de datos CSV
df = pd.read_csv('Calidad_Del_Aire_En_Colombia.csv', sep=',')

# Renombrar columnas para facilitar manejo
df.columns = [col.replace(' ', '_').replace('.', '').replace('(','').replace(')','').replace('/','') for col in df.columns]

# Paso 2: Filtrar solo registros del Valle de Aburrá
municipios_valle = [
    'MEDELLIN', 'BELLO', 'ITAGÜI', 'ENVIGADO',
    'SABANETA', 'LA ESTRELLA', 'CALDAS',
    'COPACABANA', 'GIRARDOTA', 'BARBOSA'
]

df_valle = df[df['Nombre_del_Municipio'].str.upper().isin(municipios_valle)]

# --- Función para análisis y modelado ---
def analizar_variable(df, variable):
    print(f"\n================ Análisis para {variable} ================")

    # Filtrar variable (PM2.5 o PM10)
    df_var = df[df['Variable'] == variable].copy()

    # Seleccionar columnas de interés
    df_model = df_var[['Promedio', 'Latitud', 'Longitud']]

    # Convertir "Promedio" a numérico (algunos tienen coma decimal)
    df_model['Promedio'] = pd.to_numeric(df_model['Promedio'].astype(str).str.replace(',', '.'), errors='coerce')
    df_model.dropna(inplace=True)

    # Crear variable derivada (ejemplo de Altitud)
    df_model['Altitud'] = df_model['Latitud'] * 100


    print("\nDataFrame para modelado después del preprocesamiento:")
    print(df_model.head())

    # Visualización: Dispersión
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Altitud', y='Promedio', data=df_model, alpha=0.6)
    plt.title(f'Relación entre Altitud y {variable} en el Valle de Aburrá')
    plt.xlabel('Altitud ')
    plt.ylabel(f'Promedio {variable} (ug/m3)')
    plt.show()

    # Visualización: Histograma
    plt.figure(figsize=(10, 6))
    sns.histplot(df_model['Promedio'], bins=30, kde=True)
    plt.title(f'Distribución del Promedio de {variable} en el Valle de Aburrá')
    plt.xlabel(f'Promedio {variable} (ug/m3)')
    plt.ylabel('Frecuencia')
    plt.show()

    # Preparación de datos
    X = df_model[['Altitud']]
    y = df_model['Promedio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Evaluación del modelo de Regresión Lineal ---")
    print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinación (R^2): {r2:.2f}")

    # Visualizar regresión
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Datos reales')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicciones del modelo')
    plt.title(f'Regresión Lineal: {variable} vs. Altitud (Valle de Aburrá)')
    plt.xlabel('Altitud')
    plt.ylabel(f'Promedio {variable} (ug/m3)')
    plt.legend()
    plt.show()

# --- Ejecutar análisis ---
analizar_variable(df_valle, 'PM2.5')
analizar_variable(df_valle, 'PM10')
