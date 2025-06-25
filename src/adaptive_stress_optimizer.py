# -*- coding: utf-8 -*-
"""
adaptive_stress_optimizer.py

Script principal para el análisis, predicción y optimización del estrés
de una turbina de gas naval.

Ejecución:
    python src/adaptive_stress_optimizer.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pygad

# --- Configuraciones Globales ---
sns.set_style('whitegrid')
plt.rc('figure', figsize=(12, 7))

# --- Rutas de Archivos ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# --- Parámetros del Modelo ---
TARGET_SPEED = 12.0  # Nudos


def load_and_clean_data(filepath):
    """Carga y limpia los datos del motor desde un CSV."""
    print("--- Fase 1: Cargando y Limpiando Datos ---")
    if not os.path.exists(filepath):
        print(f"Error: El archivo de datos no se encontró en {filepath}")
        return None
    df = pd.read_csv(filepath)
    # Proceso de limpieza de nombres de columna
    df.columns = [
        col.strip()
        .replace(' ', '_')
        .replace('.', '')
        .replace('(kMc)', '')
        .replace('(kMt)', '')
        for col in df.columns
    ]
    df = df.dropna(axis=1, how='all').dropna(axis=0)
    print("Datos cargados exitosamente.")
    return df


def engineer_features(df):
    """Crea el Stress_Index unificado y visualiza la degradación."""
    print("\n--- Fase 2: Creando el Índice de Estrés ---")
    df['Compressor_Degradation'] = 1 - df['GT_Compressor_decay_state_coefficient']
    df['Turbine_Degradation'] = 1 - df['GT_Turbine_decay_state_coefficient']
    df['Stress_Index'] = np.sqrt(df['Compressor_Degradation'] ** 2 + df['Turbine_Degradation'] ** 2)
    print("Stress_Index creado.")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Compressor_Degradation', y='Turbine_Degradation', hue='Stress_Index', palette='viridis',
                    s=50)
    plt.title('Mapa de Degradación vs. Índice de Estrés Combinado')
    plt.xlabel('Degradación del Compresor (1 - kMc)')
    plt.ylabel('Degradación de la Turbina (1 - kMt)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, 'degradation_map.png'), dpi=300)
    plt.show()
    return df


def prepare_model_data(df):
    """
    Prepara y escala los datos para el entrenamiento del modelo.
    Esta función ha sido modificada para buscar columnas dinámicamente y evitar KeyErrors.
    """
    print("\n--- Fase 3: Preparando Datos para el Modelo (Búsqueda Dinámica de Columnas) ---")

    # Palabras clave para encontrar cada columna necesaria
    feature_keywords = {
        'Lever_position': 'Lever_position',
        'Ship_speed': 'Ship_speed',
        'Shaft_torque': 'GTT',
        'Generator_speed': 'GGn',
        'Fuel_injection': 'TIC',
        'Exhaust_temp': 'T48',
        'Compressor_pressure': 'P2',
        'Fuel_flow': 'Fuel_flow'
    }

    actual_feature_cols = []
    print("Buscando columnas de features requeridas...")
    for key, keyword in feature_keywords.items():
        # Busca columnas que contengan la palabra clave
        matching_cols = [col for col in df.columns if keyword in col]
        if len(matching_cols) == 1:
            found_col = matching_cols[0]
            actual_feature_cols.append(found_col)
            print(f"  - '{keyword}' -> Encontrada: '{found_col}'")
        elif len(matching_cols) > 1:
            # Si hay múltiples coincidencias, intenta una más específica
            specific_match = [col for col in matching_cols if keyword + '_' in col]
            if len(specific_match) == 1:
                actual_feature_cols.append(specific_match[0])
                print(f"  - '{keyword}' -> Encontrada (con desambiguación): '{specific_match[0]}'")
            else:
                raise ValueError(
                    f"Ambigüedad: La palabra clave '{keyword}' coincide con múltiples columnas: {matching_cols}. No se pudo resolver.")
        else:
            raise KeyError(
                f"Error Crítico: No se pudo encontrar ninguna columna que contenga la palabra clave '{keyword}'. Revisa tu archivo CSV.")

    X = df[actual_feature_cols]
    y = df['Stress_Index']

    # Renombra las columnas a nombres consistentes para el resto del script
    X.columns = list(feature_keywords.keys())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nDatos listos y escalados exitosamente.")
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate_model(X_train_scaled, y_train, X_test_scaled, y_test, X_cols):
    """Entrena y evalúa el modelo de Gradient Boosting."""
    print("\n--- Fase 4: Entrenando y Evaluando el Modelo Predictivo ---")
    gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.2, max_depth=10, random_state=42)
    gbr.fit(X_train_scaled, y_train)

    y_pred = gbr.predict(X_test_scaled)
    r2, mae = r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred)
    print(f"Resultados de Evaluación: R² = {r2:.4f}, MAE = {mae:.4f}")

    feature_importances = pd.Series(gbr.feature_importances_, index=X_cols).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index)
    plt.title('Importancia de Features para Predecir el Estrés del Motor')
    plt.xlabel('Importancia Relativa')
    plt.ylabel('Feature Operativa')
    plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    return gbr


def optimize_profile(model, scaler, X_features, target_speed):
    """Usa un Algoritmo Genético para encontrar el perfil operativo óptimo."""
    print(f"\n--- Fase 5: Optimizando Perfil para {target_speed} Nudos ---")
    speed_col_name = 'Ship_speed'
    speed_col_index = list(X_features.columns).index(speed_col_name)

    def fitness_func(ga_instance, solution, solution_idx):
        solution_copy = solution.copy()
        solution_copy[speed_col_index] = target_speed
        solution_scaled = scaler.transform([solution_copy])
        predicted_stress = model.predict(solution_scaled)[0]
        return -predicted_stress

    gene_space = [{'low': X_features[col].min(), 'high': X_features[col].max()} if col != speed_col_name else {
        'low': target_speed, 'high': target_speed} for col in X_features.columns]

    ga_instance = pygad.GA(
        num_generations=100, num_parents_mating=10, sol_per_pop=50,
        num_genes=len(X_features.columns), fitness_func=fitness_func,
        gene_space=gene_space, gene_type=float, mutation_type="adaptive",
        mutation_percent_genes=[10, 2], stop_criteria="saturate_15"
    )
    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    optimal_stress = -best_fitness
    optimal_profile = pd.Series(best_solution, index=X_features.columns)
    optimal_profile[speed_col_name] = target_speed

    print("\n--- Perfil Óptimo de Mínimo Estrés Encontrado ---")
    print(f"Velocidad Objetivo: {target_speed} nudos")
    print(f"Índice de Estrés Mínimo Predicho: {optimal_stress:.6f}")
    print("\nParámetros Operativos Recomendados:")
    print(optimal_profile.round(4))

    # Guardar perfil óptimo en un archivo
    with open(os.path.join(RESULTS_DIR, 'optimal_profile.txt'), 'w') as f:
        f.write(f"Perfil Óptimo para {target_speed} Nudos\n")
        f.write(f"Índice de Estrés Mínimo Predicho: {optimal_stress:.6f}\n\n")
        f.write(optimal_profile.round(4).to_string())

    ga_instance.plot_fitness(title="Convergencia del Algoritmo Genético", xlabel="Generación",
                             ylabel="Fitness (Estrés Negativo)")
    plt.savefig(os.path.join(PLOTS_DIR, 'ga_convergence.png'), dpi=300)
    plt.show()


def main():
    """Función principal para ejecutar el pipeline completo."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df_raw = load_and_clean_data(DATA_PATH)
    if df_raw is None:
        return

    df_featured = engineer_features(df_raw)
    X, y, X_train_s, X_test_s, y_train, y_test, scaler = prepare_model_data(df_featured)
    model = train_and_evaluate_model(X_train_s, y_train, X_test_s, y_test, X.columns)
    optimize_profile(model, scaler, X, TARGET_SPEED)

    print("\n--- Análisis Completo ---")


if __name__ == '__main__':
    main()