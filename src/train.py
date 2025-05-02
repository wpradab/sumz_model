import pandas as pd
import pickle
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def preprocesar_datos_y_guardar(catalogo_producto, demanda):
    # Definir las columnas requeridas
    columnas_requeridas = ['id_producto', 'categoria', 'subcategoria', 'tamaño',
                           'premium', 'marca_exclusiva', 'estacional']

    # Filtrar solo las columnas requeridas
    df_cat_producto = catalogo_producto[columnas_requeridas].copy()

    # Llenar valores faltantes en 'subcategoria'
    df_cat_producto['subcategoria'] = df_cat_producto['subcategoria'].fillna('no_categoria')

    # Llenar valores faltantes en 'premium' con la moda
    premium_mode = df_cat_producto['premium'].mode()[0]
    df_cat_producto['premium'] = df_cat_producto['premium'].fillna(premium_mode)

    # Llenar valores faltantes en 'tamaño'
    df_cat_producto['tamaño'] = df_cat_producto['tamaño'].fillna('no_tamaño')

    # Codificar variables categóricas con one-hot encoding
    df_encoded = pd.get_dummies(df_cat_producto, columns=['categoria', 'tamaño'],
                                prefix=['cat', 'tam'], drop_first=True, dtype=float)

    # Encoding de frecuencia para 'subcategoria'
    frequency_encoding = df_cat_producto['subcategoria'].value_counts(normalize=True)
    df_encoded['subcategoria'] = df_cat_producto['subcategoria'].map(frequency_encoding)

    # Procesar demanda
    df_demanda = demanda.copy()
    df_demanda['date'] = pd.to_datetime(df_demanda['date'])
    df_demanda['month'] = df_demanda['date'].dt.month
    df_demanda['day'] = df_demanda['date'].dt.day
    df_demanda = df_demanda.drop('date', axis=1)

    # Combinar ambos dataframes por 'id_producto'
    df_final = pd.merge(df_encoded, df_demanda, on='id_producto', how='inner')

    # Definir la ruta al directorio models (un nivel arriba de src/)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    # Crear carpeta si no existe
    os.makedirs(models_dir, exist_ok=True)

    # Guardar las columnas de df_final en un archivo pkl
    columnas_path = os.path.join(models_dir, 'columnas_df_final.pkl')
    with open(columnas_path, 'wb') as f:
        pickle.dump(df_final.columns.tolist(), f)

    # Guardar el diccionario de frequency_encoding
    freqenc_path = os.path.join(models_dir, 'frequency_encoding.pkl')
    with open(freqenc_path, 'wb') as f:
        pickle.dump(frequency_encoding.to_dict(), f)

    print(f"Columnas y frequency_encoding guardados en: {models_dir}")

    return df_final

def entrenamiento_demanda(df_final, target_col='demanda', n_splits=5, random_state=42):
    # Separar X y y
    X = df_final.drop(columns=[target_col])
    y = df_final[target_col]

    # Definir modelo base
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state, n_jobs=-1)

    # Definir grid de hiperparámetros
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }

    # Definir K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Configurar GridSearchCV
    grid_search = GridSearchCV(estimator=xgb_model,
                               param_grid=param_grid,
                               cv=kf,
                               scoring='neg_root_mean_squared_error',
                               verbose=1,
                               n_jobs=-1)

    # Ejecutar búsqueda
    grid_search.fit(X, y)

    # Resultados
    best_rmse = -grid_search.best_score_  # Lo volvemos positivo
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"Mejor RMSE (validación cruzada): {best_rmse:.4f}")

    # Mejor modelo entrenado
    mejor_modelo = grid_search.best_estimator_

    # RMSE en entrenamiento final
    y_pred_train = mejor_modelo.predict(X)
    rmse_entrenamiento = np.sqrt(mean_squared_error(y, y_pred_train))
    print(f"RMSE en entrenamiento: {rmse_entrenamiento:.4f}")

    # Guardar el modelo en ../models/
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)

    modelo_path = os.path.join(models_dir, 'xgboost_model.pkl')
    with open(modelo_path, 'wb') as f:
        pickle.dump(mejor_modelo, f)

    print(f"Modelo guardado en: {modelo_path}")

    return mejor_modelo, grid_search, best_rmse, rmse_entrenamiento
