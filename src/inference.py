import pandas as pd
import pickle
import os

def preprocesar_nuevos_datos(catalogo_producto, demanda):
    # Definir la ruta al directorio models (un nivel arriba de src/)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

    # Cargar las columnas y frequency_encoding guardados
    columnas_path = os.path.join(models_dir, 'columnas_df_final.pkl')
    freqenc_path = os.path.join(models_dir, 'frequency_encoding.pkl')

    with open(columnas_path, 'rb') as f:
        columnas_guardadas = pickle.load(f)

    with open(freqenc_path, 'rb') as f:
        frequency_encoding = pickle.load(f)

    # Definir las columnas requeridas
    columnas_requeridas = ['id_producto', 'categoria', 'subcategoria', 'tamaño',
                           'premium', 'marca_exclusiva', 'estacional']

    # Filtrar solo las columnas requeridas
    df_cat_producto = catalogo_producto[columnas_requeridas].copy()

    # Llenar valores faltantes
    df_cat_producto['subcategoria'] = df_cat_producto['subcategoria'].fillna('no_categoria')
    premium_mode = df_cat_producto['premium'].mode()[0]
    df_cat_producto['premium'] = df_cat_producto['premium'].fillna(premium_mode)
    df_cat_producto['tamaño'] = df_cat_producto['tamaño'].fillna('no_tamaño')

    # Codificar variables categóricas con one-hot encoding
    df_encoded = pd.get_dummies(df_cat_producto, columns=['categoria', 'tamaño'],
                                 prefix=['cat', 'tam'], drop_first=True, dtype=float)

    # Encoding de frecuencia para 'subcategoria' usando el diccionario guardado
    df_encoded['subcategoria'] = df_cat_producto['subcategoria'].map(frequency_encoding)

    # Si hay subcategorías nuevas que no estaban en el encoding previo, asignarles 0
    df_encoded['subcategoria'] = df_encoded['subcategoria'].fillna(0)

    # Procesar demanda
    df_demanda = demanda.copy()
    df_demanda['date'] = pd.to_datetime(df_demanda['date'])
    df_demanda['month'] = df_demanda['date'].dt.month
    df_demanda['day'] = df_demanda['date'].dt.day
    df_demanda = df_demanda.drop('date', axis=1)

    # Combinar ambos dataframes por 'id_producto'
    df_final = pd.merge(df_encoded, df_demanda, on='id_producto', how='inner')

    # Ajustar las columnas para que coincidan con las del modelo original
    for col in columnas_guardadas:
        if col not in df_final.columns:
            df_final[col] = 0  # Añadir columna faltante con valor 0

    # Dejar solo las columnas guardadas y en el mismo orden
    df_final = df_final[columnas_guardadas]

    print(f"Nuevos datos preprocesados usando los modelos en: {models_dir}")

    return df_final

def predecir_con_modelo(df_nuevos_datos):
    # Ruta del modelo
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    modelo_path = os.path.join(models_dir, 'xgboost_model.pkl')

    # Verificar que existe el modelo
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo en {modelo_path}")

    # Cargar el modelo
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)

    # Realizar predicciones
    predicciones = modelo.predict(df_nuevos_datos)

    # Devolver DataFrame con predicciones
    df_resultado = pd.DataFrame({
        'prediccion': predicciones
    })

    return df_resultado
