from train import preprocesar_datos_y_guardar, entrenamiento_demanda
from inference import preprocesar_nuevos_datos, predecir_con_modelo
import pandas as pd


df_catalogo = pd.read_csv(r"C:\Users\Dell\Downloads\drive-download-20241031T211635Z-001 (1)\Datos\catalogo_productos.csv")
df_demanda = pd.read_csv(r"C:\Users\Dell\Downloads\drive-download-20241031T211635Z-001 (1)\Datos\demanda.csv")
# df_resultado = preprocesar_datos_y_guardar(df_catalogo, df_demanda)
#
# modelo_entrenado = entrenamiento_demanda(df_resultado, target_col='demanda')


# df_nueva_data = preprocesar_nuevos_datos(df_catalogo, df_demanda)
data_nueva = preprocesar_nuevos_datos(df_catalogo, df_demanda)

df_predicciones = predecir_con_modelo(data_nueva.drop(columns=['demanda']))

# Ver resultado
print(df_predicciones.head())
