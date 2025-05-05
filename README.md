# Modelo de Estimación de Demanda con XGBoost

Este repositorio contiene un modelo basado en **XGBoost** diseñado para estimar la demanda de productos a partir de datos históricos.

## Clonación del Repositorio

Para comenzar, clona el repositorio usando el siguiente comando:

```bash
git clone https://github.com/wpradab/sumz_model.git
````

## Estructura del Proyecto

El código fuente se encuentra en la carpeta `src`. Allí se encuentra el script principal para entrenamiento e inferencia del modelo.

```
sumz_model/
│
├── src/
│   └── test.py
│
└── README.md
```

## Ejecución del Modelo

Para realizar pruebas con el modelo (ya sea entrenamiento o inferencia), debes entrar a la carpeta `src` y ejecutar el script `test.py`. Por ejemplo:

```bash
cd sumz_model/src
python test.py
```

> **Nota**: Asegúrate de tener instaladas todas las dependencias necesarias. Puedes usar un entorno virtual y un archivo `requirements.txt` si está disponible.

