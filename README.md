## Introducción

Este repositorio contiene el código y los datos necesarios para participar en la competencia de Kaggle "RSNA 2024 Lumbar Spine Degenerative Classification". El objetivo de esta competencia es desarrollar modelos de aprendizaje automático que puedan identificar y clasificar condiciones médicas que afectan la columna lumbar en imágenes de resonancia magnética.

Este proyecto está basado en una competencia realizada por Kaggle y tiene como objetivo identificar las condiciones médicas que afectan la columna lumbar en las resonancias magnéticas utilizando técnicas de aprendizaje automático. El objetivo final es clasificar las imágenes médicas en función de la severidad de las afecciones observadas en los estudios de resonancia magnética.

## Descripción del Conjunto de Datos

El conjunto de datos incluye imágenes y etiquetas correspondientes a las condiciones médicas que afectan la columna lumbar. Existen tres condiciones principales a evaluar:

1. **Estenosis del canal espinal** (spinal canal stenosis).
2. **Estrechamiento del foramen neural** (neural foraminal narrowing) en ambos lados de la columna vertebral.
3. **Estenosis subarticular** (subarticular stenosis).

Cada condición se clasifica en cuatro niveles de severidad:

- **Normal**
- **Leve (Mild)**
- **Moderado (Moderate)**
- **Grave (Severe)**

El conjunto de datos también incluye coordenadas que indican la ubicación de las áreas de interés dentro de las imágenes 3D de resonancia magnética.

## Descripción de los Archivos del Repositorio

A continuación se describen los archivos más relevantes para el proyecto:

1. **train.csv**: Contiene las etiquetas del conjunto de entrenamiento.
   - `study_id`: ID del estudio (varios estudios pueden incluir múltiples series de imágenes).
   - `condition_[level]`: Las etiquetas de destino para cada condición y nivel de la columna vertebral (Normal/Mild/Moderate/Severe).

2. **train_label_coordinates.csv**: Contiene las coordenadas de las áreas etiquetadas.
   - `study_id`: ID del estudio.
   - `series_id`: ID de la serie de imágenes.
   - `instance_number`: Número de orden de la imagen dentro de la pila 3D.
   - `condition`: Condición médica (estenosis del canal espinal, estrechamiento del foramen neural, o estenosis subarticular).
   - `level`: Nivel vertebral relevante (por ejemplo, l3_l4).
   - `x/y`: Coordenadas del centro del área etiquetada.

3. **sample_submission.csv**: Ejemplo de archivo de envío para predicciones.
   - `row_id`: Identificador único que combina `study_id`, `condition` y `level` (ej. `12345_spinal_canal_stenosis_l3_l4`).
   - Tres columnas de predicción para la clasificación (Normal/Mild/Moderate/Severe).

4. **[entrenar/probar]_images/[study_id]/[series_id]/[instance_number].dcm**: Archivos de imágenes en formato DICOM (.dcm), que contienen los datos de las imágenes de resonancia magnética.

5. **[entrenar/probar]_series_descriptions.csv**: Descripciones de las series de imágenes.
   - `study_id`: ID del estudio.
   - `series_id`: ID de la serie de imágenes.
   - `series_description`: Descripción de la orientación del escaneo.

## Proceso de Entrenamiento

El proceso de entrenamiento incluye varias etapas clave:

1. **Preprocesamiento de Datos**: Las imágenes DICOM se cargan y se organizan, y las coordenadas de las áreas de interés se extraen y se preparan para el entrenamiento del modelo.
2. **Entrenamiento del Modelo**: Se utiliza un modelo de aprendizaje profundo para detectar y clasificar las condiciones lumbares en las imágenes. El entrenamiento incluye la optimización y evaluación del modelo.
3. **Evaluación y Predicción**: El modelo entrenado se evalúa utilizando un conjunto de datos de validación, y se generan predicciones para el conjunto de prueba.

## Cómo Llamar las Funciones

### Funciones Principales:

1. **`load_dicom_stack`**:
   - Esta función carga un conjunto de imágenes DICOM desde una carpeta específica y organiza las imágenes de acuerdo con las posiciones de los pacientes.
   - **Uso**:
     ```python
     sag_t2 = dcm.load_dicom_stack(dicom_folder='ruta/a/carpeta/dicom', plane='axial')
     ```

2. **`plot_img`**:
   - Esta función traza coordenadas sobre una imagen DICOM y muestra los puntos clave correspondientes a niveles lumbares.
   - **Uso**:
     ```python
     dcm.plot_img(img, coords_temp)
     ```

3. **`plot_5_crops`**:
   - Genera cinco recortes de una imagen, centrados en un par de puntos clave, y los rota para alinearlos correctamente.
   - **Uso**:
     ```python
     dcm.plot_5_crops(img, coords_temp)
     ```

4. **`show_scan_and_coords`**:
   - Muestra una serie de imágenes DICOM y las coordenadas asociadas para un paciente específico y plano.
   - **Uso**:
     ```python
     dcm.show_scan_and_coords(patient_folder='ruta/a/carpeta', plane=1, level=2)
     ```

5. **`train_model`**:
   - Entrena un modelo de aprendizaje profundo para detectar condiciones lumbares en las imágenes. Incluye el procesamiento de datos, optimización y evaluación del modelo.
   - **Uso**:
     ```python
     trainer.train(epochs=10)
     ```

6. **`visualize_prediction`**:
   - Visualiza las predicciones del modelo superpuestas en las imágenes originales de DICOM.
   - **Uso**:
     ```python
     trainer.visualize_prediction(batch, pred, epoch)
     ```

## Estructura del Proyecto

El archivo `Detection.py` se encuentra en la carpeta `src/` y contiene las funciones que permiten la manipulación de imágenes y el entrenamiento del modelo para la clasificación de patologías lumbares.

### Estructura de las carpetas:
- **src/**
  - **Detection.py**: Contiene las funciones y clases mencionadas arriba.
  - **data/**: Funciones para manipular los datos del proyecto, incluidos los DICOM.
- **data/**: Carpeta que contiene las imágenes y los archivos CSV relacionados con los estudios de resonancia magnética.

## Importante
La carpeta `data`, que contiene los archivos CSV y las imágenes de entrenamiento y prueba, debe descargarse desde Kaggle debido al tamaño de los archivos (aproximadamente 36 GB).

 https://www.kaggle.com/datasets/brendanartley/lumbar-coordinate-pretraining-dataset

 https://www.kaggle.com/datasets/brendanartley/lumbar-coordinate-pretraining-dataset?select=data

## Estructura de las Carpetas del Repositorio

A continuación se muestra la estructura de las carpetas del repositorio:

```
Lumbar/
├── data/
│   ├── train.csv
│   ├── train_label_coordinates.csv
│   ├── sample_submission.csv
│   ├── train_images/
│   │   └── [study_id]/
│   │       └── [series_id]/
│   │           └── [instance_number].dcm
│   ├── test_images/
│   │   └── [study_id]/
│   │       └── [series_id]/
│   │           └── [instance_number].dcm
│   ├── train_series_descriptions.csv
│   └── test_series_descriptions.csv
├── src/
│   ├── Detection.py
│   └── data/
│       └── __init__.py
└── README.md
```