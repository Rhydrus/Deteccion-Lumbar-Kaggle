# Descripción del Proyecto

Este proyecto esta basado en una competicion realizada por kaggle tiene como objetivo identificar las condiciones médicas que afectan la columna lumbar en las resonancias magnéticas utilizando técnicas de aprendizaje automático. El objetivo final es clasificar las imágenes médicas en función de la severidad de las afecciones observadas en los estudios de resonancia magnética.

https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification

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

## Archivos

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

## Proceso de Evaluación

La competencia utiliza un conjunto de prueba oculto, lo que significa que los datos reales de prueba solo estarán disponibles durante la evaluación final. El archivo de envío debe seguir el formato especificado en `sample_submission.csv`.

## Estructura del Proyecto

- **src/**: Código fuente del proyecto, que incluye scripts para el preprocesamiento de datos, entrenamiento y evaluación del modelo.
- **data/**: Carpeta que contiene los conjuntos de datos (imágenes y etiquetas).
- **notebooks/**: Bloc de notas Jupyter con el análisis y desarrollo del modelo
- **models/**: Contiene los modelos entrenados

## Instrucciones para Ejecutar

1. Clonar el repositorio.
2. Instalar las dependencias listadas en el archivo `requirements.txt`.
3. Ejecutar los notebooks o scripts para entrenar los modelos y realizar predicciones.

## Descripción del Archivo `Detection.py`

El archivo `Detection.py` contiene clases y funciones claves para el procesamiento de imágenes DICOM y la detección de condiciones médicas en la columna lumbar. Este archivo es responsable de realizar tareas como cargar y manipular imágenes, realizar predicciones y visualizar resultados. A continuación, se describen las funciones más importantes:

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