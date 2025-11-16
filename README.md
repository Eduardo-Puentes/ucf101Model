# UCF101 Action Recognition (Skeleton-Based)

Este proyecto implementa un modelo de deep learning para clasificar acciones humanas usando el dataset UCF101, trabajando únicamente con las coordenadas de esqueletos 2D proporcionadas por MMAction2.

El objetivo fue entrenar un modelo que pueda reconocer acciones utilizando secuencias de keypoints en lugar de video completo, aplicar un modelo baseline, mejorarlo con una arquitectura profunda y generar predicciones.

---

## 1. Dataset

- Dataset original: UCF101
- Representación utilizada: esqueletos 2D en formato .pkl
- Archivo utilizado en este proyecto: https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl

El archivo debe guardarse en:
data/ucf101_2d.pkl

Se trabajó con 5 clases del dataset para facilitar el entrenamiento.

---

## 2. Modelos implementados

### Baseline
- Calcula el promedio temporal de los keypoints.
- Clasifica con un MLP pequeño.
- Funciona como comparación inicial.

### Modelo principal: LSTM
- Trabaja con la secuencia completa de keypoints.
- Captura la información temporal del movimiento.
- Obtiene mejor desempeño que el baseline.

---

## 3. Cómo ejecutar el proyecto

### 1) Crear entorno e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Asegurarse de tener el archivo del dataset
data/ucf101_2d.pkl

### 3) Entrenar el baseline

```bash
python src/train.py \
  --pkl_path data/ucf101_2d.pkl \
  --split_name train1 \
  --selected_labels 3 11 25 43 76 \
  --model_type baseline \
  --num_epochs 20
```

### 4) Entrenar el modelo LSTM
```bash
python src/train.py \
  --pkl_path data/ucf101_2d.pkl \
  --split_name train1 \
  --selected_labels 3 11 25 43 76 \
  --model_type lstm \
  --num_epochs 20
```
Los modelos se guardan en:

checkpoints/

### 5) Generar predicciones
```bash
python src/predict.py \
  --pkl_path data/ucf101_2d.pkl \
  --split_name test1 \
  --checkpoint checkpoints/best_lstm.pt
```

## 4. Resultados

Baseline / ~0.6961

LSTM / ~0.7059


El modelo LSTM funciona mejor porque aprovecha la información temporal de los esqueletos.

## 5. Ejemplos de predicción

Correctos:

Label verdadero: 76
Label predicho: 76

Label verdadero: 25
Label predicho: 25

Incorrecto (caso confuso):

Label verdadero: 43
Label predicho: 76

## 6. Conclusión
Se implementó un modelo de deep learning basado en LSTM para clasificar acciones humanas usando esqueletos 2D.
Se comparó con un modelo baseline, se analizó el desempeño y se generaron predicciones reales, cumpliendo con los requisitos del módulo.



José Eduardo Puentes Martínez
ITESM – 2025
