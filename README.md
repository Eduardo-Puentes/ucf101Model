# UCF101 Skeleton-based Action Recognition (subset de 5 clases)

Clasificación de acciones humanas a partir de esqueletos 2D del dataset UCF101,
usando un baseline simple y un LSTM ajustado con manejo de longitudes reales.

## Dataset y splits
- Anotaciones de esqueletos 2D: `ucf101_2d.pkl` (MMAction2).  
  Descargar: `https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl`
  y guardar en `data/ucf101_2d.pkl`.
- Splits disponibles dentro del pkl: `train1`, `train2`, `train3`, `test1`, `test2`, `test3`.
- Subconjunto de 5 clases (IDs de ejemplo): BasketballShooting, JumpingJack, PullUps,
  PushUps, WalkingWithDog. Internamente se usan los IDs enteros del pkl.

## Instalación rápida
```bash
git clone <este_repo>
cd ucf101-skeleton-actions
python3 -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
```

## Cómo entrenar
- LSTM recomendado (mejor métrica actual, split `train1`):
```bash
python3 src/train.py --split_name train1 --model_type lstm \
  --selected_labels 0 1 2 3 4 --max_frames 80 \
  --lstm_hidden_dim 384 --lstm_layers 2 --bidirectional \
  --dropout 0.4 --lr 5e-4 --weight_decay 1e-4 --grad_clip 1.0 \
  --num_epochs 30
```
- Baseline de referencia:
```bash
python3 src/train.py --split_name train1 --model_type baseline \
  --selected_labels 0 1 2 3 4 --num_epochs 10
```
- Hiperparámetros expuestos: `--lstm_hidden_dim`, `--lstm_layers`, `--bidirectional`,
  `--dropout`, `--weight_decay`, `--grad_clip`, `--scheduler_step/--scheduler_gamma`,
  y para el baseline `--baseline_hidden_dim`. Los checkpoints guardan estos valores y el
  mapeo de etiquetas usado.

## Ajustes clave
- Padding enmascarado: el dataset devuelve longitudes reales y el LSTM usa
  `pack_padded_sequence`; el baseline promedia solo frames válidos. Esto eliminó la
  brecha mínima que aparecía antes entre baseline y LSTM.
- Más capacidad y regularización: control de `hidden_dim`, capas, bidireccionalidad,
  `dropout`, `weight_decay`, clip de gradiente y scheduler opcional.
- Predicción interactiva: `src/predict.py` acepta índice, `frame_dir` o keypoints externos
  (T x V x 2) con `--img_shape` para normalizar si vienen en pixeles.

## Resultados recientes (train1/test1, 5 clases)
- **LSTM bidireccional** (`hidden_dim=384`, `layers=2`, `dropout=0.4`, `max_frames=80`):  
  mejor `val_acc` **0.8587** (`checkpoints/best_lstm.pt`), probada con `test1`
  (ejemplo correcto: `v_ApplyEyeMakeup_g03_c03`).
- **Baseline temporal mean + MLP**:  
  mejor `val_acc` **0.6087** (`checkpoints/best_baseline.pt`).

La mejora de ~25 pts confirma que la información temporal aporta una ganancia real al
corregir el uso de padding y ajustar hiperparámetros.

## Predicciones individuales por consola
```bash
# Índice específico del split
python3 src/predict.py --split_name test1 --checkpoint checkpoints/best_lstm.pt --index 12

# Un frame_dir concreto (asegúrate de que está en el split elegido)
python3 src/predict.py --split_name test1 --checkpoint checkpoints/best_lstm.pt --frame_dir v_JumpingJack_g01_c01

# Keypoints externos (T x V x 2), normalizando con img_shape si vienen en pixeles
python3 src/predict.py --checkpoint checkpoints/best_lstm.pt --keypoint_path sample.npy --img_shape 240 320
```
- Flags útiles: `--topk` para probabilidades top-k, `--frame_dir` o `--index` para elegir
  muestra; `--max_frames` se toma del checkpoint.

## Comprobar qué `frame_dir` hay en un split
```bash
python3 - <<'PY'
import pickle
data = pickle.load(open("data/ucf101_2d.pkl", "rb"), encoding="latin1")
split = "test1"
fds = data["split"][split]
print(f"Total en {split}: {len(fds)}")
print("Primeros 10:", fds[:10])
PY
```

## Checklist vs feedback del profesor
- Ajustes de arquitectura/hiperparámetros documentados y reproducibles.
- Evidencia de mejora: baseline ~0.61 → LSTM ~0.86 en `val_acc` (mismo split).
- Explicación de la brecha inicial: el padding contaminaba el estado final; ahora se usa
  longitud real y pooling enmascarado.
- Predicciones individuales por consola (`src/predict.py`) con `--index`, `--frame_dir`
  o `--keypoint_path`, mostrando top-k y clase original.
