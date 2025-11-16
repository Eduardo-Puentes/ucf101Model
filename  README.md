# UCF101 Skeleton-based Action Recognition (Subset de 5 clases)

Este proyecto implementa un modelo de **aprendizaje profundo** para
clasificar acciones humanas en videos del dataset **UCF101** usando
**esqueletos 2D** preprocesados (keypoints) proporcionados por
[MMAction2 – Revisiting Skeleton-based Action Recognition](https://mmaction2.readthedocs.io/en/latest/dataset_zoo/skeleton.html).

## Dataset

- Dataset original: UCF101 (101 clases de acción)  
- Anotaciones de esqueletos 2D usadas: `ucf101_2d.pkl`  
  - Descargar desde:  
    `https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl`
  - Guardar en `data/ucf101_2d.pkl`

El archivo `.pkl` contiene:
- `split`: diccionario de splits → lista de `frame_dir`
- `annotations`: lista de videos con:
    - `frame_dir`, `total_frames`, `img_shape`, `label`,
    - `keypoint` (M x T x V x 2), `keypoint_score` (M x T x V)

Para mapear `label (int)` a nombre de clase, se puede usar el archivo
`classInd.txt` disponible en la página de UCF101.

## Subconjunto de clases

Para este proyecto se selecciona un subconjunto de 5 clases (ejemplo):

- BasketballShooting  
- JumpingJack  
- PullUps  
- PushUps  
- WalkingWithDog  

En el código, se trabaja internamente con los **IDs enteros** de UCF101
(asignados en el `.pkl`). Los IDs concretos se deben ajustar tras
inspeccionar el `.pkl` y/o `classInd.txt`.

## Instalación

```bash
git clone <este_repo>
cd ucf101-skeleton-actions
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -r requirements.txt