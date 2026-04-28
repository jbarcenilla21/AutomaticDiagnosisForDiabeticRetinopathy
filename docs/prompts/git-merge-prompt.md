# Prompt: Merge de ramas custom + finetunning → main

## Contexto del proyecto

Repositorio: Clasificación binaria de Retinopatía Diabética (DR) con PyTorch CNNs.
Dos tracks de entrega (Codabench): Custom (red desde cero) y Fine-tuning (modelo preentrenado).

Hay dos ramas de trabajo que deben fusionarse en `main` como versión final entregable:

- **`custom`** (Jorge): SEResNet9 con BenGraham preprocessing, PerImageNormalize, RandomCutout, FocalLoss(γ=2.0, α=0.5), WeightedRandomSampler, SequentialLR con warmup. Mejor val AUC: 0.7292.
- **`origin/finetunning`** (compañero): EfficientNet-B2 + DenseNet multi-scale ensemble con pesos aprendibles (Softmax), DualChannelEnhancement (CLAHE verde+rojo), FocalLoss(γ=2.0, α=0.25), AdamW + ReduceLROnPlateau + Early Stopping, TTA (10 pasadas).

## Tu tarea

Fusionar ambas ramas en `main` produciendo un repositorio limpio y funcional donde:
1. El track **Custom** usa el código de `custom`.
2. El track **Fine-tuning** usa el código de `finetunning`.
3. Ambos tracks coexisten bajo la misma estructura `src/`, sin conflictos.
4. Hay un único notebook `cv-lab.ipynb` que ejecuta los dos tracks.
5. El código pasa `pytest tests/` sin errores.

## Estado actual de las ramas

```
main          ← base original (estructura vacía, 3 commits)
custom        ← 9 commits sobre main (rama activa local)
origin/finetunning ← rama del compañero (no mergeada aún)
```

## Estructura de ficheros objetivo tras el merge

```
src/
  data/
    dataset.py          ← usar versión de custom (más completa)
    transforms.py       ← MERGE MANUAL (ver sección de conflictos)
  model/
    custom_net.py       ← de custom (SEResNet9)
    fine_tune_net.py    ← de custom (wrapper EfficientNet/DenseNet)
    ensemble_net.py     ← de finetunning (BaseModel + EnsembleModel multi-scale)
    __init__.py         ← exportar todas las clases de ambas ramas
  training/
    config.py           ← usar versión de finetunning (Config/CustomConfig/FineTuneConfig)
    losses.py           ← MERGE MANUAL (ver sección de conflictos)
    trainer.py          ← usar versión de custom (más modular)
  evaluation/
    metrics.py          ← cualquiera (idénticos)
    submission.py       ← de custom (tiene generate_submission completo)
utils/
  utils.py              ← de custom
tests/
  test_transforms.py    ← de custom (7 tests BenGraham, RandomCutout, PerImageNormalize)
  test_losses.py        ← MERGE (unificar tests de ambas ramas)
  test_models.py        ← de finetunning (tiene tests de EnsembleModel)
cv-lab.ipynb            ← de finetunning como base, añadir celdas del custom track
```

## Conflictos a resolver manualmente

### 1. `src/data/transforms.py`

**custom** tiene:
- `BenGraham(sigma=15.0)` — numpy uint8 in/out, mejora contraste de lesiones
- `PerImageNormalize()` — normaliza por canal en tensor (C,H,W), reemplaza stats ImageNet
- `RandomCutout(n_holes=2, patch_size=32)` — regularización
- Pipeline train: `CropByEye → BenGraham → Rescale → RandomCrop → Flip → Rotation → ColorJitter → RandomCutout → ToTensor → PerImageNormalize`
- **ORDEN CRÍTICO**: CropByEye SIEMPRE antes de BenGraham (BenGraham convierte bordes negros a gris128, rompiendo la detección del ojo)

**finetunning** tiene:
- `DualChannelEnhancement` — CLAHE en canal verde y rojo por separado con máscara
- `GaussianNoise` — ruido gaussiano como augmentación
- Resoluciones multi-escala: [224, 384, 512]

**Resolución**: Mantener todos los transforms de ambas. Añadir `DualChannelEnhancement` y `GaussianNoise` al fichero de custom. Las funciones `get_train_transforms()` y `get_val_transforms()` del custom track usan el pipeline de custom; las del fine-tune track usan el pipeline de finetunning. Exponer ambas desde el mismo módulo.

### 2. `src/training/losses.py`

**custom** tiene: `FocalLoss` con α=0.5 por defecto, `gamma=2.0`, implementación limpia con `alpha_t`.

**finetunning** tiene: `FocalLoss` dentro de `src/model_components.py` con α=0.25 por defecto.

**Resolución**: Usar la implementación de `custom` en `src/training/losses.py` (más testeable y modular) pero con `alpha` como parámetro configurable sin valor por defecto fijo. El custom track instancia con `alpha=0.5`; el fine-tune track con `alpha=0.25`.

```python
# Firma correcta:
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, reduction='mean'):
```

### 3. `src/model/__init__.py`

Debe exportar todas las clases de ambas ramas:
```python
from .custom_net import SEResNet9
from .fine_tune_net import FineTuneNet
from .ensemble_net import BaseModel, EnsembleModel
```

### 4. `src/training/config.py`

Usar la versión de finetunning (`Config`, `CustomConfig`, `FineTuneConfig`) que es más completa. Añadir los parámetros de warmup del custom track:
- `warmup_epochs: int = 5`
- `warmup_start_factor: float = 0.01`

### 5. Notebook `cv-lab.ipynb`

Usar `cv-lab.ipynb` de finetunning como base (ya tiene la estructura dual-track). Añadir/reemplazar las celdas del **Custom track** con el código de `train_colab.ipynb` de custom (SEResNet9 + SequentialLR + WeightedRandomSampler).

## Procedimiento de merge

```bash
# 1. Asegurarse de estar en main
git checkout main

# 2. Mergear custom primero (tiene más commits y es la rama activa)
git merge custom --no-ff -m "merge: integrate custom track (SEResNet9)"

# 3. Hacer fetch y mergear finetunning
git fetch origin
git merge origin/finetunning --no-ff -m "merge: integrate fine-tuning ensemble track"
# → Habrá conflictos. Resolverlos según la sección anterior.

# 4. Tras resolver conflictos:
git add .
git commit -m "merge: resolve conflicts — unified custom + fine-tune tracks"

# 5. Verificar
pytest tests/ -v
```

## Criterios de éxito

- [ ] `pytest tests/` pasa sin errores (mínimo 16 tests)
- [ ] `from src.model import SEResNet9, EnsembleModel` funciona
- [ ] `from src.training.losses import FocalLoss` funciona
- [ ] `from src.data.transforms import BenGraham, DualChannelEnhancement` funciona
- [ ] El notebook `cv-lab.ipynb` tiene secciones diferenciadas para Custom y Fine-tuning
- [ ] No quedan marcadores de conflicto git (`<<<<<<`, `=======`, `>>>>>>>`) en ningún fichero
- [ ] `git log --oneline main` muestra ambos merge commits

## Lo que NO debes hacer

- No eliminar código funcional de ninguna de las dos ramas sin justificación explícita
- No cambiar la lógica de `BenGraham` ni el orden `CropByEye → BenGraham`
- No cambiar la firma de `FocalLoss` (tests existentes dependen de ella)
- No hacer `git push --force` sobre ninguna rama
