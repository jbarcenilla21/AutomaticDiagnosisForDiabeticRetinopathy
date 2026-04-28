Aquí tienes el prompt ajustado a las restricciones del reporte:

---

# Prompt: Redacción del Reporte — Modelo Custom + Fine-Tuning Ensemble

Tu tarea es redactar las dos secciones del reporte del proyecto correspondientes al enfoque Custom desde cero y al enfoque de Fine-Tuning con Ensemble Heterogéneo.

**Restricciones de formato obligatorias:**
- El reporte completo tiene un límite estricto de **2 caras (2 páginas A4)**:
  - **Cara 1 — Descripción:** texto principal con las dos secciones del modelo. Debe ser denso pero legible; no hay espacio para introducciones largas ni conclusiones extensas.
  - **Cara 2 — Material extra:** tablas de resultados, figuras relevantes y referencias bibliográficas. No contiene prosa argumentativa.
- El objetivo declarado por el evaluador es que el profesor pueda valorar los *desarrollos, extensiones y decisiones tomadas al optimizar el sistema*. **No se requiere detalle exhaustivo**: basta con listar las decisiones y discutir brevemente su propósito.
- El reporte debe indicar explícitamente las **contribuciones individuales de cada miembro del grupo**, ya que la evaluación individual depende de ello.

**Estructura interna del texto (Cara 1):**

El reporte debe tener dos subsecciones explícitas:
- **"Modelo 1: Arquitectura Custom desde cero (SEResNet9)"**
- **"Modelo 2: Fine-Tuning con Ensemble Heterogéneo"**

Redacta ambas en tono académico, formal y fluido. Usa subtítulos en formato Markdown (`##`, `###`) para organizar la información. Dado el límite de espacio, cada decisión técnica debe justificarse en **1–3 frases**, priorizando el *por qué* sobre el *cómo*. Las justificaciones deben estar hiladas lógicamente bajo la premisa principal de: **lograr la máxima capacidad de generalización y extracción de características robustas a pesar de tener un conjunto de datos limitado e inbalanceado.**

---

### Información técnica — Modelo 1: Custom desde cero (SEResNet9)

**Arquitectura:**
- SEResNet9: red ResNet de 9 capas con bloques SE (Squeeze-and-Excitation) insertados tras cada ResBlock. Layer3 expandida a 256 canales. Total: ~1,65M parámetros.
- Dropout=0.1, inicialización Kaiming.
- Justificación: los bloques SE aplican atención por canal, permitiendo a la red suprimir canales irrelevantes y amplificar los informativos (p.ej. el canal verde donde destacan los microaneurismas), sin coste computacional relevante.

**Preprocesamiento:**
- `CropByEye` (recorte por umbral de brillo) aplicado **primero**, para eliminar los bordes negros inútiles antes de cualquier otra transformación.
- `BenGraham` (σ=15): `cv2.addWeighted(img, 4, GaussianBlur(img,(0,0),15), -4, 128)`. Mapea el fondo a gris(128) y maximiza el contraste de las lesiones retinianas. Se aplica **después** de CropByEye (orden crítico: si se invierte, BenGraham convierte el borde negro en gris, haciendo que CropByEye falle al detectar el ojo).
- `PerImageNormalize`: normalización por canal `(x−μ)/σ` sobre cada imagen individualmente, en lugar de estadísticas ImageNet. Justificación: las retinografías tienen distribución muy distinta a ImageNet; normalizar con sus propias estadísticas elimina el sesgo de dominio.

**Aumento de datos:**
- `RandomHorizontalFlip(p=0.5)`, `RandomRotation(180°)`, `ColorJitter`.
- `RandomCutout` (2 parches de 32×32 px, relleno con 0.5): obliga a la red a aprender características distribuidas en toda la retina en lugar de depender de regiones locales.

**Manejo del desbalance y entrenamiento:**
- `WeightedRandomSampler`: garantiza batches 50/50 positivo/negativo.
- `FocalLoss(γ=2.0, α=0.5)`: penaliza errores en ejemplos difíciles. Con α=0.5, equivale a 0.5×BCE ponderado por el factor focal; la combinación con el sampler evita doble corrección del desbalance.
- `SequentialLR`: warmup lineal (epochs 0–4, factor inicial 0.01) seguido de `CosineAnnealingLR` (epochs 5–59). El warmup evita colapso hacia la clase mayoritaria durante la inicialización aleatoria.

**Resultado:** mejor val AUC = **0.7292** (epoch 53).

---

### Información técnica — Modelo 2: Fine-Tuning con Ensemble Heterogéneo

**Preprocesamiento:**
- `CropByEye` y redimensionado con margen para evitar artefactos en el recorte final.
- `DualChannelEnhancement`: CLAHE aplicado por separado al canal Verde y al Rojo, con máscara estricta para no amplificar ruido del borde negro. Justificación médica: el canal verde maximiza contraste de microaneurismas y exudados; el canal rojo resalta hemorragias mayores y neovascularización.

**Aumento de datos:**
- Transformaciones geométricas (flips, rotaciones anatómicamente viables), Color Jitter y Ruido Gaussiano.
- El ruido gaussiano simula variabilidad de sensores médicos; las transformaciones agresivas contrarrestan el sobreajuste ante la escasez de datos.

**Arquitectura — Multi-Scale Heterogeneous Ensemble:**
- Dos modelos preentrenados en ImageNet: **EfficientNet-B2** y **DenseNet**, con descongelamiento progresivo de los últimos bloques (fine-tuning).
- **Técnica estrella — multi-escala**: cada modelo recibe la imagen a una resolución distinta (menor resolución para capturar contexto global del ojo; mayor resolución para detectar lesiones finas).
- **Pesos de ensemble aprendibles** (capa con Softmax): en lugar de promedio simple, la red aprende dinámicamente qué arquitectura (y escala) es más fiable para la predicción final — soft-voting ponderado óptimo.
- Transfer learning es esencial por la falta de datos médicos masivos: se reutilizan filtros detectores de bordes y texturas ya aprendidos.

**Manejo del desbalance y entrenamiento:**
- `WeightedRandomSampler` y `FocalLoss(α=0.25, γ=2.0)`.
- Optimizador AdamW, `ReduceLROnPlateau` y Early Stopping monitorizando AUC ROC en validación.

**Inferencia — Test-Time Augmentation (TTA):**
- N pasadas estocásticas con aumentos ligeros sobre cada imagen de test; se promedia la probabilidad final.
- Aporta robustez estadística y reduce la varianza de la predicción en datos no vistos.

---

**Instrucción final:** Al terminar el texto de la Cara 1, genera también el contenido propuesto para la **Cara 2**, indicando qué tabla(s) y figura(s) incluirías (con su descripción breve) y qué referencias bibliográficas serían pertinentes. No es necesario que redactes las referencias en formato completo; basta con indicar el tipo de fuente (p.ej. "paper original de bloques SE", "artículo de BenGraham preprocessing", "paper FocalLoss"). Genera todo esto en un md.