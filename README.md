![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)

# Inpainting for Coronary Angiography X-Rays

Forschungsprojekt am Lehrstuhl für Medizinische Physik, Universität Heidelberg.

Dieses Projekt adaptiert **CMT (Continuously Masked Transformer)** für die Inpainting-Aufgabe auf graustufigen Koronarangiographie-Bildern des [ARCADE-Datensatzes](https://arcade.grand-challenge.org/). Ziel ist die realistische Rekonstruktion von X-Ray-Hintergründen durch Inpainting von Gefäßregionen, um synthetische Bilddaten zu generieren.

> Basierend auf: Keunsoo Ko and Chang-Su Kim, "Continuously Masked Transformer for Image Inpainting", ICCV 2023

---

## Voraussetzungen

- Python 3.9
- PyTorch (CPU, MPS oder CUDA)
- ARCADE-Datensatz ([Download](https://arcade.grand-challenge.org/))

```bash
pip install -r requirements.txt
```

---

## Pipeline Übersicht

```
ARCADE Dataset
      │
      ▼
Phase 1: Backbone Pretraining (train_placesCNN.py)
      │   ResNet50 auf ARCADE-Klassen trainieren
      │   → place2.pth
      ▼
Phase 2: CMT Inpainting Training (train.py)
      │   Encoder mit place2.pth initialisiert
      │   → checkpoints/best.pth
      ▼
Inference (demo.py)
```

---

## Training auf Google Colab (empfohlen)

Das beigelegte Notebook `arcade_pretrain.ipynb` führt beide Trainingsphasen automatisch durch.

1. Notebook auf [colab.research.google.com](https://colab.research.google.com) hochladen
2. **Runtime → Change runtime type → T4 GPU**
3. `arcade.zip` über den Datei-Browser hochladen
4. Zellen der Reihe nach ausführen (~15 Min Pretraining + ~2h CMT Training)
5. Checkpoints werden am Ende automatisch heruntergeladen

---

## Lokales Training

### Phase 1 — Backbone Pretraining

COCO-Annotationen in Klassenordner konvertieren:

```bash
python coco_to_classification.py \
  --annotations arcade/syntax/train/annotations/train.json \
  --images      arcade/syntax/train/images \
  --output      data/train

python coco_to_classification.py \
  --annotations arcade/syntax/val/annotations/val.json \
  --images      arcade/syntax/val/images \
  --output      data/val
```

Leere Klassenordner entfernen und Pretraining starten:

```bash
find data/train -type d -empty -delete
find data/val   -type d -empty -delete

python train_placesCNN.py data/ \
  --arch resnet50 \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.01
```

Checkpoint verlinken:

```bash
ln -sf checkpoints/resnet50_best.pth.tar place2.pth
```

### Phase 2 — CMT Inpainting Training

```bash
python train.py \
  --train_img arcade/syntax/train/images \
  --train_ann arcade/syntax/train/annotations/train.json \
  --val_img   arcade/syntax/val/images \
  --val_ann   arcade/syntax/val/annotations/val.json \
  --epochs 100 \
  --batch_size 4 \
  --input_size 256 \
  --device cpu
```

Wichtige Flags:

| Flag | Default | Beschreibung |
|------|---------|--------------|
| `--input_size` | 256 | Bildgröße (Potenz von 2, min 32) |
| `--pretrain` | place2.pth | Pfad zum Backbone-Checkpoint |
| `--no_pretrain` | – | Backbone-Initialisierung überspringen |
| `--ckpt` | – | CMT-Checkpoint zum Fortsetzen |
| `--device` | cpu | `cpu`, `cuda` oder `mps` |

**Smoke Test** (Pipeline schnell verifizieren):

```bash
python train.py \
  --smoke_test --smoke_size 20 --epochs 1 --batch_size 1 --device cpu
```

---

## Inference

```bash
python demo.py \
  --ckpt        checkpoints/best.pth \
  --img_path    ./samples/test_img \
  --mask_path   ./samples/test_mask \
  --output_path ./samples/results \
  --device      cpu
```

Bilder werden automatisch auf die Modell-Eingabegröße skaliert. Für GPU: `--device cuda`.

---

## Änderungen gegenüber Original-CMT

- 1-Kanal Graustufen-Input (statt RGB)
- Dynamische Encoder-Decoder-Pyramide in `refine.py` via `--input_size`
- `train_placesCNN.py` — Places365-Pipeline für ARCADE-Pretraining adaptiert
- `coco_to_classification.py` — konvertiert COCO-Annotationen in ImageFolder-Layout
- Gefäßmasken automatisch aus COCO-Polygon-Annotationen generiert
- MPS/CPU/CUDA-Kompatibilität
- `--smoke_test` Flag für schnelle Pipeline-Verifikation

---

## Projektstruktur

```
CMT/
├── train.py                    # CMT Inpainting Training
├── train_placesCNN.py          # Backbone Pretraining (Places365 style)
├── coco_to_classification.py   # COCO → ImageFolder Konverter
├── demo.py                     # Inference
├── utils.py                    # Hilfsfunktionen
├── network/                    # Modellarchitektur
├── arcade_pretrain.ipynb       # Colab Notebook (komplette Pipeline)
└── requirements.txt
```

---

## Autor

Dzielski — Forschungspraktikum Medizinische Physik, Universität Heidelberg
