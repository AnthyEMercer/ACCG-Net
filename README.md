# ACCG-Net: Anatomy-Aware Case-Conditioned Visual-Semantic Knowledge Graph Reasoning for Hierarchical Radiology Report Generation
This repository contains the official implementation of the paper:

> **Anatomy-Aware Case-Conditioned Visual-Semantic Knowledge Graph Reasoning for Hierarchical Radiology Report Generation**  

---

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)

---

## Installation

Create a conda environment and install dependencies:
```bash
conda create -n adgnet python=3.9
conda activate adgnet
pip install -r requirements.txt
```

---

## Data Preparation

### 1. Dataset Requirements
This project requires the **MIMIC-CXR** dataset and its associated **Chest ImaGenome** scene graph annotations. Please apply for access through the [PhysioNet](https://physionet.org/content/mimic-cxr/).

### 2. Region Data Extraction
Extract anatomical region data from Chest ImaGenome following the methodology described in [ttanida/rgrg](https://github.com/ttanida/rgrg):

### 3. Dataset Splitting
Split the extracted data into training, validation, and test sets saved as CSV files with the following schema:

| Column Name             | Description                                      |
|-------------------------|--------------------------------------------------|
| `image_id`              | Image identifier                                 |
| `subject_id`            | Patient identifier                               |
| `study_id`              | Study identifier                                 |
| `mimic_file_path`       | Path to image file                               |
| `bbox_coordinates`      | Bounding box coordinates                         |
| `bbox_labels`           | Anatomical findings per region                   |
| `bbox_phrases`          | Associated descriptive phrases                   |
| `bbox_phrase_exists`    | Binary indicator for phrase presence             |
| `bbox_is_abnormal`      | Abnormality flag per region                      |
| `bbox_anatomicalfinding`| Anatomical findings per image                    |
| `reference_report`      | Radiologist-written reference report             |

Place the resulting CSV files and the patients' indication and history information file in `data/`:
```
data/
    ├── patient_info.csv 
    ├── train.csv
    ├── valid.csv
    └── test.csv
```

---

## Training
**Region-level report generator**: Configure training parameters in `configs.py`, then launch the trainer:

```bash
python tools/KG/Trainer_3p.py
```

Training checkpoints will be saved to the output directory.

---

## Inference

### Step 1: Generate Region-level Reports
Configure inference settings in `configs.py`, then run:

```bash
python tools/KG/reference_test.py
```

### Step 2: Generate Image-level Reports
Aggregate region-level outputs into comprehensive image-level reports:

```bash
python tools/batch_reference_prepare_RAG_filter.py
```

---
