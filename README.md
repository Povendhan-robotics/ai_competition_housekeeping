# AI Competition Housekeeping

This repository provides baseline code and reference materials for the AI competition. Participants should use Google Colab to develop and run their solutions, then integrate trained models into the product template.

## 🎯 Workflow

```
1. Study Reference Notebooks → 2. Create Your Colab Notebooks → 3. Train Models → 4. Integrate into Product
```

## 📁 Repository Structure

```
ai_competition_housekeeping/
├── README.md                    # This file
├── requirements.txt             # Python dependencies (for local testing)
├── configs/                     # Configuration files for each stage
│   ├── stage1_binary.yaml      # Binary classification
│   ├── stage2_multilabel.yaml  # Multi-label classification
│   ├── stage3_weak_cam.yaml    # Localization
│   ├── stage4_alignment.yaml   # Geometry alignment
│   └── stage5_robustness.yaml  # Robustness testing
├── notebooks/                   # Reference notebooks (use as examples)
│   ├── 00_setup_colab.ipynb    # Colab setup guide
│   ├── 01_stage1_train.ipynb   # Stage 1 training example
│   ├── 02_stage1_infer.ipynb   # Stage 1 inference example
│   ├── 03_stage2_train.ipynb   # Stage 2 training example
│   ├── 04_stage2_infer.ipynb   # Stage 2 inference example
│   ├── 05_stage4_alignment.ipynb
│   ├── 06_robustness.ipynb
│   ├── 07_stage3_weak_cam.ipynb
│   └── val_quickly_stage1.ipynb
├── src/                         # Reference baseline code
│   ├── common/                  # Shared utilities
│   │   ├── dataset.py          # Dataset loaders
│   │   ├── io.py               # Input/output utilities
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── seed.py             # Random seed initialization
│   │   ├── transforms.py       # Image transformations
│   │   └── utils.py            # Helper functions
│   ├── stage1_binary/          # Binary classification baselines
│   │   ├── model.py            # Model architecture
│   │   ├── train.py            # Training script
│   │   ├── infer.py            # Inference script
│   │   ├── eval.py             # Evaluation script
│   │   └── baseline_todos.md   # Improvement suggestions
│   ├── stage2_multilabel/      # Multi-label classification baselines
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   ├── eval.py
│   │   ├── thresholding.py
│   │   └── baseline_todos.md
│   ├── stage3_localization/    # Localization baselines
│   │   ├── weak_cam/           # CAM-based methods
│   │   └── yolo/               # YOLO-based methods
│   ├── stage4_alignment/       # Geometry alignment baselines
│   │   ├── model.py
│   │   ├── infer.py
│   │   ├── eval.py
│   │   ├── geometry_baseline.py
│   │   ├── keypoints.py
│   │   └── baseline_todos.md
│   └── stage5_robustness/      # Robustness testing
│       ├── corruptions.py
│       ├── run_robustness.py
│       └── report.py
├── rules/                       # Competition rules
│   ├── AUGMENTATION_GUIDELINES.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── LEADERBOARD_PROTOCOL.md
│   ├── ML_RULES.md
│   └── STUDENT_GUIDE.md
└── submissions/                 # Sample submission formats
    ├── README.md
    ├── sample_stage1.csv
    ├── sample_stage2.csv
    └── sample_stage4.csv
```

## 🚀 Getting Started

### Step 1: Copy Notebooks to Your Google Drive

1. Open Google Colab: https://colab.research.google.com/
2. Upload or open notebooks from the `notebooks/` directory
3. Save copies to your Google Drive for editing

### Step 2: Study Reference Baselines

Review these files to understand the expected implementation:

| Stage | Reference Files | Purpose |
|-------|----------------|---------|
| 1 | `src/stage1_binary/model.py`, `train.py` | Binary classification architecture |
| 2 | `src/stage2_multilabel/model.py`, `train.py` | Multi-label classification |
| 3 | `src/stage3_localization/` | Localization methods |
| 4 | `src/stage4_alignment/keypoints.py` | Keypoint detection |
| 5 | `src/stage5_robustness/corruptions.py` | Image corruptions |

### Step 3: Create Your Own Notebooks

Use the reference notebooks as templates to create your own Colab notebooks:

- **Start with**: `00_setup_colab.ipynb`
- **Training**: Create notebook based on `01_stage1_train.ipynb`
- **Inference**: Create notebook based on `02_stage1_infer.ipynb`

### Step 4: Train Your Models

In your Colab notebooks:

```python
# Example pattern (see notebooks/01_stage1_train.ipynb)
from src.stage1_binary.train import train
from configs.stage1_binary import Config

config = Config()
train(config)
```

### Step 5: Save Trained Models

Save models in a format compatible with product integration:

```python
# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model.config,  # Include input size, etc.
}, 'models/stage1_model.pth')
```

## 🔄 Product Integration

After training, integrate your models into the product template using the expected interface pattern.

### Integration Pattern

For each stage, replace the baseline prediction function with your trained model:

```python
import torch
from torchvision import transforms
from PIL import Image

# Load your trained model (do this once, not per request)
model = torch.load("models/stage1_model.pth")
model.eval()

def predict_stage1(
    image_path: str,
    *,
    input_width: int = 224,  # Your model's input size
    model_path: str = "models/stage1_model.pth",
) -> Stage1Result:
    """Your trained Stage 1 model inference."""
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((input_width, input_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        prob = model(img_tensor).sigmoid().item()
    
    pred_made = prob >= 0.5
    
    return Stage1Result(
        prob_made=prob,
        pred_made=pred_made,
        debug={
            "model": "custom_trained",
            "note": "Replace with your model loading logic"
        },
    )
```

### Expected Output Format

Each stage must return a typed result with probabilities and predictions:

| Stage | Result Class | Key Fields |
|-------|--------------|------------|
| 1 | `Stage1Result` | `prob_made`, `pred_made`, `debug` |
| 2 | `Stage2Result` | `prob_label1`, ..., `pred_label1`, ..., `debug` |
| 3 | `Stage3Result` | `prob_localization`, `pred_localization`, `debug` |
| 4 | `Stage4Result` | `keypoints`, `confidence`, `debug` |
| 5 | `Stage5Result` | `robustness_score`, `corruption_results`, `debug` |

### Product Template Location

See the product template repository for the integration destination:

- **Stage 1**: `backend/ml/stage1_binary.py`
- **Stage 2**: `backend/ml/stage2_multilabel.py`
- **Stage 4**: `backend/ml/stage4_alignment.py`

## 📊 Competition Stages

| Stage | Task | Files |
|-------|------|-------|
| 1 | Binary Classification | `src/stage1_binary/`, `notebooks/01_*.ipynb` |
| 2 | Multi-Label Classification | `src/stage2_multilabel/`, `notebooks/03_*.ipynb` |
| 3 | Localization | `src/stage3_localization/`, `notebooks/07_*.ipynb` |
| 4 | Geometry Alignment | `src/stage4_alignment/`, `notebooks/05_*.ipynb` |
| 5 | Robustness | `src/stage5_robustness/`, `notebooks/06_*.ipynb` |

## 📋 Rules & Guidelines


## 🛠️ Development Tips

### Data Loading in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

# Access your data
DATA_PATH = '/content/drive/MyDrive/competition/data/'
```

### Model Checkpointing

```python
# Save to Google Drive
import os
os.makedirs('/content/drive/MyDrive/competition/models/', exist_ok=True)
torch.save(model.state_dict(), '/content/drive/MyDrive/competition/models/stage1.pth')
```

### GPU in Colab

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## 📝 Notes

- This repository is for reference and learning
- All development should happen in your personal Colab notebooks
- Product integration uses the patterns shown in product template
- Baseline heuristics in product template should be replaced with trained models

