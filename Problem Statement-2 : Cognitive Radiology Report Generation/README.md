# Cognitive Radiology Report Generation

## 🎯 Project Overview

This project implements an AI-powered "Second Reader" system for automated radiology report generation from chest X-ray images. The solution addresses the critical challenge of reducing the 3-5% discrepancy rate in human-generated radiology reports caused by reader fatigue in high-pressure clinical environments.

**Challenge:** BrainDead Hackathon - Problem Statement 2  
**Marks:** 60  
**Competition Dates:** February 6-8, 2026  
**Dataset:** IU-Xray (Indiana University Chest X-Ray Collection)

---

## 📊 Problem Statement

Develop a Deep Learning framework that:
- **Ingests:** Chest X-Ray images (PA/Lateral views) and clinical indication text
- **Processes:** Aligns visual features with medical ontology and mimics clinical reasoning
- **Outputs:** Structured, clinically accurate radiology reports (Findings and Impression sections)

### Key Innovation Requirements
The solution must demonstrate "Cognitive Simulation" inspired by the Hi-CliTr framework:
1. **Hierarchical Visual Perception (PRO-FA)** - Multi-granularity feature extraction
2. **Knowledge-Enhanced Classification (MIX-MLP)** - Disease prediction before text generation
3. **Triangular Cognitive Attention (RCTA)** - Closed-loop verification mechanism

---

## 🏗️ Architecture Overview

Our implementation uses an **Encoder-Decoder architecture with Bahdanau Attention** mechanism:

### 1. **Visual Encoder (CNN-based)**
- Deep convolutional neural network based on VGG-style architecture
- Extracts spatial features from 224×224 grayscale chest X-ray images
- Output: 49×512 feature map (7×7 spatial grid with 512-dimensional features)

### 2. **Attention Mechanism**
- **Bahdanau Attention** layer for dynamic focus on relevant image regions
- Computes context vectors by attending to encoder features at each decoding step
- Enables the model to "look" at different parts of the image while generating text

### 3. **Text Decoder (LSTM-based)**
- Embedding layer (vocabulary size: 2000, embedding dim: 256)
- LSTM decoder (512 hidden units) with attention-augmented inputs
- Generates radiology findings word-by-word autoregressively
- Softmax output layer for next-word prediction

---

## 📂 Dataset Details

### IU-Xray Dataset
- **Total Reports:** 3,955 unique radiology reports
- **Images:** Frontal and lateral chest X-ray projections
- **Report Structure:**
  - `findings`: Detailed observations of the scan (Primary Target)
  - `impression`: Summary diagnosis
  - `indication`: Patient symptoms/history (Future enhancement)

### Data Files
- `indiana_reports.csv`: Contains radiology report text (findings, impression, indication)
- `indiana_projections.csv`: Contains image metadata (filename, projection type, uid)
- `images1/`: Directory containing normalized PNG chest X-ray images

---

## 🛠️ Technical Implementation

### Preprocessing Pipeline

#### Image Processing
```python
- Input: Grayscale PNG images
- Resize: 224×224 pixels
- Normalization: (pixel/255.0 - mean) / std
  - mean = 0.485
  - std = 0.229
- Channel: Single-channel grayscale (H×W×1)
```

#### Text Processing
```python
- Text cleaning: Lowercase, remove special characters
- Tokenization: Keras Tokenizer with 2000 vocabulary size
- Special tokens: <start>, <end>, <unk>
- Sequence padding: max_length = 200
- Format: "<start> {findings} <end>"
```

### Model Architecture Components

#### CNN Encoder
- **5 Convolutional Blocks** (VGG-inspired):
  - Block 1: 2×Conv(64) + MaxPool
  - Block 2: 2×Conv(128) + MaxPool
  - Block 3: 3×Conv(256) + MaxPool
  - Block 4: 3×Conv(512) + MaxPool
  - Block 5: 3×Conv(512) + MaxPool
- **Output:** Spatial feature map reshaped to (49, 512)

#### Attention-Augmented Decoder
```python
- Embedding: vocab_size=2000, output_dim=256
- Attention: BahdanauAttention(units=512)
  - Score computation: tanh(W1*encoder + W2*hidden)
  - Context vector: weighted sum of encoder features
- LSTM: 512 units, return_sequences=True
- Dense Output: softmax over vocabulary
```

### Training Configuration

#### K-Fold Cross-Validation
- **Folds:** 5-fold cross-validation
- **Train-Val Split:** Stratified random splits
- **Purpose:** Robust performance estimation and model selection

#### Hyperparameters
```yaml
Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Batch Size: 32
Epochs: 20 per fold
Max Sequence Length: 200 tokens
Vocabulary Size: 2000
Embedding Dimension: 256
LSTM Hidden Units: 512
Attention Units: 512
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- Kaggle API credentials

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/cognitive-radiology-report.git
cd cognitive-radiology-report
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
```bash
# Set up Kaggle API credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download IU-Xray dataset
kaggle datasets download -d raddar/chest-xrays-indiana-university
unzip chest-xrays-indiana-university.zip
```

### Step 4: Run Training
```bash
jupyter notebook Cognitive_Radiology_Report_Generation.ipynb
```

---

## 📊 Performance Metrics

### Target Benchmarks (Problem Statement)
| Metric Category | Metrics | Weight | Target |
|----------------|---------|--------|--------|
| Clinical Accuracy | CheXpert F1 | 40% | F1 > 0.500 |
| Structural Logic | RadGraph F1 | 30% | Relations > 0.500 |
| NLG Fluency | CIDEr, BLEU-4 | 30% | CIDEr > 0.400 |

### Current Implementation Status
- ✅ Base architecture implemented
- ✅ Attention mechanism integrated
- ✅ K-fold validation framework
- ⚠️ Advanced cognitive modules (PRO-FA, MIX-MLP, RCTA) - Baseline implementation
- 📋 Evaluation metrics (CheXpert, RadGraph) - Pending integration

---

## 💻 Usage

### Inference Example
```python
from generate_report import generate_report

# Load trained models
encoder_model = tf.keras.models.load_model("encoder_model.keras")
full_model = tf.keras.models.load_model("full_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Generate report for new X-ray
report = generate_report(
    img_path="test_xray.png",
    encoder_model=encoder_model,
    full_model=full_model,
    tokenizer=tokenizer,
    max_len=200
)

print(report)
```

### Expected Output Format
```
Findings:
The heart size is normal. The mediastinal contours are unremarkable. 
The lungs are clear without consolidation or edema. No pneumothorax 
or pleural effusion is identified.
```

---

## 📁 Repository Structure

```
cognitive-radiology-report/
├── data/
│   ├── indiana_reports.csv          # Radiology report texts
│   ├── indiana_projections.csv      # Image metadata
│   └── images1/                     # Chest X-ray images
├── models/
│   ├── encoder_model.keras          # Trained CNN encoder
│   ├── full_model.keras             # Complete end-to-end model
│   └── tokenizer.pkl                # Fitted tokenizer object
├── notebooks/
│   └── Cognitive_Radiology_Report_Generation.ipynb
├── evaluation/
│   ├── chexpert_evaluator.py        # CheXpert F1 scorer (TBD)
│   └── radgraph_evaluator.py        # RadGraph F1 scorer (TBD)
├── utils/
│   ├── preprocessing.py             # Image/text preprocessing
│   └── visualization.py             # Attention visualization
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── ARCHITECTURE.md                   # Detailed architecture documentation
└── REPORT.md                         # Technical implementation report
```

---

## 🧪 Evaluation & Testing

### Planned Evaluation Pipeline
1. **Clinical Accuracy (CheXpert Labeling)**
   - Extract 14 pathology labels from generated reports
   - Compare with ground truth labels
   - Compute precision, recall, F1-score

2. **Structural Correctness (RadGraph)**
   - Parse entity-relation triplets from reports
   - Measure graph-level similarity
   - Evaluate anatomical-observation linkages

3. **NLG Quality Metrics**
   - BLEU-4: N-gram overlap with reference reports
   - CIDEr: Consensus-based image description metric
   - METEOR: Semantic similarity with synonyms

---

## 🔬 Future Enhancements

### Mandatory Cognitive Modules (Per Competition Requirements)

#### 1. PRO-FA (Hierarchical Visual Perception)
- [ ] Multi-scale feature extraction (Pixel, Region, Organ levels)
- [ ] RadLex embedding integration for medical concept alignment
- [ ] Attention pooling at different spatial granularities

#### 2. MIX-MLP (Knowledge-Enhanced Classification)
- [ ] Multi-label disease classification branch (14 CheXpert labels)
- [ ] Residual + Expansion path architecture
- [ ] Label-aware feature fusion before decoding

#### 3. RCTA (Triangular Cognitive Attention)
- [ ] Three-way attention mechanism:
  - Image → Clinical Text → Context
  - Context → Predicted Labels → Hypothesis
  - Hypothesis → Image → Verification
- [ ] Closed-loop reasoning simulation

### Additional Improvements
- [ ] Transformer-based encoder (Vision Transformer)
- [ ] Multi-view fusion (Frontal + Lateral)
- [ ] Clinical indication incorporation
- [ ] Impression section generation
- [ ] Beam search decoding
- [ ] Uncertainty quantification

---

## 📖 References

### Datasets
1. **IU-Xray**: Demner-Fushman et al. (2016) - Indiana University Chest X-Ray Collection
2. **MIMIC-CXR**: Johnson et al. (2019) - MIT Critical Care Database

### Architecture Inspiration
1. **Hi-CliTr Framework**: Hierarchical Clinical Reasoning Transformer
2. **Show, Attend and Tell**: Xu et al. (2015) - Visual Attention in Image Captioning
3. **Bahdanau Attention**: Bahdanau et al. (2015) - Neural Machine Translation by Jointly Learning to Align and Translate

### Evaluation Tools
1. **CheXpert Labeler**: Irvin et al. (2019) - Stanford CheXpert Dataset
2. **RadGraph**: Jain et al. (2021) - Extracting Entity-Relations from Radiology Reports

---

## 👥 Team Information

**Team Name:** [Your Team Name]  
**Competition:** BrainDead Hackathon 2026  
**Platform:** Unstop  

### Contributors
- [Member 1 Name] - Model Architecture & Training
- [Member 2 Name] - Data Preprocessing & Evaluation
- [Member 3 Name] - Deployment & Documentation

---

## 📄 License

This project is developed for the BrainDead Hackathon 2026. All code is provided for educational and competition purposes.

---

## 🙏 Acknowledgments

- Indiana University for the IU-Xray dataset
- PhysioNet/MIT for MIMIC-CXR access
- BrainDead Hackathon organizers
- Open-source deep learning community

---

## 📞 Contact

For questions or collaborations:
- **Email:** [your-email@domain.com]
- **GitHub Issues:** [Repository Issues Page]
- **Unstop Profile:** [Your Unstop Profile Link]

---

**Note:** This implementation represents a baseline solution. To fully meet competition requirements, the three mandatory cognitive modules (PRO-FA, MIX-MLP, RCTA) must be integrated as described in the problem statement and ARCHITECTURE.md file.
