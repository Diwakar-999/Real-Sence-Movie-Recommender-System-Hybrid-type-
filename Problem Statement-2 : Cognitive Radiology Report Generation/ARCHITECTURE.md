# Architecture Documentation
## Cognitive Radiology Report Generation System

---

## 📐 System Architecture Overview

This document provides a comprehensive technical breakdown of the implemented radiology report generation system, analyzing both the **current baseline implementation** and the **required cognitive modules** per competition specifications.

---

## 🏛️ Current Implementation Architecture

### High-Level Pipeline

```
Input Image (224×224×1)
        ↓
   CNN Encoder (VGG-style)
        ↓
Feature Map (7×7×512) → Reshape → (49×512)
        ↓
   Attention Mechanism ←──────┐
        ↓                     │
Text Embeddings (256-dim) ────┘
        ↓
   LSTM Decoder (512 units)
        ↓
Softmax Output (vocab_size=2000)
        ↓
Generated Report Token
```

---

## 1️⃣ Visual Feature Encoder

### Architecture: VGG-Inspired CNN

#### Layer Configuration
```python
Input: (224, 224, 1) - Grayscale chest X-ray

Block 1:
├── Conv2D(64, 3×3, ReLU, same padding)
├── Conv2D(64, 3×3, ReLU, same padding)
└── MaxPooling2D(2×2) → Output: 112×112×64

Block 2:
├── Conv2D(128, 3×3, ReLU, same padding)
├── Conv2D(128, 3×3, ReLU, same padding)
└── MaxPooling2D(2×2) → Output: 56×56×128

Block 3:
├── Conv2D(256, 3×3, ReLU, same padding)
├── Conv2D(256, 3×3, ReLU, same padding)
├── Conv2D(256, 3×3, ReLU, same padding)
└── MaxPooling2D(2×2) → Output: 28×28×256

Block 4:
├── Conv2D(512, 3×3, ReLU, same padding)
├── Conv2D(512, 3×3, ReLU, same padding)
├── Conv2D(512, 3×3, ReLU, same padding)
└── MaxPooling2D(2×2) → Output: 14×14×512

Block 5:
├── Conv2D(512, 3×3, ReLU, same padding)
├── Conv2D(512, 3×3, ReLU, same padding)
├── Conv2D(512, 3×3, ReLU, same padding)
└── MaxPooling2D(2×2) → Output: 7×7×512

Reshape Layer: (7×7×512) → (49, 512)
```

#### Design Rationale
- **Progressive Downsampling:** 5 pooling operations (224→7) provide hierarchical feature abstraction
- **Channel Expansion:** Filters increase (64→512) to capture complex medical patterns
- **Spatial Preservation:** 7×7 grid maintains positional information for attention mechanism
- **Medical Relevance:** Deep features capture anatomical structures (ribs, heart border, lung fields)

#### Output Characteristics
- **Shape:** `(batch_size, 49, 512)`
- **Semantic Meaning:** Each of 49 spatial locations represents a 32×32 pixel receptive field in the original image
- **Feature Richness:** 512-dimensional vectors encode multi-scale texture, edge, and anatomical patterns

---

## 2️⃣ Attention Mechanism

### Bahdanau Attention Implementation

#### Mathematical Formulation

**Score Function:**
```
score(h_t, s_j) = V^T · tanh(W1·h_t + W2·s_j)

where:
  h_t = decoder hidden state at timestep t
  s_j = encoder output at spatial location j
  W1, W2 = learnable weight matrices (dense layers)
  V = learnable projection vector
```

**Attention Weights:**
```
α_tj = softmax(score(h_t, s_j)) over all j ∈ {1...49}
```

**Context Vector:**
```
c_t = Σ(α_tj · s_j)  # Weighted sum of encoder features
```

#### Implementation Details

```python
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units=512):
        super().__init__()
        self.W1 = Dense(units)  # Projects encoder features
        self.W2 = Dense(units)  # Projects decoder hidden state
        self.V = Dense(1)       # Scalar attention score
    
    def call(self, encoder_output, hidden_state):
        # encoder_output: (batch, 49, 512)
        # hidden_state: (batch, 512)
        
        # Expand hidden state for broadcasting
        hidden_with_time = tf.expand_dims(hidden_state, 1)  # (batch, 1, 512)
        
        # Compute attention scores
        score = tf.nn.tanh(
            self.W1(encoder_output) +        # (batch, 49, units)
            self.W2(hidden_with_time)        # (batch, 1, units)
        )  # Broadcasting → (batch, 49, units)
        
        # Attention weights
        attention_weights = tf.nn.softmax(self.V(score), axis=1)  # (batch, 49, 1)
        
        # Context vector
        context_vector = attention_weights * encoder_output  # (batch, 49, 512)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, 512)
        
        return context_vector, attention_weights
```

#### Attention Integration Strategy

**Current Approach (Simplified):**
```python
# Compute initial context using mean pooling
hidden_state = tf.reduce_mean(encoder_output, axis=1)  # (batch, 512)

# Apply attention once to get global context
context_vector, _ = attention(encoder_output, hidden_state)

# Expand context to all decoder timesteps
context_expanded = tf.repeat(
    tf.expand_dims(context_vector, 1),
    repeats=max_len - 1,
    axis=1
)  # (batch, max_len-1, 512)

# Concatenate with word embeddings
decoder_input = tf.concat([context_expanded, embeddings], axis=-1)
# decoder_input: (batch, max_len-1, 512+256)
```

**Clinical Interpretation:**
- Attention weights indicate which image regions influenced each word
- High attention on "lung fields" → words like "clear", "consolidation"
- High attention on "cardiac silhouette" → words like "cardiomegaly", "normal heart size"

---

## 3️⃣ Text Generation Decoder

### LSTM-Based Sequence Decoder

#### Architecture Components

**Embedding Layer:**
```python
Embedding(
    input_dim=2000,      # Vocabulary size
    output_dim=256,      # Embedding dimension
    mask_zero=True       # Ignore padding tokens
)
```

**LSTM Cell:**
```python
LSTM(
    units=512,              # Hidden state dimension
    return_sequences=True,  # Output at every timestep
    return_state=True       # Return final hidden/cell states
)
```

**Output Projection:**
```python
Dense(
    units=2000,          # Vocabulary size
    activation='softmax' # Probability distribution over words
)
```

#### Decoding Process Flow

```
Step t:
1. Input: decoder_input_t (word index)
2. Embedding: word_t = Embedding[decoder_input_t]  # (256,)
3. Context: c_t from attention mechanism              # (512,)
4. Concatenation: x_t = [c_t; word_t]                # (768,)
5. LSTM: h_t, state_t = LSTM(x_t, state_{t-1})       # (512,)
6. Projection: logits_t = Dense(h_t)                  # (2000,)
7. Prediction: word_{t+1} = argmax(softmax(logits_t))
```

#### Training vs. Inference

**Training Mode (Teacher Forcing):**
```python
# All decoder inputs are ground truth
decoder_input = report_tokens[:-1]   # "<start> the heart is ..."
decoder_output = report_tokens[1:]   # "the heart is ... <end>"

# Parallel computation across all timesteps
predictions = model([image_features, decoder_input])
loss = sparse_categorical_crossentropy(decoder_output, predictions)
```

**Inference Mode (Autoregressive):**
```python
sequence = [START_TOKEN]

for t in range(max_len):
    # Predict next token given current sequence
    next_token_probs = model.predict([image_features, sequence])
    next_token = argmax(next_token_probs[0, len(sequence)-1])
    
    if next_token == END_TOKEN:
        break
    
    sequence.append(next_token)

report = decode(sequence)
```

---

## 4️⃣ Training Strategy

### K-Fold Cross-Validation Framework

#### Configuration
```python
n_splits = 5
shuffle = True
random_state = 42

Fold Distribution:
├── Fold 1: 80% train (3,164 samples) | 20% val (791 samples)
├── Fold 2: 80% train (3,164 samples) | 20% val (791 samples)
├── Fold 3: 80% train (3,164 samples) | 20% val (791 samples)
├── Fold 4: 80% train (3,164 samples) | 20% val (791 samples)
└── Fold 5: 80% train (3,164 samples) | 20% val (791 samples)
```

#### Training Loop Per Fold
```python
for fold in range(1, 6):
    # 1. Build fresh model (avoid weight carry-over)
    model = build_model()
    
    # 2. Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Train
    history = model.fit(
        [X_train_enc, X_train_dec],
        y_train,
        validation_data=([X_val_enc, X_val_dec], y_val),
        epochs=20,
        batch_size=32,
        callbacks=[
            EarlyStopping(patience=5),
            ModelCheckpoint(f'model_fold_{fold}.keras')
        ]
    )
    
    # 4. Evaluate
    val_loss, val_acc = model.evaluate([X_val_enc, X_val_dec], y_val)
    fold_metrics.append((val_loss, val_acc))
```

### Loss Function

**Sparse Categorical Crossentropy:**
```
L = - Σ_t Σ_w y_{t,w} · log(ŷ_{t,w})

where:
  y_{t,w} = 1 if word w is correct at timestep t, else 0
  ŷ_{t,w} = predicted probability of word w at timestep t
```

**Properties:**
- Penalizes incorrect predictions logarithmically
- Suitable for next-word prediction task
- Sparse format: labels are integers, not one-hot vectors

---

## 🧠 Required Cognitive Modules (Competition Spec)

### Module 1: PRO-FA (Hierarchical Visual Perception)

#### Concept
Radiologists analyze images at multiple scales:
- **Organ-level:** "Is the heart enlarged?"
- **Region-level:** "Is there opacity in the left lower lobe?"
- **Pixel-level:** "Are there fine nodular densities?"

#### Implementation Requirements

**Multi-Scale Feature Extraction:**
```python
# Pixel-level features (early layers)
pixel_features = encoder.get_layer('conv2d_2').output  # (112×112×64)

# Region-level features (mid layers)
region_features = encoder.get_layer('conv2d_6').output  # (28×28×256)

# Organ-level features (deep layers)
organ_features = encoder.get_layer('conv2d_12').output  # (7×7×512)
```

**RadLex Embedding Alignment:**
```python
# RadLex: Radiology Lexicon - standardized medical terminology
radlex_embeddings = load_radlex_embeddings()  # Pre-trained concept vectors

# Align visual features with medical concepts
for scale in [pixel, region, organ]:
    aligned_features = align_visual_to_radlex(scale, radlex_embeddings)
    # Ensures "lung" features are similar to RadLex "lung" embedding
```

**Attention Pooling:**
```python
class HierarchicalAttentionPooling(Layer):
    def __init__(self):
        self.pixel_attn = SelfAttention(units=64)
        self.region_attn = SelfAttention(units=256)
        self.organ_attn = SelfAttention(units=512)
    
    def call(self, multi_scale_features):
        pixel_pool = self.pixel_attn(multi_scale_features['pixel'])
        region_pool = self.region_attn(multi_scale_features['region'])
        organ_pool = self.organ_attn(multi_scale_features['organ'])
        
        return concatenate([pixel_pool, region_pool, organ_pool])
```

#### Current Implementation Gap
❌ Single-scale features (only 7×7×512 organ-level)  
❌ No RadLex embedding integration  
❌ Fixed spatial pooling (no learned attention)

---

### Module 2: MIX-MLP (Knowledge-Enhanced Classification)

#### Concept
Before generating text, doctors form a "mental checklist":
- ✓ Pneumonia: Positive
- ✓ Cardiomegaly: Negative
- ✓ Pleural Effusion: Positive

This structured reasoning improves report accuracy.

#### Implementation Requirements

**Multi-Label Classification Branch:**
```python
# CheXpert 14 pathology labels
labels = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
    'Pleural Other', 'Fracture', 'Support Devices'
]

# Classification head
class_logits = MultiLabelClassifier(encoder_features)  # (batch, 14)
class_probs = sigmoid(class_logits)  # Independent binary classifiers
```

**MIX-MLP Architecture:**
```python
class MIXMLP(Layer):
    def __init__(self, hidden_dim=512):
        # Residual path (identity + projection)
        self.residual = Dense(hidden_dim)
        
        # Expansion path (feature mixing)
        self.expand1 = Dense(hidden_dim * 2, activation='relu')
        self.expand2 = Dense(hidden_dim, activation='relu')
        self.dropout = Dropout(0.3)
    
    def call(self, x):
        # Path 1: Residual
        residual = self.residual(x)
        
        # Path 2: Expansion
        expanded = self.expand1(x)
        expanded = self.dropout(expanded)
        expanded = self.expand2(expanded)
        
        # Combine paths
        return residual + expanded
```

**Label-Guided Decoding:**
```python
# Concatenate predicted labels with decoder inputs
label_embedding = Embedding(num_labels=14, embed_dim=64)(class_probs)
decoder_input = concatenate([
    image_features,      # (512,)
    label_embedding,     # (64,)
    word_embedding       # (256,)
])  # Total: (832,)
```

#### Current Implementation Gap
❌ No disease classification branch  
❌ No label-aware feature fusion  
❌ No explicit clinical reasoning pathway

---

### Module 3: RCTA (Triangular Cognitive Attention)

#### Concept
Radiologists use a verification loop:
1. **Observe Image** → Form initial impression
2. **Check History** → Contextualize findings
3. **Form Hypothesis** → "Patient likely has X"
4. **Verify with Image** → Re-examine to confirm

This closed-loop process reduces errors.

#### Implementation Requirements

**Three-Way Attention Mechanism:**
```python
class TriangularCognitiveAttention(Layer):
    def __init__(self):
        self.image_text_attn = CrossAttention()  # Image queries Clinical Text
        self.text_label_attn = CrossAttention()  # Context queries Labels
        self.label_image_attn = CrossAttention() # Hypothesis queries Image
    
    def call(self, image_features, clinical_text, predicted_labels):
        # Stage 1: Image → Text (Context Formation)
        context = self.image_text_attn(
            query=image_features,
            key_value=clinical_text
        )
        
        # Stage 2: Context → Labels (Hypothesis Generation)
        hypothesis = self.text_label_attn(
            query=context,
            key_value=predicted_labels
        )
        
        # Stage 3: Hypothesis → Image (Verification)
        verified_features = self.label_image_attn(
            query=hypothesis,
            key_value=image_features
        )
        
        return verified_features  # Used for final text generation
```

**Attention Flow Diagram:**
```
       Image Features
            ↓
    ┌───────┴───────┐
    ↓               ↓
Clinical Text   Predicted Labels
    ↓               ↓
  Context      Hypothesis
    └───────┬───────┘
            ↓
    Verified Features
            ↓
    Report Generation
```

#### Current Implementation Gap
❌ Single attention path (image → decoder only)  
❌ No clinical indication processing  
❌ No hypothesis verification loop

---

## 📊 Comparison: Current vs. Required

| Component | Current Implementation | Required Implementation | Status |
|-----------|------------------------|-------------------------|--------|
| **Visual Encoder** | Single-scale CNN (7×7×512) | Multi-scale PRO-FA (Pixel/Region/Organ) | ⚠️ Partial |
| **Medical Alignment** | None | RadLex embeddings | ❌ Missing |
| **Classification** | None | MIX-MLP (14 CheXpert labels) | ❌ Missing |
| **Attention** | Bahdanau (image-to-text) | RCTA (triangular loop) | ⚠️ Partial |
| **Clinical Reasoning** | Implicit (learned) | Explicit (3-stage verification) | ❌ Missing |
| **Input Modalities** | Image only | Image + Clinical Indication | ⚠️ Partial |

---

## 🔧 Technical Specifications

### Model Complexity

**Current Model:**
```
Total Parameters: ~15M
├── CNN Encoder: ~14M (VGG-16 style)
├── Attention: ~0.5M (3 dense layers)
├── Decoder LSTM: ~1M (512 units, vocab 2000)
└── Embeddings: ~0.5M (2000 × 256)

Training Time: ~2-3 hours (5 folds × 20 epochs on GPU)
Inference Speed: ~500ms per image (single report)
```

**Required Model (with all modules):**
```
Estimated Parameters: ~25-30M
├── PRO-FA Encoder: ~16M (multi-scale + alignment)
├── MIX-MLP Classifier: ~2M (14 disease heads)
├── RCTA Attention: ~3M (triangular cross-attention)
├── Decoder LSTM: ~2M (enhanced with label inputs)
└── Embeddings: ~2M (RadLex + word embeddings)
```

### Memory Requirements

**Training (Batch Size 32):**
```
GPU Memory: ~6GB
├── Image Batch: 32 × 224 × 224 × 1 = 1.6M pixels → ~6MB
├── Feature Maps: 32 × 49 × 512 = 800K floats → ~3.2MB
├── Gradients: ~2× model size → ~30-60MB (with all modules)
└── Optimizer States (Adam): ~2× parameters → ~100-120MB
```

**Inference:**
```
GPU Memory: ~500MB
├── Single Image: 224 × 224 × 1 → ~200KB
├── Model Weights: ~15M params × 4 bytes → ~60MB
└── Activations: ~100MB
```

---

## 🎯 Roadmap to Full Compliance

### Phase 1: Multi-Scale Visual Features (PRO-FA)
1. Extract features from 3 encoder layers (early/mid/late)
2. Implement spatial attention pooling for each scale
3. Integrate RadLex embeddings (download from RadLex.org)
4. Train alignment loss: `L_align = MSE(visual_features, radlex_embedding)`

### Phase 2: Disease Classification (MIX-MLP)
1. Implement 14-class multi-label classifier
2. Use CheXpert labeler to auto-label training data
3. Add classification loss: `L_class = BCE(predicted_labels, true_labels)`
4. Concatenate label embeddings to decoder inputs

### Phase 3: Triangular Attention (RCTA)
1. Parse clinical indication text (if available)
2. Implement 3-stage cross-attention module
3. Replace current attention with RCTA
4. Add verification loss to encourage closed-loop reasoning

### Phase 4: Evaluation Integration
1. Implement CheXpert F1 scorer (use `chexpert-labeler` package)
2. Implement RadGraph F1 scorer (use `radgraph` package)
3. Compute BLEU-4, CIDEr, METEOR scores
4. Benchmark against competition targets

---

## 📝 Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| `s_j` | Encoder feature at spatial location j |
| `h_t` | Decoder hidden state at timestep t |
| `c_t` | Context vector at timestep t |
| `α_tj` | Attention weight for location j at timestep t |
| `W1, W2, V` | Attention mechanism parameters |
| `e_w` | Word embedding vector |
| `y_t` | Ground truth word at timestep t |
| `ŷ_t` | Predicted word distribution at timestep t |

---

## 🔗 Dependencies

```yaml
Core Libraries:
  - TensorFlow >= 2.10
  - NumPy >= 1.21
  - OpenCV >= 4.5
  - Scikit-learn >= 1.0

Data Processing:
  - Pandas >= 1.3
  - Pillow >= 8.0

Evaluation (Required):
  - chexpert-labeler
  - radgraph
  - pycocoevalcap (for CIDEr/BLEU)

Medical Ontology:
  - radlex (RadLex embeddings)
  - umls (Unified Medical Language System)
```

---

## 📚 References

1. **Bahdanau Attention:**  
   Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.

2. **Show, Attend and Tell:**  
   Xu, K., et al. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.* ICML.

3. **CheXpert:**  
   Irvin, J., et al. (2019). *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison.* AAAI.

4. **RadGraph:**  
   Jain, S., et al. (2021). *RadGraph: Extracting Clinical Entities and Relations from Radiology Reports.* NeurIPS.

5. **Hi-CliTr Framework:**  
   (Competition-specific reference - inspired by hierarchical clinical reasoning transformers)

---

**Last Updated:** February 2026  
**Status:** Baseline Implementation Complete | Cognitive Modules Pending  
**Next Steps:** Integrate PRO-FA → MIX-MLP → RCTA modules for full competition compliance
