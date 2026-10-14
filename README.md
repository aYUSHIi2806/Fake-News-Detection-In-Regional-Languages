
**Final Year MCA Capstone Project in Artificial Intelligence & Data Science**

---

## 📌 Executive Summary
With the rapid explosion of social media and encrypted messaging platforms (such as WhatsApp, Telegram, and Twitter/X), misinformation spreads exponentially in two insidious forms:
1. **Regional Language Vernacular Disinformation**: Fake claims propagated in Indian regional languages (Hindi, Marathi, Telugu, Gujarati) that easily bypass conventional English-centric content moderation filters.
2. **Cheapfakes / Out-of-Context Multimodal Misinformation**: Real, unedited historical photos paired with fabricated, sensational headlines (e.g., attributing an old gas leak photo to a recent chemical explosion).

**TruthGuard AI** provides a production-grade, dual-module fact-checking architecture:
- **Module 1**: Multilingual Indic Text Fake News Classifier with Unicode Script Auto-Detection and Subword-level Morpho-Syntactic feature modeling.
- **Module 2**: Multimodal (Text + Image) Out-of-Context Verifier with Cross-Modal Semantic Alignment and Error Level Analysis (ELA) compression forensics.

---

## 🏗️ System Architecture Pipeline

```
                                  +------------------------------------+
                                  |    TruthGuard AI Web Application   |
                                  |         (Streamlit Dashboard)      |
                                  +------------------------------------+
                                                     |
                     +-------------------------------+-------------------------------+
                     |                                                               |
                     v                                                               v
      +-----------------------------+                                 +-----------------------------+
      |  MODULE 1: INDIC TEXT NLP   |                                 |  MODULE 2: MULTIMODAL AI    |
      +-----------------------------+                                 +-----------------------------+
                     |                                                               |
    +--------------------------------+                              +--------------------------------+
    | 1. Script & Language Detector  |                              | 1. Error Level Analysis (ELA)  |
    |    (Devanagari, Telugu, Guj)   |                              |    JPEG Delta Forensics        |
    +--------------------------------+                              +--------------------------------+
                     |                                                               |
    +--------------------------------+                              +--------------------------------+
    | 2. Text Normalization Pipeline |                              | 2. Visual Descriptor Engine    |
    |    URLs, Mentions, Punctuation |                              |    Color, Edges, Spatial Scene |
    +--------------------------------+                              +--------------------------------+
                     |                                                               |
    +--------------------------------+                              +--------------------------------+
    | 3. Dual-Stream TF-IDF Union    |                              | 3. Headline Concept Extractor  |
    |    Word (1-2) + Char_wb (2-5)  |                              |    Entity & Topic Mapping      |
    +--------------------------------+                              +--------------------------------+
                     |                                                               |
    +--------------------------------+                              +--------------------------------+
    | 4. Calibrated Ensemble Model   |                              | 4. Cross-Modal Alignment       |
    |    LinearSVC + LogReg + CNB    |                              |    Cosine Context Match Score  |
    +--------------------------------+                              +--------------------------------+
                     |                                                               |
                     v                                                               v
     [Real vs Fake Probability %]                                    [Matched vs Out-of-Context %]
     [Token-Level Explainability]                                    [Tamper Heatmap & Diagnostics]
```

---

## 📁 Project Directory Structure

```
dual_module_fake_news_detector/
│
├── app.py                      # Production Streamlit Web Dashboard (Tabs for Mod 1 & Mod 2)
├── train_text_model.py         # Module 1: Regional text training script & cross-validation
├── multimodal_model.py         # Module 2: Multimodal alignment, ELA forensics & classifier
├── create_sample_assets.py     # Generator for reference demonstration images
├── requirements.txt            # Python dependencies
│
├── data/                       # Dataset directory
│   ├── sample_structure_100.csv       # Module 1 Dataset (Hindi, Marathi, Telugu, Gujarati)
│   ├── multimodal_train_sample.csv    # Module 2 Dataset (Fakeddit Benchmark subset)
│   └── all_train_sample.csv           # Module 2 Extended Benchmark sample
│
├── models/                     # Serialized production models & metrics
│   ├── regional_text_model.joblib     # Trained Module 1 Scikit-Learn Ensemble Pipeline
│   ├── text_model_metrics.json        # Module 1 5-Fold CV and Holdout Evaluation Report
│   ├── multimodal_verifier.joblib     # Trained Module 2 Context Verification Model
│   └── multimodal_metrics.json        # Module 2 Performance Metrics
│
└── sample_images/              # Demonstration images for Module 2 viva testing
    ├── flood_rescue.jpg               # Authentic disaster scene
    ├── protest_rally.jpg              # Authentic crowd demonstration
    ├── vaccine_hospital.jpg           # Authentic medical laboratory scene
    ├── space_rocket.jpg               # Space exploration scene
    └── tampered_document.jpg          # Digitally spliced document for ELA heatmap demo
```

---

## 🚀 Setup & Execution Guide (Windows / VS Code)

### Step 1: Open Terminal in VS Code
Open VS Code, press ``Ctrl + ` `` to open the integrated PowerShell terminal, and navigate to the project directory:
```powershell
cd "C:\Users\91943\.gemini\antigravity\scratch\dual_module_fake_news_detector"
```

### Step 2: Install Dependencies
Run the following command to install all necessary packages:
```powershell
pip install -r requirements.txt
```

### Step 3: Train the Models (Optional - Pre-trained Models already included!)
If you ever want to re-train the models from scratch:
```powershell
# Train Module 1 (Regional Text Detector)
python train_text_model.py

# Train Module 2 (Multimodal Out-of-Context Verifier)
python multimodal_model.py
```

### Step 4: Launch the Streamlit Web Application
Run the Streamlit application:
```powershell
streamlit run app.py
```
Streamlit will automatically launch your default browser at:
`http://localhost:8501`

---

## 🔬 In-Depth Module Specifications

### Module 1: Regional Language Fake News Detector
- **Languages Supported**:
  - **Telugu** (`te`): Script range `\u0C00-\u0C7F`
  - **Hindi** (`hi`): Devanagari script `\u0900-\u097F`
  - **Marathi** (`mr`): Devanagari script disambiguated by grammatical verb markers
  - **Gujarati** (`gu`): Script range `\u0A80-\u0AFF`
- **Feature Engineering**:
  - Indic languages are morphologically rich and agglutinative. Standard word-level models suffer from high out-of-vocabulary (OOV) rates.
  - TruthGuard combines **Word N-Grams (1-2)** with **Character Subword N-Grams inside word boundaries (`char_wb`, 2-5)**. This captures root words, prefixes, and suffixes even with social media spelling variations.
- **Model Ensemble**:
  - Soft-voting ensemble combining **Calibrated LinearSVC**, **L2 Regularized Logistic Regression**, and **Complement Naive Bayes**.
- **Performance**:
  - Cross-Validation Accuracy: **87.00%** (±11.66%)
  - Holdout Test Accuracy: **88.00%** | Precision: **100.00%** | F1-Score: **85.71%**

### Module 2: Multimodal (Text + Image) Out-of-Context Verifier
- **Forensic Image Analysis (Error Level Analysis - ELA)**:
  - Detects digital image tampering, forged badges, and pasted breaking-news banners.
  - Formula:
    $$\Delta(x, y) = |I_{\text{original}}(x, y) - I_{\text{recompressed@Q90}}(x, y)| \times \alpha$$
- **Visual-Semantic Topic Alignment**:
  - Extracts dominant scene themes (e.g., `water_flood`, `protest_crowd`, `health_medical`, `space_astronomy`).
  - Matches inferred visual entities against the news headline claim.
  - Computes a normalized **Cross-Modal Similarity Score** and a **Context Discrepancy Index**.
- **Diagnostic Engine**:
  - Produces human-interpretable explanations detailing why an image and headline are inconsistent.

---

