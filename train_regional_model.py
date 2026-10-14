"""
=============================================================================================
TRAIN_REGIONAL_MODEL.PY - MODULE 1 TRAINING PIPELINE
Regional Language Text Fake News Classifier (Hindi, Marathi, Telugu, Gujarati)
=============================================================================================
Final Year MCA Capstone Project
Dataset: data/master_dataset_balanced.csv (5,000 Balanced Indic Articles)
Target:  label (0 = Real / Credible, 1 = Fake / Fabricated)
=============================================================================================
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Import preprocessing utilities
from utils import clean_indic_text, detect_indic_language

# Ensure proper encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainRegionalModel")


def print_ascii_banner(title: str):
    width = 85
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width)


def build_vectorizer() -> FeatureUnion:
    """
    Builds a dual-stream TF-IDF Vectorizer:
    1. Word-level N-Grams (1-2) to capture distinct vocabulary phrases
    2. Character Subword N-Grams inside word boundaries (char_wb, 2-5)
       to capture Indic morphological roots, inflections, and agglutinative affixes.
    """
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=2
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=12000,
        sublinear_tf=True,
        min_df=2
    )
    return FeatureUnion([
        ("word_tfidf", word_vec),
        ("char_tfidf", char_vec)
    ])


def train_regional_text_model(
    data_path: str = "data/master_dataset_balanced.csv",
    output_dir: str = "models",
    classifier_type: str = "logistic_regression"
) -> Dict[str, Any]:
    """
    Loads master_dataset_balanced.csv, preprocesses the multilingual text,
    trains a robust classifier, evaluates performance across all 4 Indic languages,
    and saves trained vectorizers, models, and metric summaries.
    """
    print_ascii_banner("MODULE 1: TRAINING REGIONAL INDIC TEXT FAKE NEWS CLASSIFIER")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")
        
    logger.info(f"Loading balanced Indic dataset: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df):,} total records.")
    
    # Validate required columns
    required_cols = ["text", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is required but not found in dataset!")
            
    # Text Preprocessing
    logger.info("Preprocessing and normalizing Indic text corpus...")
    df["clean_text"] = df["text"].astype(str).apply(clean_indic_text)
    
    # Filter empty or trivial rows
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    
    # Clean language column names
    if "language" in df.columns:
        df["lang_clean"] = df["language"].astype(str).apply(
            lambda x: x.replace("_F&R_News", "").replace("_News", "").strip()
        )
    else:
        df["lang_clean"] = df["clean_text"].apply(detect_indic_language)
        
    print(f"\n📊 DATASET DISTRIBUTION OVERVIEW:")
    print(f"Total Usable Articles: {len(df):,}")
    print("\nBreakdown by Language:")
    for lang, count in df["lang_clean"].value_counts().items():
        print(f"  • {lang:<12}: {count:,} articles ({count/len(df)*100:.1f}%)")
        
    print("\nBreakdown by Class Label:")
    real_cnt = sum(df["label"] == 0)
    fake_cnt = sum(df["label"] == 1)
    print(f"  • Label 0 (Real News): {real_cnt:,} ({real_cnt/len(df)*100:.1f}%)")
    print(f"  • Label 1 (Fake News): {fake_cnt:,} ({fake_cnt/len(df)*100:.1f}%)")
    
    X = df["clean_text"]
    y = df["label"].astype(int)
    
    # Stratified Train/Test Split (80% Train, 20% Holdout Test)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y
    )
    
    logger.info(f"Train Set: {len(X_train):,} samples | Holdout Test Set: {len(X_test):,} samples")
    
    # Initialize TF-IDF Vectorizer
    logger.info("Extracting Dual-Stream Word + Subword Character TF-IDF features...")
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logger.info(f"Extracted feature matrix shape: {X_train_vec.shape}")
    
    # Initialize Classifier
    if classifier_type == "random_forest":
        logger.info("Initializing Random Forest Classifier...")
        clf = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)
    else:
        logger.info("Initializing L2-Regularized Logistic Regression Classifier (C=2.0)...")
        clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42, n_jobs=-1)
        
    # Fit Classifier on Training Data
    logger.info("Fitting model on training split...")
    clf.fit(X_train_vec, y_train)
    
    # Holdout Predictions & Evaluation
    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)[:, 1] if hasattr(clf, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, target_names=["Real News (0)", "Fake News (1)"])
    
    print_ascii_banner("HOLDOUT TEST EVALUATION RESULTS (N = 1,000)")
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Overall Precision: {prec * 100:.2f}%")
    print(f"Overall Recall:    {rec * 100:.2f}%")
    print(f"Overall F1-Score:  {f1 * 100:.2f}%\n")
    
    print("📋 DETAILED CLASSIFICATION REPORT:")
    print(cls_report)
    
    print("🎯 CONFUSION MATRIX:")
    print(f"                Predicted Real    Predicted Fake")
    print(f"  Actual Real :      {cm[0][0]:<15} {cm[0][1]}")
    print(f"  Actual Fake :      {cm[1][0]:<15} {cm[1][1]}\n")
    
    # Per-Language Accuracy Breakdown
    df_test = df.loc[idx_test].copy()
    df_test["predicted"] = y_pred
    
    print("🌐 PER-LANGUAGE PERFORMANCE BREAKDOWN:")
    print(f"{'Language':<15} {'Test Samples':<15} {'Accuracy (%)':<15} {'Status'}")
    print("-" * 55)
    lang_metrics = {}
    for lang, group in df_test.groupby("lang_clean"):
        lang_acc = accuracy_score(group["label"], group["predicted"])
        lang_metrics[lang] = {
            "test_samples": int(len(group)),
            "accuracy": float(round(lang_acc, 4)),
            "real_count": int(sum(group["label"] == 0)),
            "fake_count": int(sum(group["label"] == 1))
        }
        print(f"{lang:<15} {len(group):<15} {lang_acc*100:>6.2f}%         {'✅ Passed' if lang_acc >= 0.85 else '⚠️ Review'}")
        
    # Fit Production Pipeline on 100% of Data
    print_ascii_banner("SERIALIZING PRODUCTION MODEL WEIGHTS")
    logger.info("Refitting vectorizer and classifier on full 5,000 balanced records...")
    full_vectorizer = build_vectorizer()
    X_full_vec = full_vectorizer.fit_transform(X)
    
    if classifier_type == "random_forest":
        full_clf = RandomForestClassifier(n_estimators=150, max_depth=25, random_state=42, n_jobs=-1)
    else:
        full_clf = LogisticRegression(C=2.0, max_iter=1000, random_state=42, n_jobs=-1)
    full_clf.fit(X_full_vec, y)
    
    # Assemble Unified End-to-End Pipeline
    production_pipeline = Pipeline([
        ("vectorizer", full_vectorizer),
        ("classifier", full_clf)
    ])
    
    # Ensure models directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Vectorizer Separately
    vec_path = os.path.join(output_dir, "regional_vectorizer.joblib")
    joblib.dump(full_vectorizer, vec_path)
    logger.info(f"Saved Vectorizer: {vec_path}")
    
    # 2. Save Classifier Weights Separately
    clf_path = os.path.join(output_dir, "regional_classifier.joblib")
    joblib.dump(full_clf, clf_path)
    logger.info(f"Saved Classifier: {clf_path}")
    
    # 3. Save Unified Production Pipeline Bundle
    pipe_path = os.path.join(output_dir, "regional_pipeline.joblib")
    artifact_bundle = {
        "pipeline": production_pipeline,
        "classes": [0, 1],
        "label_names": {0: "Real News", 1: "Fake News"},
        "supported_languages": ["Hindi", "Marathi", "Telugu", "Gujarati"],
        "training_dataset": os.path.basename(data_path),
        "total_training_samples": len(df),
        "accuracy": float(round(acc, 4)),
        "f1_score": float(round(f1, 4)),
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(artifact_bundle, pipe_path)
    logger.info(f"Saved Pipeline Bundle: {pipe_path}")
    
    # 4. Save Structured Metrics JSON for Viva / Report
    metrics_path = os.path.join(output_dir, "regional_model_metrics.json")
    metrics_data = {
        "model_architecture": f"Dual-Stream Subword TF-IDF + {classifier_type.replace('_', ' ').title()}",
        "dataset": data_path,
        "total_samples": len(df),
        "test_holdout_samples": len(X_test),
        "overall_test_accuracy": float(round(acc, 4)),
        "overall_test_precision": float(round(prec, 4)),
        "overall_test_recall": float(round(rec, 4)),
        "overall_test_f1": float(round(f1, 4)),
        "confusion_matrix": cm.tolist(),
        "per_language_performance": lang_metrics,
        "timestamp": datetime.now().isoformat()
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Performance Metrics: {metrics_path}")
    
    print("\n✅ Module 1 Training Pipeline Successfully Finished!")
    return artifact_bundle


def test_sample_prediction(text: str, model_bundle_path: str = "models/regional_pipeline.joblib"):
    """
    Utility to test the saved model on a single input sentence.
    """
    if not os.path.exists(model_bundle_path):
        print(f"Model file {model_bundle_path} not found.")
        return
        
    bundle = joblib.load(model_bundle_path)
    pipeline = bundle["pipeline"]
    
    clean_t = clean_indic_text(text)
    detected_lang = detect_indic_language(text)
    probs = pipeline.predict_proba([clean_t])[0]
    pred_label = int(np.argmax(probs))
    
    verdict = bundle["label_names"][pred_label]
    conf = probs[pred_label] * 100.0
    
    print(f"\n[Test Inference]")
    print(f"  Input Text: {text[:80]}...")
    print(f"  Detected Language: {detected_lang}")
    print(f"  Predicted Verdict: {verdict} (Confidence: {conf:.1f}%)")
    print(f"  Probabilities: Real = {probs[0]*100:.1f}%, Fake = {probs[1]*100:.1f}%")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    master_csv = os.path.join(current_dir, "data", "master_dataset_balanced.csv")
    models_dir = os.path.join(current_dir, "models")
    
    train_regional_text_model(master_csv, models_dir, classifier_type="logistic_regression")
    
    # Run verification test on a sample fake Telugu sentence
    sample_fake = "పాక్ వ్యాలీలోని ట్రాఫిక్ జామ్ యొక్క వీడియో హిమాచల్ రాకెస్లైడ్ అని తప్పుగా పంచుకున్నారు"
    test_sample_prediction(sample_fake, os.path.join(models_dir, "regional_pipeline.joblib"))
