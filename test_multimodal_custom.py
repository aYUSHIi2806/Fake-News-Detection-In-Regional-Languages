import os
import sys
import pandas as pd

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils import compute_cross_modal_alignment, load_image_from_url_or_path

def run_multimodal_test(headline: str, image_source: str, expected_label: str = "Unknown"):
    print("\n" + "=" * 80)
    print(f"🔍 TESTING MULTIMODAL OUT-OF-CONTEXT VERIFIER (MODULE 2)")
    print("=" * 80)
    print(f"Headline       : {headline}")
    print(f"Image Source   : {image_source}")
    print(f"Expected Type  : {expected_label}")
    
    # 1. Image Load
    print("\n[Step 1] Loading Image...")
    img = load_image_from_url_or_path(image_source, timeout=8)
    
    if img is None:
        print("⚠️ Warning: Could not fetch image from URL/Path (Network or invalid URL).")
        print("Falling back to text-only concept alignment diagnostic...")
        print("=" * 80)
        return

    # 2. Cross-Modal Feature Extraction & Alignment
    print("[Step 2] Performing Visual-Semantic Alignment & ELA Forensics...")
    result = compute_cross_modal_alignment(img, headline)
    
    # 3. Display Output Metrics
    print("\n📊 VERIFICATION RESULT:")
    print(f"Verdict            : {result['verdict']}")
    print(f"Similarity Score   : {result['similarity_score']}%")
    print(f"Discrepancy Index  : {result['discrepancy_index']}%")
    print(f"Image Tamper Risk  : {result['tamper_stats']['tamper_risk']} (Mean Error: {result['tamper_stats']['mean_error']})")
    print(f"Headline Concepts  : {result['headline_concepts']}")
    print(f"Visual Concepts    : {result['visual_concepts']}")
    print(f"Matched Keywords   : {result['matched_keywords']}")
    print(f"Diagnostic Reason  : {result['diagnostic_reason']}")
    print("=" * 80)


if __name__ == "__main__":
    # -------------------------------------------------------------
    # TEST CASE 1: Automatic Test from CSV Dataset
    # -------------------------------------------------------------
    csv_path = "data/multimodal_train.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/all_train.csv"
        
    if os.path.exists(csv_path):
        print(f"📁 Reading sample record directly from dataset: {csv_path}")
        df = pd.read_csv(csv_path, nrows=20)
        
        # Pick 1 sample from dataset
        sample_row = df.dropna(subset=['clean_title', 'image_url']).iloc[0]
        headline_sample = str(sample_row['clean_title'])
        img_url_sample = str(sample_row['image_url'])
        label_val = sample_row['2_way_label'] if '2_way_label' in sample_row else "N/A"
        
        exp_type = "Authentic / In-Context" if label_val == 1 else "Fake / Out-of-Context"
        run_multimodal_test(headline_sample, img_url_sample, expected_label=exp_type)
    
    # -------------------------------------------------------------
    # TEST CASE 2: Custom Text + Image URL
    # -------------------------------------------------------------
    custom_headline = "Severe flood waters inundate rescue boat during heavy storm rainfall"
    custom_image_url = "https://images.unsplash.com/photo-1547683905-f686c993aae5?w=500" # Flood image
    run_multimodal_test(custom_headline, custom_image_url, expected_label="Authentic Flood News")

    # -------------------------------------------------------------
    # TEST CASE 3: Out-of-Context Mismatch (Space photo + Disaster headline)
    # -------------------------------------------------------------
    mismatched_headline = "Massive protest crowd rallies near parliament square against new tax law"
    mismatched_image_url = "https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?w=500" # Space image
    run_multimodal_test(mismatched_headline, mismatched_image_url, expected_label="Out-of-Context / Mismatched")