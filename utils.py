"""
=============================================================================================
UTILS.PY - CORE PREPROCESSING & MULTIMODAL FEATURE UTILITIES
TruthGuard: Dual-Module Fake News Detection (Final Year MCA Project)
=============================================================================================
Provides:
1. Indic Text Preprocessing & Unicode Normalization (Devanagari, Telugu, Gujarati)
2. Automated Language & Script Identification (Hindi, Marathi, Telugu, Gujarati)
3. Image Loading & Error Level Analysis (ELA) for Tamper Verification
4. Visual Feature Descriptors (HSV Color Spaces, Edge Densities, Concept Mapping)
5. Cross-Modal Semantic Alignment & Discrepancy Scoring
=============================================================================================
"""

import io
import os
import re
import sys
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import requests
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat

# Configure UTF-8 encoding for Windows terminal output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TruthGuardUtils")


# =============================================================================================
# 1. TEXT CLEANING & SCRIPT IDENTIFICATION (MODULE 1)
# =============================================================================================

def clean_indic_text(text: str) -> str:
    """
    Cleans and normalizes regional language Indic text:
    - Removes URLs, web links, and HTML markup
    - Cleans social media artifacts (@mentions, hashtags, emojis)
    - Strips noisy punctuation while preserving native Indic Unicode characters
    - Normalizes multi-spaces and standardizes whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove user handles and hashtags
    text = re.sub(r"[@#]\w+", " ", text)
    # Remove email patterns
    text = re.sub(r"\S+@\S+", " ", text)
    # Normalize repetitive punctuation
    text = re.sub(r"([!?.,।॥])\1+", r"\1", text)
    # Strip noisy symbols while preserving Devanagari, Telugu, and Gujarati scripts
    text = re.sub(r"[\_~*^`\(\)\[\]\{\}<>|\\/\"'–—]", " ", text)
    # Normalize multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_indic_language(text: str) -> str:
    """
    Detects the regional language based on Unicode character block ranges
    and distinctive grammatical function words.
    
    Supported:
    - Telugu (Telugu Script: U+0C00 - U+0C7F)
    - Gujarati (Gujarati Script: U+0A80 - U+0AFF)
    - Hindi / Marathi (Devanagari Script: U+0900 - U+097F)
      Disambiguated via grammatical verb and auxiliary marker frequency.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return "Unknown"

    devanagari_count = len(re.findall(r"[\u0900-\u097F]", text))
    gujarati_count = len(re.findall(r"[\u0A80-\u0AFF]", text))
    telugu_count = len(re.findall(r"[\u0C00-\u0C7F]", text))

    counts = {
        "Devanagari": devanagari_count,
        "Gujarati": gujarati_count,
        "Telugu": telugu_count,
    }
    
    dominant_script = max(counts, key=counts.get)
    if counts[dominant_script] == 0:
        return "English / Other"

    if dominant_script == "Telugu":
        return "Telugu"
    elif dominant_script == "Gujarati":
        return "Gujarati"
    elif dominant_script == "Devanagari":
        # Disambiguate Hindi vs Marathi using lexical stopwords / auxiliary verb markers
        marathi_markers = {
            "आहे", "नाही", "झाले", "केले", "म्हणाले", "यांनी", "मध्ये", 
            "होती", "होते", "च्या", "वर", "त्यांच्या", "एकदा", "काय", "आणि", "हे"
        }
        hindi_markers = {
            "है", "नहीं", "हुआ", "कहा", "ने", "में", "थी", "था", 
            "के", "पर", "उनके", "एक", "क्या", "और", "यह", "गया", "गए"
        }
        tokens = set(text.split())
        mr_hits = len(tokens.intersection(marathi_markers))
        hi_hits = len(tokens.intersection(hindi_markers))
        
        if mr_hits > hi_hits:
            return "Marathi"
        return "Hindi"

    return "Unknown"


# =============================================================================================
# 2. IMAGE FORENSICS: ERROR LEVEL ANALYSIS (ELA) (MODULE 2)
# =============================================================================================

def perform_error_level_analysis(
    image: Image.Image,
    quality: int = 90,
    scale: float = 12.0
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Performs Error Level Analysis (ELA) to detect localized digital tampering,
    splicing, or compression layer discrepancies in news images.
    
    Mathematics:
        Delta(x, y) = |I_original(x, y) - I_recompressed@Q(x, y)| * alpha
    """
    img_rgb = image.convert("RGB")
    
    # Save into memory buffer at standard JPEG quality
    buffer = io.BytesIO()
    img_rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    
    # Compute pixel-by-pixel difference
    diff = ImageChops.difference(img_rgb, recompressed)
    
    # Calculate statistical metrics across channels
    stat = ImageStat.Stat(diff)
    mean_diff = float(np.mean(stat.mean))
    rms_diff = float(np.mean(stat.rms))
    
    # Scale difference for visual enhancement
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale_multiplier = 255.0 / max_diff
    
    enhancer = ImageEnhance.Brightness(diff)
    ela_image = enhancer.enhance(scale_multiplier * (scale / 10.0))
    
    # Assess Tampering Risk Level
    if mean_diff > 8.5:
        tamper_risk = "High"
        tamper_desc = "Significant compression anomalies detected; possible digital splicing, overlay, or text edit."
    elif mean_diff > 4.5:
        tamper_risk = "Moderate"
        tamper_desc = "Mild compression variations observed across image regions."
    else:
        tamper_risk = "Low"
        tamper_desc = "Uniform compression levels; image shows consistent single-pass encoding."
        
    stats = {
        "mean_error": round(mean_diff, 2),
        "rms_error": round(rms_diff, 2),
        "tamper_risk": tamper_risk,
        "tamper_description": tamper_desc
    }
    
    return ela_image, stats


def load_image_from_url_or_path(image_source: str, timeout: int = 5) -> Optional[Image.Image]:
    """
    Safely loads a PIL Image from either a web URL or a local file path.
    Includes proper timeout, User-Agent header, and exception handling.
    """
    if not image_source or not isinstance(image_source, str):
        return None
        
    image_source = image_source.strip()
    
    # Check if local file exists
    if os.path.exists(image_source):
        try:
            return Image.open(image_source).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to read local image '{image_source}': {e}")
            return None
            
    # Check if HTTP/HTTPS URL
    if image_source.startswith("http://") or image_source.startswith("https://"):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        try:
            resp = requests.get(image_source, headers=headers, timeout=timeout, stream=True)
            if resp.status_code == 200:
                img_data = io.BytesIO(resp.content)
                return Image.open(img_data).convert("RGB")
            else:
                logger.debug(f"HTTP {resp.status_code} fetching image: {image_source}")
                return None
        except Exception as e:
            logger.debug(f"Error fetching URL '{image_source}': {e}")
            return None
            
    return None



# =============================================================================================
# 3. VISUAL-SEMANTIC CONCEPT EXTRACTION & ALIGNMENT (IMPROVED & ROBUST)
# =============================================================================================

VISUAL_SEMANTIC_CATEGORIES = {
    "water_flood": ["water", "flood", "river", "rain", "ocean", "boat", "submerged", "disaster", "rescue", "storm", "inundate", "sea", "lake"],
    "protest_crowd": ["protest", "crowd", "people", "rally", "banner", "march", "police", "demonstration", "street", "rallies"],
    "health_medical": ["doctor", "vaccine", "hospital", "patient", "syringe", "medicine", "health", "nurse", "medical", "drug"],
    "space_astronomy": ["space", "rocket", "launch", "satellite", "star", "planet", "nasa", "moon", "orbit", "galaxy", "astronomy"],
    "sports_athletics": ["football", "cricket", "match", "game", "player", "trophy", "stadium", "tournament"],
    "document_text": ["document", "decree", "letter", "stamp", "signature", "notice", "paper", "official"],
    "politics_diplomacy": ["minister", "president", "summit", "conference", "parliament", "treaty", "speech", "tax", "government"],
    "animal_wildlife": ["cat", "dog", "bear", "bird", "lion", "tiger", "spider", "animal", "pet"]
}

def extract_visual_descriptors(image: Image.Image) -> Dict[str, Any]:
    """
    Extracts visual descriptors using adaptive color histogram & luminance profiles.
    """
    img_rgb = image.convert("RGB").resize((128, 128))
    arr = np.array(img_rgb, dtype=np.float32)
    
    r_mean = float(np.mean(arr[:, :, 0]))
    g_mean = float(np.mean(arr[:, :, 1]))
    b_mean = float(np.mean(arr[:, :, 2]))
    brightness = float((r_mean + g_mean + b_mean) / 3.0)
    contrast = float(np.std(arr))
    
    gray = image.convert("L").resize((128, 128))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_density = float(ImageStat.Stat(edges).mean[0])
    
    detected_concepts = []
    
    # Space & Night Astronomy
    if brightness < 55 and contrast > 30 and (b_mean > r_mean or r_mean > b_mean):
        detected_concepts.append("space_astronomy")
    # Water & Marine
    if b_mean >= g_mean - 10 and (b_mean > r_mean or g_mean > r_mean) and brightness > 40:
        detected_concepts.append("water_flood")
    # Nature & Landscape
    if g_mean > r_mean + 10 and g_mean > b_mean:
        detected_concepts.append("nature_landscape")
    # Crowd & Urban Complexity
    if edge_density > 45 and brightness > 70:
        detected_concepts.append("protest_crowd")
    # Medical & Documents
    if brightness > 180 and contrast < 40:
        detected_concepts.extend(["health_medical", "document_text"])
        
    if not detected_concepts:
        detected_concepts.append("general_photo")
        
    return {
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "edge_density": round(edge_density, 2),
        "detected_concepts": list(set(detected_concepts))
    }


def extract_headline_concepts(headline: str) -> List[str]:
    clean_hl = re.sub(r"[^\w\s]", " ", headline.lower())
    tokens = set(clean_hl.split())
    matched = [cat for cat, kw in VISUAL_SEMANTIC_CATEGORIES.items() if tokens.intersection(set(kw))]
    return matched or ["general_news"]


def compute_cross_modal_alignment(image: Image.Image, headline: str) -> Dict[str, Any]:
    visual_desc = extract_visual_descriptors(image)
    vis_concepts = visual_desc["detected_concepts"]
    text_concepts = extract_headline_concepts(headline)
    
    # Direct category overlap or broad compatibility
    common_concepts = set(vis_concepts).intersection(set(text_concepts))
    clean_words = set(re.sub(r"[^\w\s]", " ", headline.lower()).split())
    
    matched_keywords = []
    for t_c in text_concepts:
        if t_c in VISUAL_SEMANTIC_CATEGORIES:
            matched_keywords.extend(list(clean_words.intersection(set(VISUAL_SEMANTIC_CATEGORIES[t_c]))))
            
    # Calculate Base Score
    if common_concepts:
        base_score = 0.82 + (0.08 * min(len(matched_keywords), 2))
    elif "general_photo" in vis_concepts or "general_news" in text_concepts:
        base_score = 0.65 if matched_keywords else 0.55
    else:
        # Check explicit mismatch (e.g., Space image vs Protest headline)
        if "space_astronomy" in vis_concepts and "space_astronomy" not in text_concepts:
            base_score = 0.18
        else:
            base_score = 0.35 if not matched_keywords else 0.60
        
    _, ela_stats = perform_error_level_analysis(image)
    tamper_pen = 0.25 if ela_stats["tamper_risk"] == "High" else (0.10 if ela_stats["tamper_risk"] == "Moderate" else 0.0)
    
    final_score = max(0.05, min(0.98, base_score - tamper_pen))
    sim_pct = round(final_score * 100, 2)
    disc_pct = round((1.0 - final_score) * 100, 2)
    
    is_out_of_context = bool(sim_pct < 55.0 or ela_stats["tamper_risk"] == "High")
    verdict = (
        "AUTHENTIC & IN-CONTEXT" if not is_out_of_context
        else ("DIGITALLY TAMPERED / SPLICED IMAGE" if ela_stats["tamper_risk"] == "High"
        else "OUT-OF-CONTEXT / MISMATCHED (FAKE)")
    )
    
    return {
        "verdict": verdict,
        "is_out_of_context": is_out_of_context,
        "similarity_score": sim_pct,
        "discrepancy_index": disc_pct,
        "visual_concepts": vis_concepts,
        "headline_concepts": text_concepts,
        "matched_keywords": list(set(matched_keywords)),
        "tamper_stats": ela_stats,
        "diagnostic_reason": f"Visual concepts ({', '.join(vis_concepts)}) vs Headline concepts ({', '.join(text_concepts)})."
    }