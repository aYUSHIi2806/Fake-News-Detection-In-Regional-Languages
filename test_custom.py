import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from train_text_model import predict_regional_news

if __name__ == "__main__":
    print("=" * 80)
    print("🔍 TESTING CALIBRATED MODULE 1 MODEL WITH CUSTOM SAMPLES")
    print("=" * 80)
    
    # Yahan hum fake aur real dono tarah ke sample test karenge
    test_articles = [
    # Full News Style Fake Article
    """सोशल मीडिया पर एक पत्र तेजी से वायरल हो रहा है जिसमें दावा किया जा रहा है कि केंद्र सरकार 'प्रधानमंत्री जनधन योजना' के तहत सभी नागरिकों के बैंक खातों में 15 लाख रुपये जमा कर रही है। पत्र में एक ऑनलाइन फॉर्म का लिंक दिया गया है और दावा किया गया है कि फॉर्म भरने वाले पहले 10,000 आवेदकों को तुरंत राशि स्थानांतरित की जाएगी। आधिकारिक तौर पर सरकार या किसी बैंक ने ऐसी किसी योजना की घोषणा नहीं की है।""",
    
    # Real News Article
    """भारतीय रिजर्व बैंक (RBI) ने अपनी नवीनतम मौद्रिक नीति समिति की समीक्षा में रेपो दर को 6.5 प्रतिशत पर अपरिवर्तित रखने का निर्णय लिया है। आरबीआई गवर्नर ने प्रेस कॉन्फ्रेंस में जानकारी दी कि महंगाई दर नियंत्रण में है और देश की आर्थिक विकास दर मजबूत बनी हुई है।"""
]
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n--- Test Case {i} ---")
        result = predict_regional_news(article)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Text Preview   : {result['text_preview']}")
            print(f"Detected Lang  : {result['detected_language']}")
            print(f"Verdict        : {result['verdict']}")
            print(f"Confidence     : Real: {result['confidence_real']}% | Fake: {result['confidence_fake']}%")
            print(f"Threshold Used : {result['optimal_threshold_used']}")
            print(f"Suspicious Key : {result['suspicious_tokens']}")
            
    print("\n" + "=" * 80)