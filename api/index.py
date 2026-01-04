# api/index.py
from flask import Flask, request, jsonify
from mangum import Mangum
import sys
import os
import traceback

# ============ CRITICAL: Set NLTK data path ============
# Vercel serverless has read-only filesystem except /tmp
import nltk
nltk.data.path.append('/tmp/nltk_data')

# Create /tmp/nltk_data directory if it doesn't exist
os.makedirs('/tmp/nltk_data', exist_ok=True)

# Download NLTK data to /tmp (writable location)
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("Downloading NLTK data to /tmp/nltk_data...")
    nltk.download('wordnet', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('omw-1.4', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('punkt', download_dir='/tmp/nltk_data', quiet=True)
    nltk.download('stopwords', download_dir='/tmp/nltk_data', quiet=True)
    # Reload the data path
    nltk.data.path.append('/tmp/nltk_data')

# Add parent directory to path for your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# ============ SIMPLE TEST ENDPOINTS ============

@app.route('/api/test', methods=['GET'])
def test():
    """Simple test endpoint"""
    return jsonify({
        "status": "success",
        "message": "Flask is working!",
        "nltk_data_path": nltk.data.path,
        "wordnet_loaded": nltk.data.find("corpora/wordnet") is not None
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check with NLTK verification"""
    try:
        from nltk.corpus import wordnet as wn
        wordnet_status = "loaded"
        sample_word = "test"
        synsets = wn.synsets(sample_word)
    except Exception as e:
        wordnet_status = f"error: {str(e)}"
        synsets = []
    
    return jsonify({
        "status": "healthy",
        "service": "Knowledge Grid Generator",
        "nltk_data": wordnet_status,
        "available_paths": nltk.data.path,
        "synsets_for_test": len(synsets)
    })

# ============ LAZY IMPORT OF YOUR MAIN MODULE ============
# Import only when needed to avoid startup errors

def get_generate_function():
    """Lazy import to avoid startup issues"""
    try:
        from main import generate_knowledge_grid
        return generate_knowledge_grid
    except Exception as e:
        print(f"Error importing main module: {e}")
        raise

# ============ MAIN ENDPOINTS ============

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate knowledge grid"""
    try:
        # Lazy import your function
        generate_knowledge_grid = get_generate_function()
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        topic = data.get('topic', '')
        texts = data.get('texts', [])
        frequency = data.get('frequency', 'high')
        min_len = int(data.get('min_len', 3))
        max_len = int(data.get('max_len', 7))
        word_num = int(data.get('word_num', 20))
        
        # Validate
        if not topic:
            return jsonify({"error": "Topic is required"}), 400
        
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Texts must be a non-empty list"}), 400
        
        # Filter empty texts
        texts = [t for t in texts if t and str(t).strip()]
        if not texts:
            return jsonify({"error": "At least one non-empty text is required"}), 400
        
        # Generate
        words_with_labels = generate_knowledge_grid(
            topic=topic,
            texts=texts,
            total_words=word_num,
            frequency=frequency,
            min_len=min_len,
            max_len=max_len
        )
        
        import pandas as pd
        df = pd.DataFrame(words_with_labels, columns=["Word", "Type"])
        df["Related to Topic"] = " "
        df["Not Related to Topic"] = " "
        df["I don't know"] = " "
        
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        
        return jsonify({
            "success": True,
            "topic": topic,
            "grid": words_with_labels,
            "csv_data": csv_data,
            "dataframe": df.to_dict(orient='records')
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "details": traceback.format_exc() if app.debug else "Contact administrator"
    }), 500

# ============ VERCEL REQUIREMENT ============
handler = Mangum(app)
