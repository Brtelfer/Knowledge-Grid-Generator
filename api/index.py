# api/index.py - COMPLETELY NEW VERSION
from flask import Flask, request, jsonify
from mangum import Mangum
import sys
import os
import traceback

# ============ SKIP NLTK CHECK AT IMPORT TIME ============
# We'll handle NLTK downloads inside endpoints, not at module level

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# Store NLTK initialization status
_NLTK_INITIALIZED = False

def initialize_nltk():
    """Initialize NLTK data on first request"""
    global _NLTK_INITIALIZED
    
    if _NLTK_INITIALIZED:
        return True
    
    try:
        import nltk
        
        # Set path to /tmp for writable storage
        nltk_data_dir = '/tmp/nltk_data'
        nltk.data.path.append(nltk_data_dir)
        
        # Create directory
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Download required data
        print(f"Downloading NLTK data to {nltk_data_dir}...")
        
        # Download if not already present
        try:
            nltk.data.find('corpora/wordnet', paths=[nltk_data_dir])
        except LookupError:
            nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
            nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt', paths=[nltk_data_dir])
        except LookupError:
            nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords', paths=[nltk_data_dir])
        except LookupError:
            nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        
        _NLTK_INITIALIZED = True
        print("NLTK initialization complete")
        return True
        
    except Exception as e:
        print(f"NLTK initialization failed: {e}")
        traceback.print_exc()
        return False

# ============ SIMPLE TEST ENDPOINTS ============

@app.route('/api/test', methods=['GET'])
def test():
    """Simple test endpoint - no NLTK required"""
    return jsonify({
        "status": "success",
        "message": "Flask API is running",
        "nltk_initialized": _NLTK_INITIALIZED
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check with optional NLTK test"""
    nltk_status = "not_initialized"
    if _NLTK_INITIALIZED:
        try:
            import nltk
            from nltk.corpus import wordnet
            nltk_status = "working"
        except Exception as e:
            nltk_status = f"error: {str(e)}"
    
    return jsonify({
        "status": "healthy",
        "service": "Knowledge Grid Generator",
        "nltk": nltk_status,
        "endpoints": {
            "GET /api/test": "Simple test",
            "GET /api/health": "Health check",
            "POST /api/init-nltk": "Initialize NLTK",
            "POST /api/generate": "Generate grid"
        }
    })

@app.route('/api/init-nltk', methods=['GET', 'POST'])
def init_nltk():
    """Manually initialize NLTK data"""
    try:
        success = initialize_nltk()
        if success:
            return jsonify({
                "success": True,
                "message": "NLTK data initialized in /tmp/nltk_data"
            })
        else:
            return jsonify({
                "success": False,
                "message": "NLTK initialization failed"
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

# ============ MAIN ENDPOINTS ============

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate knowledge grid"""
    try:
        # Initialize NLTK first
        if not initialize_nltk():
            return jsonify({
                "success": False,
                "error": "Failed to initialize NLTK data"
            }), 500
        
        # Now import your module (NLTK should be ready)
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        try:
            from main import generate_knowledge_grid
            import pandas as pd
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to import main module: {str(e)}"
            }), 500
        
        # Get request data
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
        texts = [str(t).strip() for t in texts if t and str(t).strip()]
        if not texts:
            return jsonify({"error": "At least one non-empty text is required"}), 400
        
        # Generate (with smaller pool for performance)
        print(f"Generating grid for topic: {topic}")
        words_with_labels = generate_knowledge_grid(
            topic=topic,
            texts=texts,
            total_words=min(word_num, 30),  # Limit for testing
            frequency=frequency,
            min_len=min_len,
            max_len=max_len
        )
        
        # Create CSV
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
            "count": len(words_with_labels)
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
        "details": traceback.format_exc()
    }), 500

# ============ VERCEL REQUIREMENT ============
handler = Mangum(app)

# ============ DO NOT RUN FLASK DIRECTLY ============
# No app.run() here!
