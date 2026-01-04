# api/index.py
from flask import Flask, request, jsonify
from mangum import Mangum
import sys
import os

# Add parent directory to path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import NLTK and download data
import nltk

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Now import your main module
try:
    from main import generate_knowledge_grid
    import pandas as pd
    import random
    import io
except ImportError as e:
    print(f"Import error: {e}")
    # Create a dummy function for testing
    def generate_knowledge_grid(*args, **kwargs):
        return [("test", "From Texts"), ("example", "Related")]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "default-secret-key-change-in-production")

@app.route('/api', methods=['GET', 'POST'])
def api_root():
    """Main API endpoint"""
    if request.method == 'GET':
        return jsonify({
            "message": "Knowledge Grid Generator API",
            "endpoints": {
                "POST /api/generate": "Generate knowledge grid",
                "POST /api/download": "Download CSV"
            }
        })
    
    # Handle POST requests
    data = request.get_json() or {}
    
    # Check if it's a generate request
    if 'topic' in data and 'texts' in data:
        return generate_grid()
    
    return jsonify({"error": "Invalid request"}), 400

@app.route('/api/generate', methods=['POST'])
def generate_grid():
    """Generate knowledge grid"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        topic = data.get('topic', '')
        texts = data.get('texts', [])
        frequency = data.get('frequency', 'high')
        min_len = int(data.get('min_len', 3))
        max_len = int(data.get('max_len', 7))
        word_num = int(data.get('word_num', 20))
        
        # Validate inputs
        if not topic or not isinstance(topic, str):
            return jsonify({"error": "Valid topic is required"}), 400
        
        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Texts must be a list"}), 400
        
        # Filter out empty texts
        texts = [text for text in texts if text and isinstance(text, str) and text.strip()]
        
        if not texts:
            return jsonify({"error": "At least one non-empty text is required"}), 400
        
        # Generate knowledge grid
        words_with_labels = generate_knowledge_grid(
            topic, texts,
            total_words=word_num,
            frequency=frequency,
            min_len=min_len,
            max_len=max_len
        )
        
        # Create DataFrame
        df = pd.DataFrame(words_with_labels, columns=["Word", "Type"])
        df["Related to Topic"] = " "
        df["Not Related to Topic"] = " "
        df["I don't know"] = " "
        
        # Convert to CSV
        csv_data = df.to_csv(index=False, encoding='utf-8-sig')
        
        return jsonify({
            "success": True,
            "topic": topic,
            "grid": words_with_labels,
            "csv_data": csv_data,
            "dataframe": df.to_dict(orient='records')
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in generate_grid: {e}\n{error_details}")
        return jsonify({
            "success": False,
            "error": str(e),
            "details": error_details
        }), 500

@app.route('/api/download', methods=['POST'])
def download_csv():
    """Generate download link for CSV"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        csv_data = data.get('csv_data', '')
        topic = data.get('topic', 'knowledge_grid')
        
        if not csv_data:
            return jsonify({"error": "No CSV data to download"}), 400
        
        import base64
        # Create a data URL for download
        csv_b64 = base64.b64encode(csv_data.encode('utf-8-sig')).decode()
        
        return jsonify({
            "success": True,
            "filename": f"{topic}_knowledge_grid.csv",
            "download_url": f"data:text/csv;base64,{csv_b64}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Knowledge Grid Generator",
        "nltk_data": "loaded" if nltk.data.find("corpora/wordnet") else "not loaded"
    })

# ============ IMPORTANT ============
# Vercel requires this handler variable
# DO NOT use app.run() or if __name__ == '__main__':
handler = Mangum(app)
