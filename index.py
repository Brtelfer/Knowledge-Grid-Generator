from flask import Flask, request, jsonify
from mangum import Mangum

app = Flask(__name__)

# Your existing Flask routes
@app.route('/api', methods=['GET'])
def hello():
    return jsonify({"message": "Hello from Flask on Vercel!"})

@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json()
    # Your processing logic here
    return jsonify({"result": "processed"})

# This is crucial: Mangum adapter for Vercel
handler = Mangum(app, api_gateway_base_path="/api")
