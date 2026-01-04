from http.server import BaseHTTPRequestHandler
import json
import os
import sys

# Add the parent directory to path to import your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import generate_knowledge_grid, extract_vocabulary_from_texts
import pandas as pd
import random
import io

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # Extract data from request
            topic = data.get('topic', '')
            texts = data.get('texts', [])
            frequency = data.get('frequency', 'high')
            min_len = int(data.get('min_len', 3))
            max_len = int(data.get('max_len', 7))
            word_num = int(data.get('word_num', 20))
            
            if not topic or not texts:
                raise ValueError("Topic and at least one text are required")
            
            words_with_labels = generate_knowledge_grid(
                topic, texts,
                total_words=word_num,
                frequency=frequency,
                min_len=min_len,
                max_len=max_len
            )
            
            df = pd.DataFrame(words_with_labels, columns=["Word", "Type"])
            df["Related to Topic"] = " "
            df["Not Related to Topic"] = " "
            df["I don't know"] = " "
            
            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
            
            response = {
                "success": True,
                "grid": words_with_labels,
                "csv_data": csv_data,
                "topic": topic
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            error_response = {
                "success": False,
                "error": str(e)
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))
