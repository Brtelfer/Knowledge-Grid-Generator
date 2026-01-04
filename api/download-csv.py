from http.server import BaseHTTPRequestHandler
import json
import base64
import io

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            csv_data = data.get('csv_data', '')
            topic = data.get('topic', 'knowledge_grid')
            
            if not csv_data:
                raise ValueError("No CSV data provided")
            
            # Create download response
            response = {
                "success": True,
                "filename": f"{topic}_knowledge_grid.csv",
                "csv_data": csv_data,
                "download_url": f"data:text/csv;base64,{base64.b64encode(csv_data.encode()).decode()}"
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
