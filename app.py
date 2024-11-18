import os
import logging
from flask import Flask, request, render_template, jsonify
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.api_core.exceptions import ServiceUnavailable
import base64

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

def init_vertexai():
    try:
        vertexai.init(project="convai-442105", location="us-central1")
        return GenerativeModel("gemini-1.5-flash-002")
    except Exception as e:
        app.logger.error(f"Failed to initialize Vertex AI: {str(e)}")
        return None

def process_audio(audio_data, model):
    try:
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        audio_part = Part.from_data(audio_bytes, mime_type="audio/webm")
        
        prompt = "Transcribe the audio and perform sentiment analysis on the transcription. Provide the transcript and a brief sentiment analysis."
        response = model.generate_content([audio_part, prompt])
        return response.text
    except ServiceUnavailable as e:
        app.logger.error(f"Vertex AI service unavailable: {str(e)}")
        raise
    except ValueError as e:
        app.logger.error(f"Invalid audio data: {str(e)}")
        raise
    except Exception as e:
        app.logger.error(f"Unexpected error in process_audio: {str(e)}")
        raise

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            audio_data = request.json.get('audio')
            if not audio_data:
                return jsonify({"error": "No audio data received"}), 400
            
            model = init_vertexai()
            if not model:
                return jsonify({"error": "Failed to initialize Vertex AI"}), 500
            
            result = process_audio(audio_data, model)
            
            with open('result.txt', 'w') as f:
                f.write(result)
            
            return jsonify({"result": result})
        except ServiceUnavailable:
            return jsonify({"error": "Vertex AI service is currently unavailable"}), 503
        except ValueError:
            return jsonify({"error": "Invalid audio data provided"}), 400
        except Exception as e:
            app.logger.error(f"Unhandled exception in index route: {str(e)}")
            return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
    
    return render_template('index.html')

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not Found", "details": str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal Server Error", "details": str(error)}), 500

if __name__ == '__main__':
    app.run(debug=True)
