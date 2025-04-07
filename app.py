from flask import Flask, jsonify, request, g, send_from_directory
import google.generativeai as genai
from functions.question_generation import generate_questions
from functions.emotion_analysis import analyze_fun
from functions.review_generation import generate_review as generate_interview_review
from flask_cors import CORS
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor
import queue

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})  # Enables CORS for all routes

executor = ThreadPoolExecutor(max_workers=3)
emotion_queue = queue.Queue()

# Load API key for Google Gemini AI
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY3')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY3 is not set in environment variables!")

genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

@app.before_request
def before_request():
    """ Attach the AI model to the global request context """
    g.model = model

@app.route("/")
def home():
    return send_from_directory('.', 'homepage.html')

@app.route("/interview")
def interview():
    return send_from_directory('.', 'interviewwindow.html')

@app.route("/result")
def result():
    return send_from_directory('.', 'result.html')

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"status": 404, "message": "Not Found"}), 404

# ✅ Generate Interview Questions
@app.route('/api/get-questions', methods=['POST'])
def get_question():
    try:
        data = request.get_json()
        print("📩 Received data for question generation:", data)

        company = data.get('company')
        job_role = data.get('job_role')
        experience_lvl = data.get('experience_lvl')

        if not company or not job_role or not experience_lvl:
            return jsonify({'error': 'Missing required fields'}), 400

        questions = generate_questions(job_role, experience_lvl, company)
        print("✅ Generated Questions:", questions)

        if not isinstance(questions, list) or not questions:
            return jsonify({'error': 'Failed to generate questions'}), 400

        return jsonify({'job_role': job_role, 'exp_level': experience_lvl, 'qtns': questions}), 200
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        return jsonify({'error': "Something went wrong"}), 500


# ✅ Generate Interview Review
@app.route('/api/generate-review', methods=['POST'])
def generate_review_route():
    try:
        data = request.get_json()
        job_role = data.get("job_role")
        company = data.get("company")
        questions = data.get("questions", [])
        answers = data.get("answers", [])
        emotion = data.get("emotion", "neutral")
        suspicious_count = data.get("suspiciousCount", 0)

        if not job_role or not questions or not answers:
            return jsonify({"error": "Missing required data"}), 400

        review_text = generate_interview_review(job_role,company, questions, answers, emotion, suspicious_count)
        return jsonify({"review": review_text}), 200
    except Exception as e:
        print(f"❌ Error generating review: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/api/analyze-emotions', methods=['POST'])
def analyze_emotions():
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        
        if not frames:
            return jsonify({'error': 'No frames provided for analysis'}), 400
            
        print(f"📊 Analyzing emotions from {len(frames)} frames")
        
        # Process emotions using the analyze_fun from emotion_analysis.py
        emotions = analyze_fun(frames)
        
        if not emotions or 'emotions' not in emotions:
            return jsonify({'error': 'Failed to analyze emotions'}), 500
            
        print("✅ Emotion Analysis Result:", emotions)
        return jsonify({'emotions': emotions}), 200
    except Exception as e:
        print(f"❌ Error analyzing emotions: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
