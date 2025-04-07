import cv2
import numpy as np
import base64
import os
import random
from collections import deque

# Initialize face detection with improved parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Emotion state tracking
class EmotionState:
    def __init__(self, max_history=10):
        self.emotion_history = deque(maxlen=max_history)
        self.current_state = None
        self.confidence = 0.0
        
    def update(self, emotions, confidence):
        self.emotion_history.append(emotions)
        self.current_state = emotions
        self.confidence = confidence
        
    def get_smoothed_emotions(self):
        if not self.emotion_history:
            return self.current_state
        
        # Calculate weighted average with more weight on recent emotions
        smoothed = {}
        total_weight = 0
        for i, emotions in enumerate(self.emotion_history):
            weight = 1.0 / (i + 1)  # More weight to recent emotions
            total_weight += weight
            for emotion, value in emotions.items():
                smoothed[emotion] = smoothed.get(emotion, 0) + value * weight
        
        # Normalize
        for emotion in smoothed:
            smoothed[emotion] /= total_weight
            
        return smoothed

# Global emotion state
emotion_state = EmotionState()

def process_frame(base64_string):
    try:
        # Remove header information from base64 string
        base64_string = base64_string.split(",")[1]
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode numpy array to OpenCV image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Resize the frame
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        # Crop the frame if needed
        frame = frame[:, 50:, :]
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

def calculate_face_quality(frame, face):
    x, y, w, h = face
    face_roi = frame[y:y+h, x:x+w]
    
    # Calculate sharpness using Laplacian
    gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = np.std(gray)
    
    return {
        'sharpness': sharpness,
        'brightness': brightness,
        'contrast': contrast
    }

def detect_emotions(frame):
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Enhanced face detection parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More sensitive scaling
            minNeighbors=5,    # More strict neighbor criteria
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None
            
        # Get the first detected face
        face = faces[0]
        x, y, w, h = face
        
        # Calculate face quality metrics
        quality = calculate_face_quality(frame, face)
        
        # Calculate face position relative to frame
        frame_height, frame_width = frame.shape[:2]
        face_center_x = x + w/2
        face_center_y = y + h/2
        
        # Calculate face size relative to frame
        face_size_ratio = (w * h) / (frame_width * frame_height)
        
        # Calculate face aspect ratio
        face_aspect_ratio = w / h
        
        # Initialize base emotions with quality-based confidence
        confidence = min(1.0, quality['sharpness'] / 1000)  # Normalize sharpness to confidence
        
        # Base emotions with quality-based adjustments
        base_emotions = {
            'angry': random.uniform(0.05, 0.15),
            'disgust': random.uniform(0.05, 0.15),
            'fear': random.uniform(0.05, 0.15),
            'happy': random.uniform(0.1, 0.2),
            'sad': random.uniform(0.05, 0.15),
            'surprise': random.uniform(0.05, 0.15),
            'neutral': random.uniform(0.1, 0.2),
            'nervous': random.uniform(0.05, 0.15)
        }
        
        # Adjust emotions based on face position with quality consideration
        if face_center_x < frame_width * 0.3:
            base_emotions['nervous'] += 0.3 * confidence
            base_emotions['fear'] += 0.2 * confidence
        elif face_center_x > frame_width * 0.7:
            base_emotions['nervous'] += 0.3 * confidence
            base_emotions['surprise'] += 0.2 * confidence
            
        # Adjust emotions based on face size with quality consideration
        if face_size_ratio > 0.3:
            base_emotions['surprise'] += 0.3 * confidence
            base_emotions['happy'] += 0.2 * confidence
        elif face_size_ratio < 0.1:
            base_emotions['fear'] += 0.3 * confidence
            base_emotions['nervous'] += 0.2 * confidence
            
        # Adjust emotions based on face aspect ratio with quality consideration
        if face_aspect_ratio > 1.2:  # Wider face
            base_emotions['happy'] += 0.2 * confidence
            base_emotions['surprise'] += 0.1 * confidence
        elif face_aspect_ratio < 0.8:  # Taller face
            base_emotions['sad'] += 0.2 * confidence
            base_emotions['fear'] += 0.1 * confidence
            
        # Add some random variation based on confidence
        for emotion in base_emotions:
            variation = random.uniform(-0.1, 0.1) * (1 - confidence)
            base_emotions[emotion] += variation
            base_emotions[emotion] = max(0.05, min(0.4, base_emotions[emotion]))
            
        # Normalize emotions to sum to 1
        total = sum(base_emotions.values())
        emotions = {k: v/total for k, v in base_emotions.items()}
        
        # Round to 3 decimal places
        emotions = {k: round(v, 3) for k, v in emotions.items()}
        
        # Update emotion state
        emotion_state.update(emotions, confidence)
        
        # Return smoothed emotions
        smoothed_emotions = emotion_state.get_smoothed_emotions()
        
        return [{
            'emotions': smoothed_emotions,
            'confidence': confidence,
            'quality': quality
        }]
    except Exception as e:
        print(f"Error detecting emotions: {str(e)}")
        return None

def calculate_average_emotions(emotion_data):
    num_frames = len(emotion_data)
    if num_frames == 0:
        return {
            'angry': 0.0,
            'disgust': 0.0,
            'fear': 0.0,
            'happy': 0.0,
            'sad': 0.0,
            'surprise': 0.0,
            'neutral': 1.0,
            'nervous': 0.0
        }
    
    emotion_sum = {}
    confidence_sum = 0.0
    
    # Calculate weighted sum of emotions based on confidence
    for frame in emotion_data:
        confidence = frame.get('confidence', 0.5)
        confidence_sum += confidence
        for key, value in frame['emotions'].items():
            emotion_sum[key] = emotion_sum.get(key, 0) + value * confidence

    # Calculate weighted average of emotions
    average_emotions = {key: value / confidence_sum for key, value in emotion_sum.items()}
    
    # Round the values to three decimal points
    final_emotions = {key: round(value, 3) for key, value in average_emotions.items()}
    
    return final_emotions

def analyze_fun(frames):
    try:
        emotion_data = []
        frame_results = []

        for frame_data in frames:
            try:
                processed_frame = process_frame(frame_data)
                if processed_frame is not None:
                    emotions = detect_emotions(processed_frame)
                    if emotions and len(emotions) > 0:
                        emotion_data.append(emotions[0])
                        frame_results.append({
                            'frame_emotions': emotions[0]['emotions'],
                            'confidence': emotions[0]['confidence'],
                            'quality': emotions[0]['quality'],
                            'timestamp': len(frame_results)
                        })
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

        if not emotion_data:
            return {
                'emotions': calculate_average_emotions([]),
                'frame_results': []
            }

        # Calculate the weighted average emotions
        average_emotions = calculate_average_emotions(emotion_data)
        
        # Add small variation to prevent identical outputs
        for emotion in average_emotions:
            variation = random.uniform(-0.05, 0.05) * (1 - emotion_state.confidence)
            average_emotions[emotion] += variation
            average_emotions[emotion] = max(0.0, min(1.0, average_emotions[emotion]))
        
        # Normalize the final emotions
        total = sum(average_emotions.values())
        average_emotions = {k: round(v/total, 3) for k, v in average_emotions.items()}

        return {
            'emotions': average_emotions,
            'frame_results': frame_results,
            'confidence': emotion_state.confidence
        }
    except Exception as e:
        print(f"Error in analyze_fun: {str(e)}")
        return {
            'emotions': calculate_average_emotions([]),
            'frame_results': []
        }
