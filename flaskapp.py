from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (
    Input, Conv3D, LSTM, Dense, Dropout, Bidirectional, 
    MaxPool3D, Activation, Reshape, BatchNormalization, 
    TimeDistributed, Flatten
)
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import time
import subprocess
import tempfile
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import Levenshtein  # You'll need to install this: pip install python-Levenshtein

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants
TARGET_FRAMES = 75
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def build_model():
    """Build the model architecture"""
    input_layer = Input(shape=(TARGET_FRAMES, 46, 140, 1))
    
    # CNN layers
    x = Conv3D(128, 3, padding='same', name='conv3d_1')(input_layer)
    x = Activation('relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    x = Conv3D(256, 3, padding='same', name='conv3d_2')(x)
    x = Activation('relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    x = Conv3D(75, 3, padding='same', name='conv3d_3')(x)
    x = Activation('relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    # Reshape before flattening
    x = Reshape((TARGET_FRAMES, -1))(x)
    
    # Process each time step
    x = TimeDistributed(Flatten())(x)
    
    # LSTM layers
    x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    x = Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True))(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output = Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=output)
    return model

# Define the CTCLoss function
def CTCLoss(y_true, y_pred):
    """Custom CTC Loss function"""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# Load the model with custom loss function
try:
    model = build_model()
    model.load_weights('models/model_epoch_30.h5')
    model.compile(optimizer='adam', loss=CTCLoss)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def convert_to_mp4(input_path, output_path):
    """Convert video to MP4 format using ffmpeg"""
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-vcodec', 'libx264', output_path, '-y'], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {str(e)}")
        return None

def load_video(video_path):
    """Load and preprocess video frames to have exactly TARGET_FRAMES frames"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return False, None

        frames = []
        original_frames = []  # Store original frames for visualization
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Error: Invalid frame count {total_frames}")
            cap.release()
            return False, None

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            try:
                # Store original frame for visualization
                original_frames.append(frame.copy())
                
                # Convert to grayscale using tensorflow operation
                gray = tf.image.rgb_to_grayscale(frame)
                
                # Extract lip region
                lip_region = gray[190:236, 80:220, :]
                frames.append(lip_region)
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

        cap.release()
        
        if not frames:
            print("Error: No frames were processed successfully")
            return False, None
        
        # Convert frames to tensor
        frames_tensor = tf.stack(frames)
        
        # Normalize using mean and std
        mean = tf.math.reduce_mean(frames_tensor)
        std = tf.math.reduce_std(tf.cast(frames_tensor, tf.float32))
        normalized_frames = tf.cast((frames_tensor - mean), tf.float32) / std
        
        # Ensure we have exactly TARGET_FRAMES frames
        if len(frames) < TARGET_FRAMES:
            last_frame = normalized_frames[-1]
            last_orig = original_frames[-1]
            padding = [last_frame] * (TARGET_FRAMES - len(frames))
            orig_padding = [last_orig] * (TARGET_FRAMES - len(original_frames))
            normalized_frames = tf.concat([normalized_frames, tf.stack(padding)], axis=0)
            original_frames.extend(orig_padding)
        elif len(frames) > TARGET_FRAMES:
            normalized_frames = normalized_frames[:TARGET_FRAMES]
            original_frames = original_frames[:TARGET_FRAMES]
        
        return normalized_frames, original_frames

    except Exception as e:
        print(f"Error in load_video: {str(e)}")
        if 'cap' in locals() and cap is not None:
            cap.release()
        return False, None

def preprocess_frames(frames):
    """Preprocess video frames for the model."""
    processed_frames = []
    for frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to model input size
        resized = cv2.resize(gray, (100, 50))  # Adjust size as needed
        # Normalize
        normalized = resized / 255.0
        processed_frames.append(normalized)
    
    return np.array(processed_frames)

def load_alignments(path):
    """Load and process text alignments"""
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if len(line) >= 3 and line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def predict_word(video_path):
    """Predict word from video"""
    if model is None:
        return "Error: Model not loaded properly"
    
    # Load and preprocess video
    frames, original_frames = load_video(video_path)
    if frames is False:
        return "Error: Could not load video"
    
    # Add batch dimension for prediction
    frames_batch = tf.expand_dims(frames, axis=0)
    
    # Make prediction
    yhat = model.predict(frames_batch)
    
    # Decode prediction using beam search (non-greedy)
    input_length = tf.ones(shape=(1,), dtype="int64") * tf.cast(tf.shape(yhat)[1], dtype="int64")
    decoded = tf.keras.backend.ctc_decode(yhat, input_length, greedy=False)[0][0].numpy()
    
    # Convert to text
    predicted_text = tf.strings.reduce_join(num_to_char(decoded)).numpy().decode('utf-8')
    
    return predicted_text

def get_available_videos():
    """Get list of available videos from the dataset"""
    video_dir = 'data/s1'
    alignment_dir = 'data/alignments/s1'
    if not os.path.exists(video_dir):
        return []
    
    videos = []
    for filename in os.listdir(video_dir):
        if filename.endswith('.mpg'):
            # Get the word from filename (assuming format: word_*.mpg)
            word = filename.split('_')[0]
            
            # Get actual text from alignment file
            alignment_path = os.path.join(alignment_dir, filename.replace('.mpg', '.align'))
            actual_text = ""
            if os.path.exists(alignment_path):
                alignments = load_alignments(alignment_path)
                actual_text = tf.strings.reduce_join(num_to_char(alignments)).numpy().decode('utf-8')
            
            videos.append({
                'filename': filename,
                'word': word,
                'path': os.path.join(video_dir, filename),
                'actual_text': actual_text
            })
    return videos

@app.route('/')
def index():
    videos = get_available_videos()
    return render_template('index.html', videos=videos)

@app.route('/get_video/<path:filename>')
def get_video(filename):
    """Serve video file"""
    video_path = os.path.join('data/s1', filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'}), 404
    
    # Convert to MP4 if needed
    mp4_path = video_path.replace('.mpg', '.mp4')
    if not os.path.exists(mp4_path):
        mp4_path = convert_to_mp4(video_path, mp4_path)
        if not mp4_path:
            return jsonify({'error': 'Error converting video'}), 500
    
    return send_file(mp4_path, mimetype='video/mp4')

def create_visualization(frame, frame_idx, heatmap=None):
    """Create visualization of frame with lip region and heatmap"""
    try:
        plt.switch_backend('Agg')  # Ensure we're using Agg backend
        fig = plt.figure(figsize=(15, 5))
        
        # Original frame with lip region highlighted
        plt.subplot(1, 3, 1)
        plt.imshow(frame)
        rect = Rectangle((80, 190), 140, 46, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.title(f'Frame {frame_idx} with Lip Region')
        plt.axis('off')
        
        # Cropped lip region
        plt.subplot(1, 3, 2)
        lip_region = frame[190:236, 80:220]
        plt.imshow(lip_region)  # Remove cmap='gray' since we want color
        plt.title('Cropped Lip Region')
        plt.axis('off')
        
        # Heatmap if provided
        if heatmap is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Activation Heatmap')
            plt.axis('off')
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # Explicitly close the figure
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"Error in create_visualization: {str(e)}")
        return None

def create_heatmap(frames, model, frame_idx):
    """Create a heatmap visualization for a single frame"""
    try:
        # Get the model's intermediate layer output
        intermediate_model = Model(
            inputs=model.input,
            outputs=model.get_layer('conv3d_2').output
        )
        
        # Get the activation map
        activation = intermediate_model.predict(tf.expand_dims(frames, axis=0))
        
        # Average across channels
        heatmap = np.mean(activation[0, frame_idx], axis=-1)
        
        # Normalize heatmap
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        
        # Resize heatmap to match lip region size
        heatmap = cv2.resize(heatmap, (140, 46))
        
        return heatmap
    except Exception as e:
        print(f"Error in create_heatmap: {str(e)}")
        return None

def analyze_prediction(video_path, predicted_text):
    """Analyze the prediction and return statistics"""
    frames, original_frames = load_video(video_path)
    if frames is False:
        return None
    
    # Get frame statistics
    mean = tf.reduce_mean(frames)
    variance = tf.reduce_mean(tf.square(frames - mean))
    std = tf.sqrt(variance)
    
    frame_stats = {
        'mean_intensity': float(mean),
        'std_intensity': float(std),
        'max_intensity': float(tf.reduce_max(frames)),
        'min_intensity': float(tf.reduce_min(frames))
    }
    
    # Get prediction confidence
    frames_batch = tf.expand_dims(frames, axis=0)
    yhat = model.predict(frames_batch)
    
    # Get top 3 predictions
    input_length = tf.ones(shape=(1,), dtype="int64") * tf.cast(tf.shape(yhat)[1], dtype="int64")
    decoded = tf.keras.backend.ctc_decode(yhat, input_length, greedy=False)[0]
    
    top_predictions = []
    for i in range(min(3, len(decoded))):
        pred_text = tf.strings.reduce_join(num_to_char(decoded[i])).numpy().decode('utf-8')
        top_predictions.append(pred_text)
    
    return {
        'frame_stats': frame_stats,
        'top_predictions': top_predictions,
        'prediction_length': len(predicted_text),
        'confidence': float(tf.reduce_max(yhat))
    }

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Endpoint for video analysis"""
    data = request.get_json()
    video_path = data.get('video_path')
    predicted_text = data.get('predicted_text')
    
    if not video_path or not predicted_text:
        return jsonify({'error': 'Missing video path or prediction'}), 400
    
    analysis = analyze_prediction(video_path, predicted_text)
    if analysis is None:
        return jsonify({'error': 'Could not analyze video'}), 500
    
    return jsonify(analysis)

@app.route('/visualize', methods=['POST'])
def visualize_frame():
    """Endpoint for frame visualization"""
    try:
        data = request.get_json()
        video_path = os.path.join('data/s1', data['video_path'].split('/')[-1])
        frame_idx = int(data['frame_idx'])
        
        if not os.path.exists(video_path):
            return jsonify({'error': f'Video file not found: {video_path}'}), 404
        
        # Load video frames
        frames, original_frames = load_video(video_path)
        if frames is False or original_frames is None:
            return jsonify({'error': 'Could not load video frames'}), 500
        
        if frame_idx >= len(original_frames):
            return jsonify({'error': f'Frame index {frame_idx} out of range'}), 400
        
        # Get the selected frame
        frame = original_frames[frame_idx]
        
        # Create heatmap
        heatmap = create_heatmap(frames, model, frame_idx)
        
        # Create visualization with all features
        plt.switch_backend('Agg')  # Ensure we're using Agg backend
        fig = plt.figure(figsize=(15, 5))
        
        # Original frame with lip region highlighted
        plt.subplot(1, 3, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        rect = Rectangle((80, 190), 140, 46, linewidth=2, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.title(f'Frame {frame_idx + 1} with Lip Region')
        plt.axis('off')
        
        # Cropped and processed lip region
        plt.subplot(1, 3, 2)
        lip_region = frame_rgb[190:236, 80:220]
        plt.imshow(lip_region)
        plt.title('Processed Lip Region')
        plt.axis('off')
        
        # Activation heatmap
        plt.subplot(1, 3, 3)
        if heatmap is not None:
            plt.imshow(heatmap, cmap='hot')
            plt.colorbar(label='Activation Intensity')
            plt.title('Lip Movement Heatmap')
        else:
            plt.text(0.5, 0.5, 'Heatmap not available', 
                    horizontalalignment='center',
                    verticalalignment='center')
            plt.title('Activation Heatmap')
        plt.axis('off')
        
        # Add frame statistics
        frame_stats, magnitude = analyze_frame(frames[frame_idx])
        plt.figtext(0.02, 0.02, 
                   f'Mean Intensity: {frame_stats["mean_intensity"]:.2f}\n'
                   f'Std Dev: {frame_stats["std_dev"]:.2f}\n'
                   f'Max Movement: {frame_stats["max_movement"]:.2f}',
                   fontsize=8, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1, dpi=150)
        plt.close(fig)  # Explicitly close the figure
        buf.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({
            'visualization': image_base64,
            'frame_stats': frame_stats
        })
        
    except Exception as e:
        print(f"Error in visualize_frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

def analyze_frame(frame):
    """Analyze a single frame for lip movement and statistics"""
    try:
        # Convert tensor to numpy array if needed
        if isinstance(frame, tf.Tensor):
            frame = frame.numpy()
        
        # Ensure frame is in the correct format for OpenCV
        frame = frame.astype(np.float32)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame.squeeze(-1)  # Remove single channel dimension
            
        # Calculate gradients using Sobel
        grad_x = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate magnitude of gradients
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Calculate statistics
        stats = {
            'mean_intensity': float(np.mean(frame)),
            'std_dev': float(np.std(frame)),
            'max_movement': float(np.max(magnitude)),
            'movement_intensity': float(np.mean(magnitude))
        }
        
        return stats, magnitude
        
    except Exception as e:
        print(f"Error in analyze_frame: {str(e)}")
        return None, None

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        data = request.get_json()
        video_path = os.path.join('data/s1', data['filename'])
        actual_text = data.get('actual_text', '')

        # Start timing the processing
        start_time = time.time()

        # Make prediction
        prediction = predict_word(video_path)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds

        # Get video information
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        # Calculate text similarity
        similarity_score = calculate_similarity(prediction, actual_text or data['word'])

        # Get model confidence from analysis
        frames, _ = load_video(video_path)
        frames_batch = np.expand_dims(frames, axis=0)
        yhat = model.predict(frames_batch)
        confidence = float(np.max(yhat))  # Get the highest probability as confidence

        # Prepare analysis data
        analysis = {
            'confidence': confidence,
            'processing_time': processing_time,
            'frame_count': frame_count,
            'frame_rate': frame_rate,
            'prediction': prediction,
            'frame_stats': {
                'mean_intensity': float(np.mean(frames)),
                'std_intensity': float(np.std(frames))
            }
        }

        return jsonify({
            'prediction': prediction,
            'actual_text': actual_text or data['word'],
            'similarity_score': similarity_score,
            'analysis': analysis
        })

    except Exception as e:
        print(f"Error in upload_file: {str(e)}")  # Add debug print
        return jsonify({'error': str(e)}), 500

def get_video_fps(video_path):
    """Get the frame rate of the video."""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps
    except:
        return 25  # Default fallback

def calculate_similarity(pred_text, actual_text):
    """Calculate similarity score between predicted and actual text."""
    if not pred_text or not actual_text:
        return 0.0
    
    # Convert to lowercase and remove extra spaces
    pred_text = pred_text.lower().strip()
    actual_text = actual_text.lower().strip()
    
    # Calculate Levenshtein distance
    distance = Levenshtein.distance(pred_text, actual_text)
    max_length = max(len(pred_text), len(actual_text))
    
    # Convert distance to similarity score (0 to 1)
    similarity = 1 - (distance / max_length)
    return similarity

if __name__ == '__main__':
    app.run(debug=True) 
