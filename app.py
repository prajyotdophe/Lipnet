import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from typing import List, Tuple
import tempfile
import h5py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, LSTM, Dense, Dropout, Bidirectional, 
    MaxPool3D, Activation, Reshape, BatchNormalization, 
    TimeDistributed, Flatten
)
import base64

# Set Streamlit layout
st.set_page_config(layout='wide')

# Constants
TARGET_FRAMES = 75
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def convert_to_mp4(input_path: str, output_path: str) -> str:
    """Convert video to MP4 format using ffmpeg"""
    os.system(f'ffmpeg -i "{input_path}" -vcodec libx264 "{output_path}" -y')
    return output_path

def get_video_html(video_path):
    """Convert video to HTML for playing in Streamlit"""
    with open(video_path, 'rb') as f:
        data_url = base64.b64encode(f.read()).decode()
    return f'''
        <video width="100%" controls>
            <source src="data:video/mp4;base64,{data_url}" type="video/mp4">
        </video>
    '''

def load_video(path: str) -> tf.Tensor:
    """Load and preprocess video frames to have exactly TARGET_FRAMES frames"""
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()
    
    if not frames:
        return False
    
    if len(frames) < TARGET_FRAMES:
        last_frame = frames[-1]
        padding = [last_frame] * (TARGET_FRAMES - len(frames))
        frames.extend(padding)
    elif len(frames) > TARGET_FRAMES:
        frames = frames[:TARGET_FRAMES]
        
    frames_tensor = tf.stack(frames)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(tf.cast(frames_tensor, tf.float32))
    return tf.cast((frames_tensor - mean), tf.float32) / std

def load_alignments(path: str) -> tf.Tensor:
    """Load and process text alignments"""
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if len(line) >= 3 and line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def CTCLoss(y_true, y_pred):
    """Custom CTC Loss function"""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def create_heatmap(frame, model, frame_idx):
    """Create a heatmap visualization for a single frame"""
    # Get the model's intermediate layer output
    intermediate_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('conv3d_2').output
    )
    
    # Get the activation map
    activation = intermediate_model.predict(tf.expand_dims(frame, axis=0))
    
    # Average across channels
    heatmap = np.mean(activation[0, frame_idx], axis=-1)
    
    # Normalize heatmap
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    
    return heatmap

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

def main():
    st.title("Lip Reading Model Demo")
    st.write("Select a video from the dataset to test the lip reading model")

    # Load the model
    model_path = 'models/model_epoch_30.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please ensure the model is saved at 'models/model_epoch_30.h5'")
        return

    try:
        # First try to load the model weights
        model = build_model()
        model.load_weights(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Attempting to rebuild model from scratch...")
        try:
            model = build_model()
            model.compile(optimizer='adam', loss=CTCLoss)
            st.success("Model rebuilt successfully!")
        except Exception as e:
            st.error(f"Failed to rebuild model: {str(e)}")
            return

    # Get list of available videos
    video_dir = 'data/s1'
    if not os.path.exists(video_dir):
        st.error("Dataset directory not found. Please ensure the dataset is in the correct location.")
        return

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mpg')]
    
    if not video_files:
        st.error("No video files found in the dataset directory.")
        return

    selected_video = st.selectbox("Select a video file:", video_files)
    
    if selected_video:
        video_path = os.path.join(video_dir, selected_video)
        alignment_path = os.path.join('data/alignments/s1', selected_video.replace('.mpg', '.align'))

        # Create two columns for video and heatmap
        col1, col2 = st.columns([2, 1])

        with col1:
            # Display playable video
            st.subheader("Video Preview")
            st.info('🎥 The video below displays the converted video in MP4 format')
            
            # Convert to MP4 format
            converted_video_path = "temp_video.mp4"
            convert_to_mp4(video_path, converted_video_path)
            
            # Show video in Streamlit
            with open(converted_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)

            # Load and process the video
            frames = load_video(video_path)
            if frames is False:
                st.error("Error loading video file")
                return

            # Add batch dimension for prediction
            frames_batch = tf.expand_dims(frames, axis=0)
            
            # Make prediction
            yhat = model.predict(frames_batch)
            
            # Decode prediction
            input_length = tf.ones(shape=(1,), dtype="int64") * tf.cast(tf.shape(yhat)[1], dtype="int64")
            decoded = tf.keras.backend.ctc_decode(yhat, input_length, greedy=False)[0][0].numpy()
            predicted_text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode('utf-8')

            # Load actual alignment
            actual_alignment = load_alignments(alignment_path)
            actual_text = tf.strings.reduce_join(num_to_char(actual_alignment)).numpy().decode('utf-8')

            # Display results
            st.subheader("Results")
            st.write("Actual Text:", actual_text)
            st.write("Predicted Text:", predicted_text)

        with col2:
            # Create heatmap visualization
            st.subheader("Lip Region Heatmap")
            frame_idx = st.slider("Select frame to visualize:", 0, TARGET_FRAMES-1, TARGET_FRAMES//2)
            
            # Get the frame and create heatmap
            frame = frames[frame_idx]
            heatmap = create_heatmap(frames, model, frame_idx)
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
            
            # Original frame
            ax1.imshow(frame[:, :, 0], cmap='gray')
            ax1.set_title('Original Frame')
            ax1.axis('off')
            
            # Heatmap
            ax2.imshow(heatmap, cmap='hot')
            ax2.set_title('Activation Heatmap')
            ax2.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main() 