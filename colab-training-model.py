# Install required packages
#!pip install opencv-python matplotlib imageio gdown tensorflow

import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List, Tuple
from matplotlib import pyplot as plt
import imageio
import gdown
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation
from tensorflow.keras.layers import Reshape, BatchNormalization, TimeDistributed, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


# 1. Download and extract dataset
url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'
gdown.download(url, output, quiet=False)
gdown.extractall('data.zip')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Target number of frames - we'll pad or trim to this length
TARGET_FRAMES = 75

# 2. Build Data Loading Functions with standardized frame count
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
        print(f"Warning: No frames found in {path}")
        return False
    
    # Standardize frame count to TARGET_FRAMES
    if len(frames) < TARGET_FRAMES:
        # If we have fewer frames, duplicate the last frame
        last_frame = frames[-1]
        padding = [last_frame] * (TARGET_FRAMES - len(frames))
        frames.extend(padding)
    elif len(frames) > TARGET_FRAMES:
        # If we have more frames, trim to TARGET_FRAMES
        frames = frames[:TARGET_FRAMES]
        
    # Stack frames and normalize
    frames_tensor = tf.stack(frames)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(tf.cast(frames_tensor, tf.float32))
    return tf.cast((frames_tensor - mean), tf.float32) / std

# Set up vocabulary for text processing
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

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

def load_data(path: str):
    """Load both video and alignment data"""
    try:
        path = bytes.decode(path.numpy())
        file_name = os.path.basename(path).split('.')[0]
        
        video_path = os.path.join('data', 's1', f'{file_name}.mpg')
        alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')
        
        if not os.path.exists(alignment_path) or not os.path.exists(video_path):
            print(f"Warning: Files not found for {file_name}. Skipping.")
            return False
        
        frames = load_video(video_path)
        if frames is False:
            return False
            
        alignments = load_alignments(alignment_path)
        if alignments is None:
            return False
            
        # Validate frame shape
        frame_shape = tf.shape(frames)
        if frame_shape[0] != TARGET_FRAMES:
            print(f"Warning: Unexpected frame count: {frame_shape[0]} for {file_name}. Skipping.")
            return False
            
        return frames, alignments
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return False

# 3. Create improved Data Pipeline with shape validation
def prepare_dataset():
    # List all video files
    all_videos = list(tf.data.Dataset.list_files('./data/s1/*.mpg').as_numpy_iterator())
    print(f"Found {len(all_videos)} video files")

    # Process videos one by one to avoid None value issues
    valid_data = []

    for i, video_path in enumerate(all_videos):
        if i % 100 == 0:
            print(f"Processing video {i}/{len(all_videos)}...")

        result = load_data(tf.convert_to_tensor(video_path))
        if result != False:
            # Check frame shapes before adding to valid data
            frames, alignments = result
            if isinstance(frames, tf.Tensor) and len(frames.shape) == 4:
                frame_count = frames.shape[0]
                if frame_count == TARGET_FRAMES:
                    valid_data.append((frames, alignments))

    print(f"Successfully loaded {len(valid_data)} valid video-alignment pairs")

    if not valid_data:
        raise ValueError("No valid data found. Check your data paths and files.")

    # Create dataset from valid data
    frames = [item[0] for item in valid_data]
    alignments = [item[1] for item in valid_data]

    # Validate frame shapes before creating the dataset
    frame_shapes = [frame.shape for frame in frames]
    print(f"Frame shapes: {set(frame_shapes)}")

     # Pad alignments to a uniform length
    max_alignment_length = max(len(alignment) for alignment in alignments)
    alignments = [tf.pad(alignment, [[0, max_alignment_length - len(alignment)]]) for alignment in alignments]  # Padding applied to all alignments

    # Split into train and test sets first
    train_size = int(len(valid_data) * 0.9)
    train_frames = frames[:train_size]
    train_alignments = alignments[:train_size]  # Using padded alignments
    test_frames = frames[train_size:]
    test_alignments = alignments[train_size:]  # Using padded alignments for testing as well


    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_frames, train_alignments))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_frames, test_alignments))

    # Batch after splitting
    batch_size = 2
    train_dataset = train_dataset.shuffle(buffer_size=min(len(train_frames), 100))
    train_dataset = train_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Add repeat only to training dataset
    train_dataset = train_dataset.repeat()

    # Add prefetching
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, train_size

# 4. Design the Deep Neural Network
def build_model():
    # Input layer with explicit TARGET_FRAMES
    input_layer = Input(shape=(TARGET_FRAMES, 46, 140, 1))
    
    # CNN layers
    x = Conv3D(128, 3, padding='same')(input_layer)
    x = Activation('relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    x = Conv3D(256, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPool3D((1, 2, 2))(x)
    
    x = Conv3D(75, 3, padding='same')(x)
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

# 5. Define loss and callbacks
def CTCLoss(y_true, y_pred):
    """Custom CTC Loss function"""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

def scheduler(epoch, lr):
    """Learning rate scheduler"""
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

class ProduceExample(tf.keras.callbacks.Callback):
    """Callback to show prediction examples after each epoch"""
    def _init_(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()
    
    def on_epoch_end(self, epoch, logs=None) -> None:
        try:
            data = next(self.dataset)
            yhat = self.model.predict(data[0])
            decoded = tf.keras.backend.ctc_decode(yhat, [TARGET_FRAMES, TARGET_FRAMES], greedy=False)[0][0].numpy()
            for x in range(len(yhat)):
                print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
                print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
                print('~' * 100)
        except (StopIteration, tf.errors.OutOfRangeError):
            print("No more validation data available for examples")
        except Exception as e:
            print(f"Error in ProduceExample callback: {e}")

# 6. Train the model
def train_model():
    print("Preparing dataset...")
    train, test, train_size = prepare_dataset()
    
    print("Building model...")
    model = build_model()
    model.summary()
    
    print("Compiling model...")
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        os.path.join('models', 'model_epoch_{epoch:02d}.h5'),
        monitor='loss',
        save_best_only=True,
        save_freq='epoch',
        verbose=1
    )
    
    schedule_callback = LearningRateScheduler(scheduler)
    example_callback = ProduceExample(test)
    
    # Calculate steps per epoch
    steps_per_epoch = train_size // 2  # Batch size is 2
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    
    print(f"Training with {steps_per_epoch} steps per epoch...")
    history = model.fit(
        train, 
        validation_data=test, 
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_steps=10,
        callbacks=[checkpoint_callback, schedule_callback, example_callback],
        verbose=1
    )
    
    # Save final model
    final_model_path = os.path.join('models', 'final_lip_reading_model.h5')
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, history

# Run the training
if _name_ == "_main_":
    print("Starting lip reading model training...")
    model, history = train_model()
    print("Training completed!")