import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from config import *

class FeatureExtractor:
    def __init__(self, model_type=ModelType.RESNET50):
        self.model_type = model_type
        self.model = self._build_feature_extractor()
        
    def _build_feature_extractor(self):
        """Build feature extractor model"""
        input_shape = FACE_DETECTION_SIZE + (3,)
        inputs = Input(shape=input_shape)
        
        if self.model_type == ModelType.RESNET50:
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif self.model_type == ModelType.EFFICIENTNET:
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:  # Custom CNN
            x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu')(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            return Model(inputs=inputs, outputs=x)
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dense(FEATURE_DIM, activation='relu')(x)
        
        return Model(inputs=inputs, outputs=x)
    
    def extract_features(self, frames):
        """Extract features from video frames"""
        if len(frames.shape) == 3:  # Single frame
            frames = np.expand_dims(frames, axis=0)
            
        # Preprocess input based on model type
        if self.model_type == ModelType.RESNET50:
            processed_frames = tf.keras.applications.resnet50.preprocess_input(frames)
        elif self.model_type == ModelType.EFFICIENTNET:
            processed_frames = tf.keras.applications.efficientnet.preprocess_input(frames)
        else:
            processed_frames = frames / 255.0
            
        # Extract features
        features = self.model.predict(processed_frames)
        return features
    
    def extract_video_features(self, video_frames):
        """Extract features for all frames in a video"""
        frame_features = []
        for frame in video_frames:
            features = self.extract_features(frame)
            frame_features.append(features)
            
        # Aggregate features across time (mean pooling)
        video_features = np.mean(frame_features, axis=0)
        return video_features