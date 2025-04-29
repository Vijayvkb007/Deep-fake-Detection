import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from config import *

class DeepfakeVisualizer:
    def __init__(self, feature_extractor, classifier):
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATION_PATH}/training_history.png")
        plt.close()
        
    def generate_gradcam(self, frame, layer_name=None):
        """Generate Grad-CAM heatmap for a frame"""
        # Create a model that maps the input image to the activations
        # of the last conv layer and the output predictions
        if layer_name is None:
            if self.feature_extractor.model_type == ModelType.RESNET50:
                layer_name = "conv5_block3_out"
            elif self.feature_extractor.model_type == ModelType.EFFICIENTNET:
                layer_name = "top_activation"
            else:  # Custom CNN
                layer_name = "conv2d_3"
                
        grad_model = Model(
            inputs=[self.feature_extractor.model.inputs],
            outputs=[self.feature_extractor.model.get_layer(layer_name).output,
                    self.classifier.model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(frame, axis=0))
            loss = predictions[:, 0]
            
        # Extract filters and gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool gradients across the channels dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array
        # by "how important this channel is" with regard to the predicted class
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization purpose
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, FACE_DETECTION_SIZE)
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed_img = cv2.addWeighted(
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), 0.6,
            heatmap, 0.4, 0
        )
        
        return superimposed_img
    
    def visualize_eva_effect(self, original_frames, processed_frames, n_frames=3):
        """Visualize the effect of EVA processing"""
        plt.figure(figsize=(15, 5 * n_frames))
        
        for i in range(n_frames):
            idx = i * len(original_frames) // n_frames
            
            plt.subplot(n_frames, 2, 2*i+1)
            plt.imshow(original_frames[idx])
            plt.title(f"Original Frame {idx}")
            plt.axis('off')
            
            plt.subplot(n_frames, 2, 2*i+2)
            plt.imshow(processed_frames[idx])
            plt.title(f"Processed Frame {idx}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATION_PATH}/eva_effect.png")
        plt.close()