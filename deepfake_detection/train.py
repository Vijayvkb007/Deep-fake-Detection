import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import DeepfakeDataLoader
from eva_processor import EulerianVideoAmplification
from feature_extractor import FeatureExtractor
from classifier import DeepfakeClassifier
from visualization import DeepfakeVisualizer
from config import *
import os

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    # Create necessary directories
    create_directory(MODEL_SAVE_PATH)
    create_directory(VISUALIZATION_PATH)
    
    # Step 1: Load and preprocess data
    print("Loading dataset...")
    data_loader = DeepfakeDataLoader()
    real_videos, fake_videos = data_loader.load_dataset()
    
    # Create labels (0=real, 1=fake)
    X = real_videos + fake_videos
    y = np.array([0]*len(real_videos) + [1]*len(fake_videos))
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-TRAIN_RATIO, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Step 2: Process videos with EVA
    print("Processing videos with EVA...")
    eva_processor = EulerianVideoAmplification()
    
    def process_videos(videos):
        processed = []
        for video in videos:
            processed_video = eva_processor.process_video(video)
            processed.append(processed_video)
        return processed
    
    X_train_processed = process_videos(X_train)
    X_val_processed = process_videos(X_val)
    X_test_processed = process_videos(X_test)
    
    # Step 3: Extract features
    print("Extracting features...")
    feature_extractor = FeatureExtractor(model_type=ModelType.RESNET50)
    
    def extract_features(videos):
        features = []
        for video in videos:
            video_features = feature_extractor.extract_video_features(video)
            features.append(video_features)
        return np.vstack(features)
    
    X_train_features = extract_features(X_train_processed)
    X_val_features = extract_features(X_val_processed)
    X_test_features = extract_features(X_test_processed)
    
    # Step 4: Train classifier
    print("Training classifier...")
    classifier = DeepfakeClassifier()
    history = classifier.train(X_train_features, y_train, X_val_features, y_val)
    
    # Step 5: Evaluate
    print("Evaluating model...")
    test_loss, test_acc, test_auc = classifier.evaluate(X_test_features, y_test)
    print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    
    # Step 6: Visualize results
    print("Generating visualizations...")
    visualizer = DeepfakeVisualizer(feature_extractor, classifier)
    
    # Plot training history
    visualizer.plot_training_history(history)
    
    # Visualize EVA effect
    sample_idx = np.random.randint(len(X_train))
    visualizer.visualize_eva_effect(X_train[sample_idx], X_train_processed[sample_idx])
    
    # Generate Grad-CAM examples
    sample_frame = X_test_processed[0][0]  # First frame of first test video
    gradcam_img = visualizer.generate_gradcam(sample_frame)
    cv2.imwrite(f"{VISUALIZATION_PATH}/gradcam_example.png", gradcam_img)

if __name__ == "__main__":
    main()