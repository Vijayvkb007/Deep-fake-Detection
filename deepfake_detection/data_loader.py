import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import os
from config import *

class DeepfakeDataLoader:
    def __init__(self):
        self.detector = MTCNN()
        
    def _detect_face(self, frame):
        """Detect and crop face using MTCNN"""
        results = self.detector.detect_faces(frame)
        if not results:
            return None
        
        # Get face with highest confidence
        best_face = max(results, key=lambda x: x['confidence'])
        if best_face['confidence'] < MIN_FACE_CONFIDENCE:
            return None
            
        # Get bounding box
        x, y, w, h = best_face['box']
        x, y = max(0, x), max(0, y)
        
        # Crop and resize face
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, FACE_DETECTION_SIZE)
        return face
    
    def load_video_frames(self, video_path, max_frames=100):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % FRAME_SAMPLE_RATE == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = self._detect_face(frame)
                if face is not None:
                    frames.append(face)
            
            frame_count += 1
            
        cap.release()
        return np.array(frames)
    
    def load_dataset(self, dataset_path=DATASET_PATH):
        """Load entire dataset"""
        real_videos = []
        fake_videos = []
        
        # Assuming dataset structure: dataset/{real,fake}/video_files
        # need to change this to the actual dataset structure
        real_dir = os.path.join(dataset_path, "real")
        fake_dir = os.path.join(dataset_path, "fake")
        
        print("Loading real videos...")
        for video_file in tqdm(os.listdir(real_dir)):
            video_path = os.path.join(real_dir, video_file)
            frames = self.load_video_frames(video_path)
            if len(frames) > 0:
                real_videos.append(frames)
        
        print("Loading fake videos...")
        for video_file in tqdm(os.listdir(fake_dir)):
            video_path = os.path.join(fake_dir, video_file)
            frames = self.load_video_frames(video_path)
            if len(frames) > 0:
                fake_videos.append(frames)
                
        return real_videos, fake_videos