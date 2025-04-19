import numpy as np
import cv2
from scipy import fftpack
from config import *

class EulerianVideoAmplification:
    def __init__(self):
        self.levels = EVA_LEVELS
        self.alpha = EVA_AMPLIFICATION_FACTOR
        self.freq_min = EVA_FREQUENCY_MIN
        self.freq_max = EVA_FREQUENCY_MAX
        
    def _build_gaussian_pyramid(self, frame):
        """Build Gaussian pyramid for spatial processing"""
        pyramid = [frame]
        for _ in range(self.levels - 1):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)
        return pyramid
    
    def _reconstruct_frame(self, pyramid):
        """Reconstruct frame from Gaussian pyramid"""
        current = pyramid[-1]
        for i in range(len(pyramid)-2, -1, -1):
            current = cv2.pyrUp(current, dstsize=(
                pyramid[i].shape[1], pyramid[i].shape[0]))
            current = cv2.add(current, pyramid[i])
        return current
    
    def _temporal_bandpass_filter(self, frames):
        """Apply temporal bandpass filter"""
        fft = fftpack.fft(frames, axis=0)
        frequencies = fftpack.fftfreq(len(frames))
        
        # Create bandpass mask
        mask = (np.abs(frequencies) >= self.freq_min) & (
            np.abs(frequencies) <= self.freq_max)
        
        # Apply mask
        fft[~mask] = 0
        return fftpack.ifft(fft, axis=0).real
    
    def amplify(self, video_frames):
        """Amplify subtle motions in video"""
        if len(video_frames) < 2:
            return video_frames
            
        # Convert to grayscale for motion processing
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
                      for frame in video_frames]
        gray_frames = np.array(gray_frames)
        
        # Apply temporal bandpass filter
        filtered = self._temporal_bandpass_filter(gray_frames)
        
        # Process each frame
        amplified_frames = []
        for i, frame in enumerate(video_frames):
            # Build pyramid for current frame
            pyramid = self._build_gaussian_pyramid(frame)
            
            # Amplify each level
            for level in range(self.levels):
                # Get corresponding filtered pyramid level
                filtered_level = cv2.pyrDown(filtered[i]) if level > 0 else filtered[i]
                for _ in range(level - 1):
                    filtered_level = cv2.pyrDown(filtered_level)
                
                # Amplify motion
                pyramid[level] = pyramid[level] + self.alpha * filtered_level
            
            # Reconstruct frame
            amplified_frame = self._reconstruct_frame(pyramid)
            amplified_frames.append(amplified_frame)
            
        return np.array(amplified_frames)
    
    def process_video(self, video_frames):
        """Process video with EVA and return amplified frames"""
        # Apply EVA
        amplified_frames = self.amplify(video_frames)
        
        # Additional processing: edge enhancement
        processed_frames = []
        for frame in amplified_frames:
            # Convert to LAB color space for edge enhancement
            lab = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge((l, a, b))
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            processed_frames.append(enhanced_rgb)
            
        return np.array(processed_frames)